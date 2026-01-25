use crate::storage::Vector;
use anyhow::{Context, Result};
use async_channel::Sender;
use aws_config::meta::region::RegionProviderChain;
use aws_config::BehaviorVersion;
use aws_sdk_s3::{config::Credentials, config::Region, Client};
use std::thread;
use tokio::io::{AsyncRead, AsyncReadExt};

const HEADER_BYTES: usize = 12;
const MAX_READ_RETRIES: usize = 3;

async fn open_body_at(
    client: &Client,
    bucket: &str,
    key: &str,
    offset: u64,
) -> Result<Box<dyn AsyncRead + Unpin + Send>> {
    let range = format!("bytes={}-", offset);
    let object = client
        .get_object()
        .bucket(bucket)
        .key(key)
        .range(range)
        .send()
        .await
        .context("failed to get object range from s3")?;

    Ok(Box::new(object.body.into_async_read()))
}

pub fn spawn_s3_reader(
    endpoint: String,
    bucket: String,
    key: String,
    access_key: String,
    secret_key: String,
    batch_size: usize,
) -> async_channel::Receiver<Result<Vec<Vector>>> {
    let (tx, rx) = async_channel::bounded(10);

    thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to build tokio runtime");

        rt.block_on(async move {
            if let Err(e) = run_s3_stream(
                tx.clone(),
                endpoint,
                bucket,
                key,
                access_key,
                secret_key,
                batch_size,
            )
            .await
            {
                let _ = tx.send(Err(e)).await;
            }
        });
    });

    rx
}

async fn run_s3_stream(
    tx: Sender<Result<Vec<Vector>>>,
    endpoint: String,
    bucket: String,
    key: String,
    access_key: String,
    secret_key: String,
    batch_size: usize,
) -> Result<()> {
    let credentials = Credentials::new(access_key, secret_key, None, None, "static");
    let region_provider = RegionProviderChain::default_provider().or_else(Region::new("auto"));
    let config = aws_config::defaults(BehaviorVersion::latest())
        .region(region_provider)
        .endpoint_url(endpoint)
        .credentials_provider(credentials)
        .load()
        .await;

    let client = Client::new(&config);

    log::info!("Connecting to S3: bucket={}, key={}", bucket, key);

    let object = client
        .get_object()
        .bucket(bucket.as_str())
        .key(key.as_str())
        .send()
        .await
        .context("failed to get object from s3")?;

    let mut body: Box<dyn AsyncRead + Unpin + Send> = Box::new(object.body.into_async_read());

    // Read Header
    // dim: u32 (4 bytes)
    // count: u64 (8 bytes)
    let mut header = [0u8; HEADER_BYTES];
    body.read_exact(&mut header)
        .await
        .context("failed to read header")?;

    let dim = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let count = u64::from_le_bytes(header[4..12].try_into().unwrap()) as usize;

    log::info!("S3 Stream: dim={}, count={}", dim, count);

    let vector_size = dim * 4;
    let mut buffer = vec![0u8; batch_size * vector_size];
    let mut current_idx = 0;

    while current_idx < count {
        let remaining = count - current_idx;
        let this_batch = remaining.min(batch_size);
        let bytes_needed = this_batch * vector_size;

        let data_offset = HEADER_BYTES as u64 + (current_idx as u64 * vector_size as u64);
        let mut bytes_read = 0usize;
        let mut attempts = 0usize;
        while bytes_read < bytes_needed {
            match body.read(&mut buffer[bytes_read..bytes_needed]).await {
                Ok(0) => {
                    attempts += 1;
                    if attempts > MAX_READ_RETRIES {
                        return Err(anyhow::anyhow!(
                            "S3 stream stalled after {} retries at offset {}",
                            MAX_READ_RETRIES,
                            data_offset + bytes_read as u64
                        ));
                    }
                    log::warn!(
                        "S3 stream stalled at offset {}, retrying ({}/{})",
                        data_offset + bytes_read as u64,
                        attempts,
                        MAX_READ_RETRIES
                    );
                    body = open_body_at(&client, &bucket, &key, data_offset + bytes_read as u64)
                        .await?;
                }
                Ok(n) => {
                    bytes_read += n;
                    attempts = 0;
                }
                Err(err) => {
                    attempts += 1;
                    if attempts > MAX_READ_RETRIES {
                        return Err(err).context("failed to read batch from s3");
                    }
                    log::warn!(
                        "S3 stream read error at offset {}: {:?} (retry {}/{})",
                        data_offset + bytes_read as u64,
                        err,
                        attempts,
                        MAX_READ_RETRIES
                    );
                    body = open_body_at(&client, &bucket, &key, data_offset + bytes_read as u64)
                        .await?;
                }
            }
        }

        let mut vectors = Vec::with_capacity(this_batch);
        let mut offset = 0;
        for i in 0..this_batch {
            let mut data = Vec::with_capacity(dim);
            for _ in 0..dim {
                let bytes: [u8; 4] = buffer[offset..offset + 4].try_into().unwrap();
                data.push(f32::from_le_bytes(bytes));
                offset += 4;
            }
            let id = current_idx as u64 + i as u64;
            vectors.push(Vector::new(id, data));
        }

        if tx.send(Ok(vectors)).await.is_err() {
            log::warn!("S3 stream receiver dropped, stopping download.");
            break;
        }
        current_idx += this_batch;
    }

    log::info!("S3 Stream finished. Read {} vectors.", current_idx);

    Ok(())
}
