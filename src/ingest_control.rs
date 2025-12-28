use std::sync::atomic::{AtomicBool, Ordering};

static ALLOW_INGESTION: AtomicBool = AtomicBool::new(true);

pub fn ingestion_allowed() -> bool {
    ALLOW_INGESTION.load(Ordering::Acquire)
}

pub fn disallow_ingestion() {
    ALLOW_INGESTION.store(false, Ordering::Release);
}

pub fn allow_ingestion() {
    ALLOW_INGESTION.store(true, Ordering::Release);
}
