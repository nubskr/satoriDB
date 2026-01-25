BIGANN_BASE_URL = ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz
BIGANN_QUERY_URL = ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
BIGANN_GND_URL = ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz

DATA_DIR ?= .
BASE = $(DATA_DIR)/bigann_base.bvecs.gz
QUERY = $(DATA_DIR)/bigann_query.bvecs.gz
GND = $(DATA_DIR)/bigann_gnd.tar.gz
BASE_F32 = $(DATA_DIR)/bigann_base.bvecs.f32bin

CARGO ?= cargo

.PHONY: bigann-download bigann-prepare benchmark clean-benchmark-data

bigann-download: warn-benchmark-space $(BASE) $(QUERY) $(GND)

warn-benchmark-space:
	@echo "WARNING: Benchmark assets (downloads + extracted) can exceed 1TB total disk usage."
	@echo "Ensure you have enough space before continuing."

$(BASE):
	curl -fL --progress-bar $(BIGANN_BASE_URL) -o $@

$(QUERY):
	curl -fL --progress-bar $(BIGANN_QUERY_URL) -o $@

$(GND):
	curl -fL --progress-bar $(BIGANN_GND_URL) -o $@

bigann-prepare: $(BASE_F32)

$(BASE_F32): $(BASE)
	$(CARGO) run --release --bin prepare_dataset -- $< $@

benchmark: bigann-download bigann-prepare
	ulimit -n 524288; WALRUS_QUIET=1 SATORI_REBALANCE_POLL_MS=0 SATORI_RUN_BENCH=1 $(CARGO) run --release --bin satoridb

clean-db:
	rm -rf wal_files vector_index bucket_index

benchmark-s3: clean-db $(QUERY) $(GND)
	@echo "Running benchmark with S3 streaming (requires .env with R2 credentials)..."
	ulimit -n 524288; WALRUS_QUIET=1 RUST_LOG=info SATORI_REBALANCE_POLL_MS=1000 SATORI_RUN_BENCH=1 $(CARGO) run --release --bin satoridb

clean-benchmark-data:
	rm -f $(BASE) $(BASE_F32) $(QUERY) $(GND)
