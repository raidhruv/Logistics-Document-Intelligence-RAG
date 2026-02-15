from src.ingestion import ingest_pdf

def run_basic_test():
    chunks = ingest_pdf("data/sample_logistics.pdf")

    assert len(chunks) > 0
    print("Ingestion working")

if __name__ == "__main__":
    run_basic_test()
