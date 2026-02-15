from src.rag_pipeline import answer_query

if __name__ == "__main__":
    pdf = "data/sample_logistics.pdf"
    query = "What is customs clearance in international logistics?"

    answer = answer_query(pdf, query)

    print("\n=== FINAL ANSWER ===\n")
    print(answer)
