from src.data_loader import load_all_documents
from src.embedded import EmbeddingPipeline
from src.vector_store import FaissVectorStore
from src.search import RAGSearch
# example 

# if __name__ =="__main__":
#     docs = load_all_documents("data")
#     print(f"Loaded {len(docs)} documents.")
#     print("Example document:", docs[0] if docs else None)
#     emb_pipe = EmbeddingPipeline()
#     chunks = emb_pipe.chunk_documents(docs)
#     embeddings = emb_pipe.embed_chunks(chunks)
#     print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
#     store = FaissVectorStore("faiss_store")
#     store.build_from_documents(docs)

#     store.load()

#     results = store.query(
#         "What is attention mechanism?",
#         top_k=3
#     )

#     for result in results:
#         print(result)
#     rag_search = RAGSearch()

#     query = "tell me about tushar "
#     summary = rag_search.search_and_summarize(query, top_k=3)

#     print("Answer:", summary)

if __name__ == "__main__":
    
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    #store.build_from_documents(docs)
    store.load()
    #print(store.query("What is attention mechanism?", top_k=3))
    rag_search = RAGSearch()
    query = "tell me about tushar?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)



