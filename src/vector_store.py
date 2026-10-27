# import os
# import faiss
# import numpy as np
# import pickle
# from typing import List, Any
# from sentence_transformers import SentenceTransformer
# from src.embedding import EmbeddingPipeline

# class FaissVectorStore:
#     def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
#         self.persist_dir = persist_dir
#         os.makedirs(self.persist_dir, exist_ok=True)
#         self.index = None
#         self.metadata = []
#         self.embedding_model = embedding_model
#         self.model = SentenceTransformer(embedding_model)
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         print(f"[INFO] Loaded embedding model: {embedding_model}")

#     def build_from_documents(self, documents: List[Any]):
#         print(f"[INFO] Building vector store from {len(documents)} raw documents...")
#         emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
#         chunks = emb_pipe.chunk_documents(documents)
#         embeddings = emb_pipe.embed_chunks(chunks)
#         metadatas = [{"text": chunk.page_content} for chunk in chunks]
#         self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
#         self.save()
#         print(f"[INFO] Vector store built and saved to {self.persist_dir}")

#     def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
#         dim = embeddings.shape[1]
#         if self.index is None:
#             self.index = faiss.IndexFlatL2(dim)
#         self.index.add(embeddings)
#         if metadatas:
#             self.metadata.extend(metadatas)
#         print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

#     def save(self):
#         faiss_path = os.path.join(self.persist_dir, "faiss.index")
#         meta_path = os.path.join(self.persist_dir, "metadata.pkl")
#         faiss.write_index(self.index, faiss_path)
#         with open(meta_path, "wb") as f:
#             pickle.dump(self.metadata, f)
#         print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

#     def load(self):
#         faiss_path = os.path.join(self.persist_dir, "faiss.index")
#         meta_path = os.path.join(self.persist_dir, "metadata.pkl")
#         self.index = faiss.read_index(faiss_path)
#         with open(meta_path, "rb") as f:
#             self.metadata = pickle.load(f)
#         print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

#     def search(self, query_embedding: np.ndarray, top_k: int = 5):
#         D, I = self.index.search(query_embedding, top_k)
#         results = []
#         for idx, dist in zip(I[0], D[0]):
#             meta = self.metadata[idx] if idx < len(self.metadata) else None
#             results.append({"index": idx, "distance": dist, "metadata": meta})
#         return results

#     def query(self, query_text: str, top_k: int = 5):
#         print(f"[INFO] Querying vector store for: '{query_text}'")
#         query_emb = self.model.encode([query_text]).astype('float32')
#         return self.search(query_emb, top_k=top_k)

# # Example usage
# if __name__ == "__main__":
#     from data_loader import load_all_documents
#     docs = load_all_documents("data")
#     store = FaissVectorStore("faiss_store")
#     store.build_from_documents(docs)
#     store.load()
#     print(store.query("What is attention mechanism?", top_k=3))


import os
import faiss
import numpy as np
import pickle
import ollama
from typing import List, Any
from src.embedded import EmbeddingPipeline


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Verify Ollama embedding model
        try:
            response = ollama.embed(
                model=self.embedding_model,
                input="Test embedding"
            )
            self.dimension = len(response.embeddings[0])
            print(
                f"[INFO] Ollama model ready: {embedding_model}, "
                f"dimension: {self.dimension}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot initialize Ollama model '{embedding_model}'. "
                "Check that Ollama is running and the model is installed."
            ) from e

    def build_from_documents(self, documents: List[Any]):
        print(
            f"[INFO] Building vector store from "
            f"{len(documents)} raw documents..."
        )

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = emb_pipe.chunk_documents(documents)

        if not chunks:
            raise ValueError("No document chunks were generated.")

        embeddings = emb_pipe.embed_chunks(chunks)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        metadatas = [
            {
                "text": chunk.page_content,
                "source": str(chunk.metadata.get("source", "unknown"))
            }
            for chunk in chunks
        ]

        self.add_embeddings(embeddings, metadatas)
        self.save()

        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: List[Any] = None
    ):
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)

        self.index.add(embeddings)

        if metadatas:
            if len(metadatas) != len(embeddings):
                raise ValueError(
                    "Metadata count must match embedding count."
                )
            self.metadata.extend(metadatas)

        print(f"[INFO] Added {len(embeddings)} vectors to FAISS.")

    def save(self):
        if self.index is None:
            raise ValueError("Cannot save an empty, uninitialized index.")

        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[INFO] Saved FAISS index to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        self.index = faiss.read_index(faiss_path)

        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        if self.index.d != self.dimension:
            raise ValueError(
                "Index dimension does not match the current embedding "
                "model. Rebuild the index with the same model."
            )

        if self.index.ntotal != len(self.metadata):
            raise ValueError("FAISS index and metadata counts do not match.")

        print(f"[INFO] Loaded {self.index.ntotal} vectors.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index is None:
            raise ValueError("Index not initialized. Build or load it first.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = np.asarray(
            query_embedding, dtype=np.float32
        ).reshape(1, -1)

        if query_embedding.shape[1] != self.index.d:
            raise ValueError("Query embedding dimension mismatch.")

        k = min(top_k, self.index.ntotal)
        if k == 0:
            return []

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0:
                continue

            results.append({
                "index": int(idx),
                "distance": float(distance),
                "metadata": self.metadata[idx]
            })

        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store: '{query_text}'")

        response = ollama.embed(
            model=self.embedding_model,
            input=query_text
        )

        query_emb = np.asarray(
            response.embeddings, dtype=np.float32
        )

        return self.search(query_emb, top_k=top_k)


# Example usage
if __name__ == "__main__":
    from src.data_loader import load_all_documents

    docs = load_all_documents("data")

    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)

    store.load()

    results = store.query(
        "What is attention mechanism?",
        top_k=3
    )

    for result in results:
        print(result)
