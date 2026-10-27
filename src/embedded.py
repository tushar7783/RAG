from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents

# class EmbeddingPipeline:
#     def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.model = SentenceTransformer(model_name)
#         print(f"[INFO] Loaded embedding model: {model_name}")

#     # def __init__(self,model_name: str = "nomic-embed-text",chunk_size: int = 1000,chunk_overlap: int = 20 ):
#     #     self.chunk_size = chunk_size
# #       self.chunk_overlap = chunk_overlap
# #       self.model_name = model_name

# #  try:
# #             response = ollama.embed(
# #                 model=self.model_name,
# #                 input="Test embedding"
# #             )
# #             dimension = len(response.embeddings[0])
# #             print(
# #                 f"[INFO] Loaded embedding model: {model_name}"
# #                 f" | Dimension: {dimension}"
# #             )
# #         except Exception as e:
# #             raise RuntimeError(
# #                 f"Could not initialize Ollama model '{model_name}'. "
# #                 "Ensure Ollama is running and the model is downloaded."
# #             ) from e

#     def chunk_documents(self, documents: List[Any]) -> List[Any]:
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=self.chunk_size,
#             chunk_overlap=self.chunk_overlap,
#             length_function=len,
#             separators=["\n\n", "\n", " ", ""]
#         )
#         chunks = splitter.split_documents(documents)
#         print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
#         return chunks

#     def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
#         texts = [chunk.page_content for chunk in chunks]
#         print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
#         embeddings = self.model.encode(texts, show_progress_bar=True)
#         print(f"[INFO] Embeddings shape: {embeddings.shape}")
#         return embeddings




from typing import List, Any

import numpy as np
import ollama
# from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.data_loader import load_all_documents


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name

        # Verify that Ollama can generate embeddings
        try:
            response = ollama.embed(
                model=self.model_name,
                input="Test embedding"
            )
            dimension = len(response.embeddings[0])
            print(
                f"[INFO] Loaded embedding model: {model_name}"
                f" | Dimension: {dimension}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not initialize Ollama model '{model_name}'. "
                "Ensure Ollama is running and the model is downloaded."
            ) from e

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(documents)

        print(
            f"[INFO] Split {len(documents)} documents "
            f"into {len(chunks)} chunks."
        )
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        if not chunks:
            return np.empty((0, 768), dtype=np.float32)

        texts = [chunk.page_content for chunk in chunks]

        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")

        # Ollama accepts a list of texts as input
        response = ollama.embed(
            model=self.model_name,
            input=texts
        )

        embeddings = np.asarray(
            response.embeddings,
            dtype=np.float32
        )

        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings



# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)