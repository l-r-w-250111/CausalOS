import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.ollama import OllamaEmbedding

class RAGHandler:
    def __init__(self, persist_dir="./storage"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        # Configure the Embedding model to use the CPU-based Ollama service.
        # The LLM is now configured dynamically in app.py based on user selection.
        EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "embeddinggemma")

        Settings.embed_model = OllamaEmbedding(
            base_url=os.getenv("OLLAMA_CPU_URL", "http://localhost:11435"),
            model_name=EMBEDDING_MODEL_NAME,
        )

        self.index = self._load_index()

    def _load_index(self):
        """Loads the index from storage if it exists, otherwise builds it."""
        try:
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            index = load_index_from_storage(storage_context)
            print("Loaded existing RAG index from storage.")
        except FileNotFoundError:
            print("No existing RAG index found. Building a new one.")
            index = self._build_index_from_documents()
        return index

    def _build_index_from_documents(self, documents_path="./data"):
        """Builds a new index from the documents directory and persists it."""
        os.makedirs(documents_path, exist_ok=True)
        documents = SimpleDirectoryReader(documents_path).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=self.persist_dir)
        print(f"New RAG index built and persisted to {self.persist_dir}")
        return index

    def add_document(self, file_path):
        """Adds a new document to the index and persists the changes."""
        document = SimpleDirectoryReader(input_files=[file_path]).load_data()
        self.index.insert(document[0])
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        print(f"Document '{os.path.basename(file_path)}' added to RAG index and persisted.")

    def add_text_to_rag(self, text: str):
        """Adds a text snippet to the index and persists the changes."""
        from llama_index.core.schema import Document
        doc = Document(text=text)
        self.index.insert(doc)
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        print("Text snippet added to RAG index and persisted.")

    def get_query_engine(self):
        return self.index.as_query_engine()

    def get_retriever(self):
        return self.index.as_retriever()
