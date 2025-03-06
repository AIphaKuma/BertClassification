from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os

class CapitalRAG:
    def __init__(self, docs_path="data/my_docs.txt"):
        self.docs_path = docs_path
        self.documents = self._load_documents()

        self.text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        self.splits = self.text_splitter.split_documents(self.documents)

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = FAISS.from_documents(self.splits, self.embeddings)

    def _load_documents(self):
        if not os.path.exists(self.docs_path):
            raise FileNotFoundError(f"Le fichier {self.docs_path} n'existe pas.")

        loader = TextLoader(self.docs_path)
        return loader.load()

    def retrieve(self, query, top_k=1):
        docs = self.db.similarity_search(query, k=top_k)
        return [doc.page_content for doc in docs]