"""Vector store memory implementation for the Trip Agent system."""

import os
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field


class VectorStoreMemory(BaseModel):
    """Vector store memory for storing and retrieving information."""
    
    collection_name: str = Field(description="Name of the collection in the vector store")
    persist_directory: str = Field(description="Directory to persist the vector store")
    embeddings: Any = Field(description="Embeddings model to use")
    vector_store: Optional[Any] = Field(None, description="Vector store instance")
    
    class Config:
        """Pydantic config."""
        
        arbitrary_types_allowed = True
    
    def __init__(
        self,
        collection_name: str,
        persist_directory: Optional[str] = None,
        embeddings: Optional[Any] = None,
        **data
    ):
        """Initialize the vector store memory.
        
        Args:
            collection_name: Name of the collection in the vector store.
            persist_directory: Directory to persist the vector store.
            embeddings: Embeddings model to use.
            **data: Additional data for the memory.
        """
        # Set default values
        persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIRECTORY", "./chroma_db"
        )
        embeddings = embeddings or OpenAIEmbeddings()
        
        # Initialize the vector store
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        super().__init__(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embeddings=embeddings,
            vector_store=vector_store,
            **data
        )
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add texts to the vector store.
        
        Args:
            texts: List of texts to add.
            metadatas: Optional list of metadata for each text.
            
        Returns:
            List[str]: List of IDs of the added texts.
        """
        return self.vector_store.add_texts(texts, metadatas)
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents to add.
            
        Returns:
            List[str]: List of IDs of the added documents.
        """
        return self.vector_store.add_documents(documents)
    
    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents in the vector store.
        
        Args:
            query: Query text to search for.
            k: Number of results to return.
            filter: Optional filter to apply to the search.
            
        Returns:
            List[Document]: List of similar documents.
        """
        return self.vector_store.similarity_search(query, k=k, filter=filter)
    
    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """Search for similar documents in the vector store with relevance scores.
        
        Args:
            query: Query text to search for.
            k: Number of results to return.
            filter: Optional filter to apply to the search.
            
        Returns:
            List[tuple[Document, float]]: List of similar documents with relevance scores.
        """
        return self.vector_store.similarity_search_with_score(query, k=k, filter=filter)
    
    def persist(self) -> None:
        """Persist the vector store to disk."""
        self.vector_store.persist()
    
    def delete_collection(self) -> None:
        """Delete the collection from the vector store."""
        self.vector_store.delete_collection()
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )