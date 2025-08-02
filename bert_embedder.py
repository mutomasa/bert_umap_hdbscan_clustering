"""
BERT Embedding Module
Handles BERT model loading and text embedding generation.
"""

import torch
import numpy as np
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import streamlit as st


class BERTEmbedder:
    """BERT-based text embedding generator"""
    
    def __init__(self):
        """Initialize the BERT embedder"""
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self, model_name: str) -> bool:
        """
        Load BERT model
        
        Args:
            model_name: Name of the BERT model to load
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if self.model_name == model_name and self.model is not None:
                return True
            
            with st.spinner(f"Loading BERT model: {model_name}"):
                if model_name.startswith('sentence-transformers/'):
                    # Use sentence-transformers for better sentence embeddings
                    self.model = SentenceTransformer(model_name, device=self.device)
                    self.model_name = model_name
                else:
                    # Use transformers for raw BERT
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModel.from_pretrained(model_name)
                    self.model.to(self.device)
                    self.model.eval()
                    self.model_name = model_name
            
            return True
            
        except Exception as e:
            st.error(f"Error loading model {model_name}: {str(e)}")
            return False
    
    def generate_embeddings(self, texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            model_name: BERT model name
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        # Load model if needed
        if not self.load_model(model_name):
            raise RuntimeError(f"Failed to load model: {model_name}")
        
        # Generate embeddings
        if isinstance(self.model, SentenceTransformer):
            # Use sentence-transformers
            embeddings = self.model.encode(texts, show_progress_bar=True)
        else:
            # Use raw BERT
            embeddings = self._generate_bert_embeddings(texts)
        
        return embeddings
    
    def _generate_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using raw BERT model
        
        Args:
            texts: List of texts
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate embeddings
                outputs = self.model(**inputs)
                
                # Use [CLS] token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding.flatten())
        
        return np.array(embeddings)
    
    def get_embedding_dimension(self) -> Optional[int]:
        """
        Get the dimension of the embeddings
        
        Returns:
            Embedding dimension or None if model not loaded
        """
        if self.model is None:
            return None
        
        if isinstance(self.model, SentenceTransformer):
            return self.model.get_sentence_embedding_dimension()
        else:
            return self.model.config.hidden_size
    
    def compute_similarity(self, embeddings: np.ndarray, method: str = "cosine") -> np.ndarray:
        """
        Compute similarity matrix between embeddings
        
        Args:
            embeddings: Numpy array of embeddings
            method: Similarity method ('cosine', 'euclidean', 'dot')
            
        Returns:
            Similarity matrix
        """
        if method == "cosine":
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norms
            
            # Compute cosine similarity
            similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
            
        elif method == "euclidean":
            # Compute pairwise euclidean distances
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(embeddings)
            # Convert distances to similarities (1 / (1 + distance))
            similarity_matrix = 1 / (1 + distances)
            
        elif method == "dot":
            # Compute dot product
            similarity_matrix = np.dot(embeddings, embeddings.T)
            
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        return similarity_matrix
    
    def find_most_similar(self, embeddings: np.ndarray, query_idx: int, top_k: int = 5) -> List[tuple]:
        """
        Find most similar texts to a query text
        
        Args:
            embeddings: Numpy array of embeddings
            query_idx: Index of the query text
            top_k: Number of most similar texts to return
            
        Returns:
            List of (index, similarity) tuples
        """
        similarity_matrix = self.compute_similarity(embeddings, method="cosine")
        
        # Get similarities for query text
        query_similarities = similarity_matrix[query_idx]
        
        # Get indices of top-k most similar texts (excluding self)
        similar_indices = np.argsort(query_similarities)[::-1][1:top_k+1]
        
        # Return (index, similarity) pairs
        return [(idx, query_similarities[idx]) for idx in similar_indices]
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        info = {
            "model_name": self.model_name,
            "device": str(self.device),
            "embedding_dimension": self.get_embedding_dimension()
        }
        
        if isinstance(self.model, SentenceTransformer):
            info["model_type"] = "SentenceTransformer"
        else:
            info["model_type"] = "BERT"
            info["vocab_size"] = self.model.config.vocab_size
            info["hidden_size"] = self.model.config.hidden_size
            info["num_layers"] = self.model.config.num_hidden_layers
        
        return info 