"""
Text Processing Module
Handles text preprocessing, cleaning, and normalization for BERT embeddings.
"""

import re
import string
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import streamlit as st


class TextProcessor:
    """Text preprocessing and cleaning utilities"""
    
    def __init__(self):
        """Initialize the text processor"""
        self._download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text
        
        Args:
            text: Input text
            
        Returns:
            Text with stopwords removed
        """
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words]
        return ' '.join(filtered_tokens)
    
    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatize text
        
        Args:
            text: Input text
            
        Returns:
            Lemmatized text
        """
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)
    
    def preprocess_text(self, text: str, remove_stopwords: bool = False, lemmatize: bool = False) -> str:
        """
        Complete text preprocessing pipeline
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize
            
        Returns:
            Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Remove stopwords if requested
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        # Lemmatize if requested
        if lemmatize:
            text = self.lemmatize_text(text)
        
        return text
    
    def preprocess_texts(self, texts: List[str], remove_stopwords: bool = False, lemmatize: bool = False) -> List[str]:
        """
        Preprocess a list of texts
        
        Args:
            texts: List of input texts
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess_text(text, remove_stopwords, lemmatize) for text in texts]
    
    def get_text_statistics(self, texts: List[str]) -> dict:
        """
        Get statistics about the texts
        
        Args:
            texts: List of texts
            
        Returns:
            Dictionary containing text statistics
        """
        if not texts:
            return {}
        
        # Calculate statistics
        total_texts = len(texts)
        total_chars = sum(len(text) for text in texts)
        total_words = sum(len(text.split()) for text in texts)
        
        # Length statistics
        text_lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        stats = {
            'total_texts': total_texts,
            'total_characters': total_chars,
            'total_words': total_words,
            'avg_text_length': total_chars / total_texts,
            'avg_word_count': total_words / total_texts,
            'min_text_length': min(text_lengths),
            'max_text_length': max(text_lengths),
            'min_word_count': min(word_counts),
            'max_word_count': max(word_counts)
        }
        
        return stats
    
    def filter_texts_by_length(self, texts: List[str], min_length: int = 10, max_length: int = 1000) -> List[str]:
        """
        Filter texts by length
        
        Args:
            texts: List of texts
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            Filtered list of texts
        """
        return [text for text in texts if min_length <= len(text) <= max_length]
    
    def remove_duplicates(self, texts: List[str]) -> List[str]:
        """
        Remove duplicate texts
        
        Args:
            texts: List of texts
            
        Returns:
            List with duplicates removed
        """
        return list(dict.fromkeys(texts))
    
    def validate_texts(self, texts: List[str]) -> tuple[List[str], List[str]]:
        """
        Validate texts and return valid and invalid texts
        
        Args:
            texts: List of texts
            
        Returns:
            Tuple of (valid_texts, invalid_texts)
        """
        valid_texts = []
        invalid_texts = []
        
        for text in texts:
            if text and isinstance(text, str) and len(text.strip()) > 0:
                valid_texts.append(text.strip())
            else:
                invalid_texts.append(text)
        
        return valid_texts, invalid_texts 