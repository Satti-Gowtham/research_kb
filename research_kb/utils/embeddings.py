import logging
import aiohttp
from typing import List
import numpy as np
from scipy.spatial.distance import cosine
import math

logger = logging.getLogger(__name__)

class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", url: str = "http://localhost:11434"):
        self.model = model
        self.url = url
        self.api_url = f"{url}/api/embeddings"
        logger.info(f"Initialized OllamaEmbedder with model: {model}, API URL: {self.api_url}")

    async def embed_text(self, text: str) -> List[float]:
        """Create embedding for a single text"""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * 768  # Default embedding dimension
            
        try:
            logger.info(f"Embedding text: {text[:100]}...")
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": text
                }
                
                async with session.post(self.api_url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error ({response.status}): {error_text}")
                        return [0.0] * 768
                    
                    result = await response.json()
                    if "embedding" not in result:
                        logger.error(f"No embedding in response. Keys: {list(result.keys())}")
                        return [0.0] * 768
                        
                    return result["embedding"]
                    
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            return [0.0] * 768

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float], query=None, text=None) -> float:
        """Calculate similarity using both cosine similarity and keyword matching"""
        if not embedding1 or not embedding2 or len(embedding1) != len(embedding2):
            return 0.0
        
        try:
            try:
                embedding1 = [float(x) for x in embedding1]
                embedding2 = [float(x) for x in embedding2]
                
                dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
                mag1 = math.sqrt(sum(a * a for a in embedding1))
                mag2 = math.sqrt(sum(b * b for b in embedding2))
                
                if mag1 < 1e-10 or mag2 < 1e-10:
                    return 0.0
                    
                cosine_sim = dot_product / (mag1 * mag2)
                cosine_sim = max(-1.0, min(1.0, cosine_sim))
                
            except (TypeError, ValueError):
                return 0.0
            
            if cosine_sim <= 0:
                semantic_score = 0.0
            else:
                shifted = (cosine_sim - 0.7) * 5
                semantic_score = 1 / (1 + math.exp(-shifted))
            
            keyword_boost = 0.0
            if query and text:
                query_words = {w.strip('?.,!') for w in query.lower().split() if len(w.strip('?.,!')) > 3}
                text_words = {w.strip('?.,!') for w in text.lower().split() if len(w.strip('?.,!')) > 3}
                
                common_words = query_words.intersection(text_words)
                if common_words and query_words:
                    overlap_ratio = len(common_words) / len(query_words)
                    if overlap_ratio >= 0.5:
                        keyword_boost = overlap_ratio * 0.3
            
            final_score = semantic_score * 0.8 + keyword_boost
            return min(1.0, final_score)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0 