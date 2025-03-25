from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class SemanticChunker:
    def __init__(self, min_size: int = 512, max_size: int = 1024):
        self.min_size = min_size
        self.max_size = max_size

    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """Split text into semantic chunks"""
        if not text or not text.strip():
            return []

        # Split into sentences (rough approximation)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_start = 0

        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size > self.max_size and current_chunk:
                # Create a chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "start": chunk_start,
                    "end": chunk_start + len(chunk_text)
                })
                current_chunk = []
                current_size = 0
                chunk_start = chunk_start + len(chunk_text)

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "start": chunk_start,
                "end": chunk_start + len(chunk_text)
            })

        return chunks 