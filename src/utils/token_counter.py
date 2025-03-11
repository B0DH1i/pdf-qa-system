"""
Token sayımı ve optimizasyonu için yardımcı sınıf.
"""

import tiktoken
from typing import List, Dict, Any, Optional, Union
from loguru import logger

class TokenCounter:
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 max_tokens: int = 4096,
                 buffer_tokens: int = 500):
        """
        Token sayacı başlatma
        
        Args:
            model_name: Kullanılan model adı
            max_tokens: Maksimum token limiti
            buffer_tokens: Tampon token sayısı
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.buffer_tokens = buffer_tokens
        self.encoding = tiktoken.encoding_for_model(model_name)
        
    def count_tokens(self, text: Union[str, List[str], Dict[str, Any]]) -> int:
        """
        Metin veya metin koleksiyonundaki token sayısını hesapla
        
        Args:
            text: Token sayısı hesaplanacak metin(ler)
            
        Returns:
            Token sayısı
        """
        if isinstance(text, str):
            return len(self.encoding.encode(text))
        elif isinstance(text, list):
            return sum(self.count_tokens(item) for item in text)
        elif isinstance(text, dict):
            return sum(self.count_tokens(str(value)) for value in text.values())
        else:
            raise ValueError(f"Desteklenmeyen veri tipi: {type(text)}")
            
    def optimize_context(self, 
                        messages: List[Dict[str, Any]], 
                        new_message_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Optimize context window to stay within token limits.
        
        Args:
            messages: List of messages to optimize
            new_message_tokens: Token count of new message to be added
            
        Returns:
            Optimized message list
        """
        # Calculate total token count
        total_tokens = sum(self.count_tokens(msg) for msg in messages)
        if new_message_tokens:
            total_tokens += new_message_tokens
            
        # Return all messages if token limit is not exceeded
        available_tokens = self.max_tokens - self.buffer_tokens
        if total_tokens <= available_tokens:
            return messages
            
        # Optimize by removing oldest messages
        optimized_messages = []
        current_tokens = 0
        
        # Reserve space for new message
        if new_message_tokens:
            available_tokens -= new_message_tokens
            
        # Add messages from end to beginning
        for msg in reversed(messages):
            msg_tokens = self.count_tokens(msg)
            if current_tokens + msg_tokens <= available_tokens:
                optimized_messages.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
                
        logger.info(f"Bağlam penceresi optimize edildi: {len(messages)} -> {len(optimized_messages)} mesaj")
        return optimized_messages
        
    def get_token_stats(self, messages: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Token istatistiklerini hesapla
        
        Args:
            messages: Mesaj listesi
            
        Returns:
            Token istatistikleri
        """
        total_tokens = sum(self.count_tokens(msg) for msg in messages)
        available_tokens = self.max_tokens - self.buffer_tokens
        remaining_tokens = available_tokens - total_tokens
        
        return {
            "total_tokens": total_tokens,
            "available_tokens": available_tokens,
            "remaining_tokens": remaining_tokens,
            "buffer_tokens": self.buffer_tokens
        } 