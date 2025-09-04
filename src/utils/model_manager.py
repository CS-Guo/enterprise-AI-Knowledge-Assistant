# src/utils/model_manager.py
import threading
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config.settings import settings
from config.logging_config import get_performance_logger

logger = get_performance_logger(__name__)

class ModelManager:
    """
    模型管理器 - 单例模式，优化模型加载性能
    """
    
    _instance: Optional['ModelManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'ModelManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._chat_model: Optional[ChatOpenAI] = None
        self._embedding_model: Optional[OpenAIEmbeddings] = None
        self._model_lock = threading.Lock()
        self._initialized = True
    
    def get_chat_model(self) -> ChatOpenAI:
        """
        获取聊天模型（延迟加载）
        
        Returns:
            ChatOpenAI实例
        """
        if self._chat_model is None:
            with self._model_lock:
                if self._chat_model is None:
                    self._chat_model = ChatOpenAI(
                        model=settings.openai_model,
                        temperature=settings.temperature,
                        api_key=settings.openai_api_key,
                        base_url=settings.openai_api_base,
                        request_timeout=30,  # 设置超时
                        max_retries=2  # 减少重试次数
                    )
        return self._chat_model
    
    def get_embedding_model(self) -> OpenAIEmbeddings:
        """
        获取嵌入模型（延迟加载）
        
        Returns:
            OpenAIEmbeddings实例
        """
        if self._embedding_model is None:
            with self._model_lock:
                if self._embedding_model is None:
                    self._embedding_model = OpenAIEmbeddings(
                        model="text-embedding-3-large",
                        base_url="https://www.dmxapi.cn/v1",
                        api_key="sk-BdFxZ0abG1APGay9jk6QUf47xARiNQrqDhgTG6y2bAS2ruaz",
                        request_timeout=30,  # 设置超时
                        max_retries=2  # 减少重试次数
                    )
        return self._embedding_model
    
    def clear_models(self) -> None:
        """
        清理模型缓存
        """
        with self._model_lock:
            self._chat_model = None
            self._embedding_model = None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "chat_model_loaded": self._chat_model is not None,
            "embedding_model_loaded": self._embedding_model is not None,
            "chat_model_name": settings.openai_model,
            "embedding_model_name": "text-embedding-3-large"
        }

# 全局模型管理器实例
model_manager = ModelManager()