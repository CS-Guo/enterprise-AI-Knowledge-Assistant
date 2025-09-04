# src/utils/cache.py
import hashlib
import json
import time
from typing import Any, Dict, Optional
from config.logging_config import get_performance_logger

logger = get_performance_logger(__name__)

class SimpleCache:
    """
    简单的内存缓存实现，用于提升响应速度
    """
    
    def __init__(self, max_size: int = 100, ttl: int = 300):
        """
        初始化缓存
        
        Args:
            max_size: 最大缓存条目数
            ttl: 生存时间（秒）
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def _generate_key(self, data: Any) -> str:
        """
        生成缓存键
        
        Args:
            data: 要缓存的数据
            
        Returns:
            缓存键
        """
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, key: Any) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值或None
        """
        cache_key = self._generate_key(key)
        
        if cache_key not in self.cache:
            return None
        
        entry = self.cache[cache_key]
        
        # 检查是否过期
        if time.time() - entry['timestamp'] > self.ttl:
            del self.cache[cache_key]
            return None
        
        return entry['value']
    
    def set(self, key: Any, value: Any) -> None:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        cache_key = self._generate_key(key)
        
        # 如果缓存已满，删除最旧的条目
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[cache_key] = {
            'value': value,
            'timestamp': time.time()
        }
    
    def clear(self) -> None:
        """
        清空缓存
        """
        self.cache.clear()
    
    def size(self) -> int:
        """
        获取缓存大小
        
        Returns:
            缓存条目数
        """
        return len(self.cache)
    
    def cleanup_expired(self) -> int:
        """
        清理过期的缓存条目
        
        Returns:
            清理的条目数
        """
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time - entry['timestamp'] > self.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)

# 全局缓存实例
intent_cache = SimpleCache(max_size=50, ttl=300)  # 意图分析缓存
response_cache = SimpleCache(max_size=100, ttl=600)  # 响应缓存