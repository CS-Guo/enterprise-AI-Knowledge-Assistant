from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging
import asyncio

logger = logging.getLogger(__name__)

class BaseMCPTool(ABC):
    """MCP工具基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"mcp.{name}")
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行工具操作"""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """获取工具的JSON Schema"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_schema()
        }
    
    @abstractmethod
    def get_parameters_schema(self) -> Dict[str, Any]:
        """获取参数schema"""
        pass
    
    async def safe_execute(self, **kwargs) -> Dict[str, Any]:
        """安全执行工具，包含错误处理与有限次重试"""
        from config.settings import settings
        retry_enabled = getattr(settings, "tool_retry_enabled", True)
        max_retries = max(0, int(getattr(settings, "tool_retry_times", 2)))
        backoff_ms = max(0, int(getattr(settings, "tool_retry_backoff_ms", 200)))
        attempts = 0
        last_err: Exception | None = None
        while True:
            try:
                attempts += 1
                self.logger.info(f"执行工具 {self.name}，参数: {kwargs}")
                result = await self.execute(**kwargs)
                self.logger.info(f"工具 {self.name} 执行成功")
                return {
                    "success": True,
                    "result": result,
                    "tool_name": self.name,
                    "attempts": attempts
                }
            except Exception as e:
                last_err = e
                self.logger.error(f"工具 {self.name} 执行失败(第{attempts}次): {e}")
                if not retry_enabled or attempts > max_retries:
                    return {
                        "success": False,
                        "error": str(e),
                        "tool_name": self.name,
                        "attempts": attempts
                    }
                # backoff
                if backoff_ms > 0:
                    try:
                        await asyncio.sleep(backoff_ms / 1000.0)
                    except Exception:
                        pass
                # retry loop继续