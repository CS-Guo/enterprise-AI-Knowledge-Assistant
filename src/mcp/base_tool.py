from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging

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
        """安全执行工具，包含错误处理"""
        try:
            self.logger.info(f"执行工具 {self.name}，参数: {kwargs}")
            result = await self.execute(**kwargs)
            self.logger.info(f"工具 {self.name} 执行成功")
            return {
                "success": True,
                "result": result,
                "tool_name": self.name
            }
        except Exception as e:
            self.logger.error(f"工具 {self.name} 执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": self.name
            }