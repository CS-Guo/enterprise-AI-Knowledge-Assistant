import os
import aiofiles
from pathlib import Path
from typing import Dict, Any, List, Optional
from .base_tool import BaseMCPTool

class FileSearchTool(BaseMCPTool):
    """文件搜索工具"""
    
    def __init__(self):
        super().__init__(
            name="file_search",
            description="在指定目录中搜索文件"
        )
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "搜索目录路径"
                },
                "filename_pattern": {
                    "type": "string", 
                    "description": "文件名模式（支持通配符）"
                },
                "file_extension": {
                    "type": "string",
                    "description": "文件扩展名过滤"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "是否递归搜索子目录",
                    "default": True
                }
            },
            "required": ["directory"]
        }
    
    async def execute(self, directory: str, filename_pattern: str = "*", 
                     file_extension: Optional[str] = None, recursive: bool = True) -> Dict[str, Any]:
        """执行文件搜索"""
        directory_path = Path(directory)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")
        
        files_found = []
        
        if recursive:
            pattern = f"**/{filename_pattern}"
        else:
            pattern = filename_pattern
            
        if file_extension:
            if not file_extension.startswith('.'):
                file_extension = f".{file_extension}"
            pattern = f"{pattern}{file_extension}"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file():
                files_found.append({
                    "filename": file_path.name,
                    "full_path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified_time": file_path.stat().st_mtime
                })
        
        return {
            "files_found": files_found,
            "total_count": len(files_found),
            "search_directory": directory,
            "pattern": filename_pattern
        }

class FileReadTool(BaseMCPTool):
    """文件读取工具"""
    
    def __init__(self):
        super().__init__(
            name="file_read",
            description="读取文件内容"
        )
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "要读取的文件路径"
                },
                "encoding": {
                    "type": "string",
                    "description": "文件编码",
                    "default": "utf-8"
                },
                "max_size": {
                    "type": "integer",
                    "description": "最大读取字节数",
                    "default": 1024000  # 1MB
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, file_path: str, encoding: str = "utf-8", max_size: int = 1024000) -> Dict[str, Any]:
        """执行文件读取"""
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_size = file_path_obj.stat().st_size
        if file_size > max_size:
            raise ValueError(f"文件太大: {file_size} 字节，最大允许: {max_size} 字节")
        
        async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
            content = await f.read()
        
        return {
            "file_path": file_path,
            "content": content,
            "file_size": file_size,
            "encoding": encoding
        }

class FileWriteTool(BaseMCPTool):
    """文件写入工具"""
    
    def __init__(self):
        super().__init__(
            name="file_write",
            description="写入内容到文件"
        )
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "要写入的文件路径"
                },
                "content": {
                    "type": "string",
                    "description": "要写入的内容"
                },
                "encoding": {
                    "type": "string",
                    "description": "文件编码",
                    "default": "utf-8"
                },
                "append": {
                    "type": "boolean",
                    "description": "是否追加写入",
                    "default": False
                }
            },
            "required": ["file_path", "content"]
        }
    
    async def execute(self, file_path: str, content: str, encoding: str = "utf-8", append: bool = False) -> Dict[str, Any]:
        """执行文件写入"""
        file_path_obj = Path(file_path)
        
        # 创建目录（如果不存在）
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        mode = 'a' if append else 'w'
        
        async with aiofiles.open(file_path, mode, encoding=encoding) as f:
            await f.write(content)
        
        return {
            "file_path": file_path,
            "bytes_written": len(content.encode(encoding)),
            "mode": "append" if append else "write"
        }

class FileTools:
    """文件工具集合"""
    
    def __init__(self):
        self.tools = {
            "file_search": FileSearchTool(),
            "file_read": FileReadTool(),
            "file_write": FileWriteTool()
        }
    
    def get_all_tools(self) -> Dict[str, BaseMCPTool]:
        """获取所有文件工具"""
        return self.tools
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """执行指定的文件工具"""
        if tool_name not in self.tools:
            raise ValueError(f"未知的文件工具: {tool_name}")
        
        return await self.tools[tool_name].safe_execute(**kwargs)
