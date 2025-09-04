from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import uuid
from pathlib import Path
import logging

from src.agents.workflow import create_workflow, AgentState
from src.rag.retriever import DocumentRetriever
from src.mcp.file_tools import FileTools
from src.mcp.email_tools import EmailTools
from src.mcp.calendar_tools import CalendarTools
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic模型定义
class ChatRequest(BaseModel):
    query: str
    conversation_history: Optional[List[Dict[str, Any]]] = []
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    intent_analysis: Optional[Dict[str, Any]] = None
    documents_used: Optional[List[str]] = []
    actions_performed: Optional[List[Dict[str, Any]]] = []

class ToolExecutionRequest(BaseModel):
    tool_category: str  # file, email, calendar
    tool_name: str
    parameters: Dict[str, Any]

class DocumentUploadResponse(BaseModel):
    filename: str
    file_size: int
    upload_id: str
    status: str

# 全局变量
workflow = None
retriever = None
file_tools = None
email_tools = None
calendar_tools = None

# 初始化组件
async def initialize_components():
    global workflow, retriever, file_tools, email_tools, calendar_tools
    
    if workflow is None:
        workflow = create_workflow()
        logger.info("LangGraph工作流已初始化")
    
    if retriever is None:
        retriever = DocumentRetriever()
        logger.info("文档检索器已初始化")
    
    if file_tools is None:
        file_tools = FileTools()
        logger.info("文件工具已初始化")
    
    if email_tools is None:
        email_tools = EmailTools()
        logger.info("邮件工具已初始化")
    
    if calendar_tools is None:
        calendar_tools = CalendarTools()
        logger.info("日历工具已初始化")

# 聊天接口
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """主要的聊天接口"""
    try:
        await initialize_components()
        
        # 生成会话ID
        conversation_id = request.session_id or str(uuid.uuid4())
        
        # 构建初始状态
        initial_state = AgentState(
            query=request.query,
            intent_analysis={},
            documents=[],
            context="",
            response="",
            actions=[],
            conversation_history=request.conversation_history,
            error="",
            iteration_count=0
        )
        
        # 执行工作流
        result = await workflow.ainvoke(initial_state)
        
        # 构建响应
        response = ChatResponse(
            response=result.get("response", "抱歉，我无法处理您的请求。"),
            conversation_id=conversation_id,
            intent_analysis=result.get("intent_analysis"),
            documents_used=result.get("documents", []),
            actions_performed=result.get("actions", [])
        )
        
        logger.info(f"聊天请求处理完成，会话ID: {conversation_id}")
        return response
        
    except Exception as e:
        logger.error(f"聊天请求处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"聊天请求处理失败: {str(e)}")

# 文档上传接口
@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """上传文档到系统"""
    try:
        # 检查文件类型
        allowed_extensions = {'.pdf', '.docx', '.txt'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件类型: {file_extension}。支持的类型: {allowed_extensions}"
            )
        
        # 检查文件大小
        if file.size > settings.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"文件太大: {file.size} 字节。最大允许: {settings.max_file_size} 字节"
            )
        
        # 生成唯一的文件名
        upload_id = str(uuid.uuid4())
        safe_filename = f"{upload_id}_{file.filename}"
        file_path = Path(settings.documents_path) / safe_filename
        
        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存文件
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # 后台任务：处理文档并添加到向量存储
        background_tasks.add_task(process_uploaded_document, str(file_path))
        
        response = DocumentUploadResponse(
            filename=file.filename,
            file_size=len(content),
            upload_id=upload_id,
            status="uploaded"
        )
        
        logger.info(f"文档上传成功: {file.filename}, 上传ID: {upload_id}")
        return response
        
    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")

async def process_uploaded_document(file_path: str):
    """处理上传的文档并添加到向量存储"""
    try:
        logger.info(f"开始处理上传的文档: {file_path}")
        
        # 初始化组件
        await initialize_components()
        logger.info("组件初始化完成")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return
            
        # 处理文档
        from src.rag.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        logger.info(f"开始使用DocumentProcessor处理文档: {file_path}")
        document_chunks = processor.process_document(file_path)
        
        # 添加到向量存储
        if document_chunks:
            logger.info(f"文档处理成功，生成了 {len(document_chunks)} 个文档块，准备添加到向量存储")
            success = retriever.vector_store.add_documents(document_chunks)
            if success:
                logger.info(f"文档已成功添加到向量存储: {file_path}")
                # 获取检索器统计信息
                stats = retriever.get_retriever_stats()
                logger.info(f"当前向量存储统计: {stats}")
            else:
                logger.error(f"文档添加到向量存储失败: {file_path}")
        else:
            logger.warning(f"文档处理未生成任何文档块: {file_path}")
    except Exception as e:
        logger.error(f"处理上传文档失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

# 文档列表接口
@router.get("/documents")
async def list_documents():
    """获取已上传的文档列表"""
    try:
        await initialize_components()
        
        stats = retriever.get_retriever_stats()
        
        # 获取文档目录中的文件
        documents_dir = Path(settings.documents_path)
        documents = []
        
        if documents_dir.exists():
            for file_path in documents_dir.iterdir():
                if file_path.is_file():
                    stat = file_path.stat()
                    documents.append({
                        "filename": file_path.name,
                        "file_size": stat.st_size,
                        "upload_time": stat.st_ctime,
                        "file_extension": file_path.suffix
                    })
        
        return {
            "documents": documents,
            "total_count": len(documents),
            "vector_store_stats": stats
        }
        
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")

# 工具执行接口
@router.post("/tools/execute")
async def execute_tool(request: ToolExecutionRequest):
    """执行MCP工具"""
    try:
        await initialize_components()
        
        tool_category = request.tool_category.lower()
        
        # 根据类别选择工具集
        if tool_category == "file":
            result = await file_tools.execute_tool(request.tool_name, **request.parameters)
        elif tool_category == "email":
            result = await email_tools.execute_tool(request.tool_name, **request.parameters)
        elif tool_category == "calendar":
            result = await calendar_tools.execute_tool(request.tool_name, **request.parameters)
        else:
            raise HTTPException(status_code=400, detail=f"不支持的工具类别: {tool_category}")
        
        logger.info(f"工具执行完成: {tool_category}.{request.tool_name}")
        return result
        
    except Exception as e:
        logger.error(f"工具执行失败: {e}")
        raise HTTPException(status_code=500, detail=f"工具执行失败: {str(e)}")

# 获取可用工具列表
@router.get("/tools")
async def list_tools():
    """获取所有可用的MCP工具"""
    try:
        await initialize_components()
        
        all_tools = {}
        
        # 文件工具
        file_tool_schemas = {name: tool.get_schema() for name, tool in file_tools.get_all_tools().items()}
        all_tools["file"] = file_tool_schemas
        
        # 邮件工具
        email_tool_schemas = {name: tool.get_schema() for name, tool in email_tools.get_all_tools().items()}
        all_tools["email"] = email_tool_schemas
        
        # 日历工具
        calendar_tool_schemas = {name: tool.get_schema() for name, tool in calendar_tools.get_all_tools().items()}
        all_tools["calendar"] = calendar_tool_schemas
        
        return {
            "tools": all_tools,
            "total_tools": sum(len(tools) for tools in all_tools.values())
        }
        
    except Exception as e:
        logger.error(f"获取工具列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取工具列表失败: {str(e)}")

# 系统状态接口
@router.get("/status")
async def system_status():
    """获取系统状态"""
    try:
        await initialize_components()
        
        # 获取检索器统计
        retriever_stats = retriever.get_retriever_stats()
        
        # 检查各组件状态
        status = {
            "system": "healthy",
            "components": {
                "workflow": "active" if workflow else "inactive",
                "retriever": "active" if retriever else "inactive", 
                "file_tools": "active" if file_tools else "inactive",
                "email_tools": "active" if email_tools else "inactive",
                "calendar_tools": "active" if calendar_tools else "inactive"
            },
            "retriever_stats": retriever_stats,
            "settings": {
                "documents_path": settings.documents_path,
                "vector_db_path": settings.vector_db_path,
                "max_file_size": settings.max_file_size
            }
        }
        
        return status
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        return {
            "system": "error",
            "error": str(e)
        }