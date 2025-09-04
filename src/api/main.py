from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from .routes import router
from config.settings import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="企业智能知识助手API",
    description="基于LangGraph+RAG+MCP的AI Agent系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加路由
app.include_router(router, prefix="/api/v1")

# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"全局异常处理: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "内部服务器错误", "detail": str(exc)}
    )

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# 启动事件
@app.on_event("startup")
async def startup_event():
    logger.info("企业智能知识助手API启动中...")
    logger.info(f"文档路径: {settings.documents_path}")
    logger.info(f"向量数据库路径: {settings.vector_db_path}")

# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("企业智能知识助手API关闭")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info"
    )