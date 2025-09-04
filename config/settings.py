import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI配置
    openai_api_key: Optional[str] = "sk-BdFxZ0abG1APGay9jk6QUf47xARiNQrqDhgTG6y2bAS2ruaz"
    openai_model: str = "GLM-4.5-Flash"
    base_url: str = "https://www.dmxapi.cn/v1"
    
    # 向量数据库配置
    vector_db_path: str = "/Users/guoshengfeng/study/enterprise-AI-Knowledge-Assistant/vectordb"
    embedding_model: str = "text-embedding-3-small"
    
    # API配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # 文档配置
    documents_path: str = "./data/documents"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # Agent配置
    max_iterations: int = 10
    temperature: float = 0.1
    
    # 邮件配置
    email_sender: str = "208621381@qq.com"
    email_password: str = "lgntvtzvjpzbbibc"
    email_smtp_server: str = "smtp.qq.com"
    email_smtp_port: int = 587
    default_recipient: str = "example@company.com"
    
    class Config:
        env_file = ".env"

settings = Settings()