import os
from typing import Optional, List, Dict
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

    # 开放域回退配置
    allow_open_domain_fallback: bool = True
    open_domain_allowed_categories: List[str] = ["tech", "general"]
    min_rag_hits: int = 1
    min_rag_similarity: float = 0.2
    
    # 邮件配置
    email_sender: str = "208621381@qq.com"
    email_password: str = "lgntvtzvjpzbbibc"
    email_smtp_server: str = "smtp.qq.com"
    email_smtp_port: int = 587
    default_recipient: str = "example@company.com"

    # 工具安全与失败恢复配置
    tool_whitelist_categories: List[str] = ["file", "email", "calendar"]
    tool_blacklist: List[str] = []  # 形如 "email:email_send" 的完整ID
    # 需要用户确认的工具（按类别配置工具名）；更细粒度的动作确认见 tool_confirm_actions
    tool_confirm_required: Dict[str, List[str]] = {
        "file": ["file_write"],
        "email": [],
        "calendar": []
    }
    # 需要确认的具体动作配置：{category: {tool_name: [actions...]}}
    tool_confirm_actions: Dict[str, Dict[str, List[str]]] = {
        "calendar": {"calendar_event": ["delete"]}
    }
    # 失败重试设置
    tool_retry_enabled: bool = True
    tool_retry_times: int = 2
    tool_retry_backoff_ms: int = 200

    # 响应水印配置（使用零宽字符追加在末尾，不影响可见前缀断言）
    watermark_enabled: bool = True

    class Config:
        env_file = ".env"

settings = Settings()