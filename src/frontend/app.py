# src/frontend/app.py
import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
from pathlib import Path

# 页面配置
st.set_page_config(
    page_title="企业智能知识助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/help',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': '企业智能知识助手 v2.0 - 基于AI的企业知识管理平台'
    }
)

# 常量配置
API_BASE_URL = "http://localhost:8000/api/v1"

# 自定义CSS - 现代化设计
st.markdown("""
<style>
    /* 导入Google字体 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* 全局样式重置 */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #f8f9fa;
        min-height: 100vh;
    }
    
    /* 主容器样式 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: #ffffff;
        border-radius: 12px;
        margin: 0.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
        max-width: 100%;
    }
    
    /* 主标题样式 */
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: clamp(1.8rem, 4vw, 3rem);
        font-weight: 700;
        margin-bottom: 2rem;
        text-shadow: none;
    }
    
    /* 聊天消息样式 */
    .chat-message {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .chat-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        border-radius: 2px;
    }
    
    .user-message {
        background: linear-gradient(135deg, #e8f5e8, #f0f8f0);
        border: 1px solid rgba(40, 167, 69, 0.2);
        margin-left: 2rem;
    }
    
    .user-message::before {
        background: linear-gradient(135deg, #28a745, #20c997);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #e3f2fd, #f0f8ff);
        border: 1px solid rgba(31, 119, 180, 0.2);
        margin-right: 2rem;
    }
    
    .assistant-message::before {
        background: linear-gradient(135deg, #1f77b4, #2196f3);
    }
    
    .tool-execution {
        background: linear-gradient(135deg, #fff8e1, #fffbf0);
        border: 1px solid rgba(255, 193, 7, 0.2);
    }
    
    .tool-execution::before {
        background: linear-gradient(135deg, #ffc107, #ff9800);
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa, #ffffff);
        border-right: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    .sidebar-info {
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid rgba(0, 0, 0, 0.1);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: #3498db;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(52, 152, 219, 0.2);
    }
    
    .stButton > button:hover {
        background: #2980b9;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
    }
    
    /* 输入框样式 */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid rgba(52, 152, 219, 0.2);
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
    }
    
    /* 选项卡样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        border: 1px solid rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: #3498db;
        color: white;
    }
    
    /* 指标卡片样式 */
    .metric-card {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    }
    
    /* 加载动画 */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    
    /* 响应式设计 */
    @media (max-width: 768px) {
        .main .block-container {
            margin: 0.25rem;
            padding: 1rem;
            border-radius: 8px;
        }
        
        .main-header {
            font-size: clamp(1.5rem, 6vw, 2rem);
            margin-bottom: 1rem;
        }
        
        .chat-message {
            margin-left: 0.25rem;
            margin-right: 0.25rem;
            padding: 1rem;
        }
        
        .user-message {
            margin-left: 0.25rem;
        }
        
        .assistant-message {
            margin-right: 0.25rem;
        }
        
        .metric-card {
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
    }
    
    @media (max-width: 480px) {
        .main .block-container {
            margin: 0.1rem;
            padding: 0.75rem;
        }
        
        .stButton > button {
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }
        
        .chat-message {
            padding: 0.75rem;
            margin: 0.5rem 0.1rem;
        }
    }
    
    /* 滚动条样式 */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #3498db;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# 工具函数
def make_api_request(endpoint: str, method: str = "GET", data: Dict = None, files=None):
    """发送API请求"""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, data=data)
            else:
                response = requests.post(url, json=data)
        else:
            raise ValueError(f"不支持的HTTP方法: {method}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API请求失败: {e}")
        return None

def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}"
    if "system_status" not in st.session_state:
        st.session_state.system_status = None

def render_message(message: Dict[str, Any]):
    """渲染聊天消息"""
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 您:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    elif role == "thinking":
        st.markdown(f"""
        <div class="chat-message assistant-message" style="opacity: 0.7;">
            <strong>🤖 助手:</strong><br>
            <div style="display: flex; align-items: center;">
                <div class="loading" style="margin-right: 10px;">💭</div>
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif role == "assistant":
        provenance = message.get("provenance")
        is_open = bool(message.get("open_domain_fallback"))
        disclaimers = message.get("disclaimers") or []
        badge_html = ""
        if is_open:
            source_text = f"来源: {provenance}" if provenance else "来源: 未知"
            badge_html = f"""
            <div style="margin-top: 8px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                <span style="display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:#e3f2fd; color:#1f77b4; border:1px solid rgba(31,119,180,0.35);">🌐 开放域回答</span>
                <span style="font-size:12px; color:#6c757d;">{source_text}</span>
            </div>
            """
        disclaimers_html = ""
        if disclaimers:
            items = "".join([f"<li>{d}</li>" for d in disclaimers])
            disclaimers_html = f"""
            <div style="margin-top:8px; font-size:12px; color:#6c757d; background:#fff8e1; border-left:3px solid #ffc107; padding:8px 10px; border-radius:6px;">
                <strong>提示：</strong>
                <ul style="margin: 4px 0 0 1rem;">{items}</ul>
            </div>
            """
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 助手:</strong><br>
            {content}
            {badge_html}
            {disclaimers_html}
        </div>
        """, unsafe_allow_html=True)
    elif role == "tool":
        st.markdown(f"""
        <div class="chat-message tool-execution">
            <strong>🔧 操作结果:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)

# 主应用
def main():
    init_session_state()
    
    # 主标题
    st.markdown('<h1 class="main-header">🤖 企业智能知识助手</h1>', unsafe_allow_html=True)
    
    # 顶部信息栏 - 技术架构和技术支持
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        with st.expander("🔧 技术架构", expanded=False):
            st.markdown("""
            **后端技术栈:**
            - LangGraph: 工作流编排
            - RAG: 文档检索和理解
            - MCP: 工具协议
            - FastAPI: 后端API
            
            **前端技术:**
            - Streamlit: 用户界面
            - Pandas: 数据处理
            - Requests: API通信
            """)
    
    with col2:
        with st.expander("📞 技术支持", expanded=False):
            st.markdown("""
            如遇到问题，请联系技术支持：
            - 📧 Email: shengfeng.guo@gmail.com
            - 📱 Phone: +(86)1008611
            - 🕒 工作时间: 9:00-18:00
            """)
    
    with col3:
        with st.expander("ℹ️ 系统信息", expanded=False):
            st.markdown(f"""
            - 🆔 会话ID: `{st.session_state.session_id[:8]}...`
            - 💬 消息数: {len(st.session_state.messages)}
            - 🕒 当前时间: {datetime.now().strftime('%H:%M:%S')}
            """)
    
    st.divider()
    
    # 侧边栏 - 现代化设计
    with st.sidebar:
        # 系统状态卡片
        st.markdown("### 🔧 系统控制")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 检查状态", use_container_width=True, key="check_status_btn"):
                with st.spinner("检查中..."):
                    status = make_api_request("/status")
                    if status:
                        st.session_state.system_status = status
        
        with col2:
            if st.button("🔄 刷新", use_container_width=True, key="refresh_sidebar_btn"):
                st.rerun()
        
        # 显示系统状态 - 美化版本
        if st.session_state.system_status:
            status = st.session_state.system_status
            
            # 系统状态指示器
            system_status = status["system"]
            status_color = "🟢" if system_status == "running" else "🔴"
            st.markdown(f"""
            <div class="metric-card">
                <h4>{status_color} 系统状态</h4>
                <p style="font-size: 1.2em; font-weight: 600; color: {'#28a745' if system_status == 'running' else '#dc3545'};">
                    {system_status.upper()}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # 组件状态网格
            st.markdown("#### 📊 组件状态")
            components = status["components"]
            
            # 创建组件状态网格
            for i, (component, state) in enumerate(components.items()):
                if i % 2 == 0:
                    cols = st.columns(2)
                
                with cols[i % 2]:
                    icon = "✅" if state == "active" else "❌"
                    color = "#28a745" if state == "active" else "#dc3545"
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #ffffff, #f8f9fa);
                        padding: 0.75rem;
                        border-radius: 8px;
                        border: 1px solid rgba(0, 0, 0, 0.1);
                        text-align: center;
                        margin: 0.25rem 0;
                    ">
                        <div style="font-size: 1.2em;">{icon}</div>
                        <div style="font-size: 0.8em; color: {color}; font-weight: 500;">{component}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 知识库统计
            if "retriever_stats" in status:
                stats = status["retriever_stats"]
                doc_count = stats.get('total_documents', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📚 知识库</h4>
                    <p style="font-size: 2em; font-weight: 700; color: #667eea; margin: 0;">{doc_count}</p>
                    <p style="margin: 0; color: #666;">文档数量</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # 快捷操作 - 改进版本
        st.markdown("### ⚡ 快捷操作")
        
        # 对话管理
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清空", use_container_width=True, help="清空当前对话历史", key="clear_chat_btn"):
                st.session_state.messages = []
                st.success("对话已清空！")
                time.sleep(1)
                st.rerun()
        
        with col2:
            if st.button("💾 导出", use_container_width=True, help="导出对话记录", key="export_chat_btn"):
                if st.session_state.messages:
                    chat_export = "\n\n".join([
                        f"{msg['role'].upper()}: {msg['content']}"
                        for msg in st.session_state.messages
                    ])
                    st.download_button(
                        "📥 下载对话",
                        chat_export,
                        file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.info("暂无对话记录")
        
        # 会话信息
        st.markdown("### ℹ️ 会话信息")
        st.markdown(f"""
        <div class="sidebar-info">
            <p><strong>🆔 会话ID:</strong><br><code style="font-size: 0.8em;">{st.session_state.session_id}</code></p>
            <p><strong>💬 消息数:</strong> {len(st.session_state.messages)}</p>
            <p><strong>🕒 当前时间:</strong><br>{datetime.now().strftime('%H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 主要内容区域 - 调整比例让聊天界面更大
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 聊天界面
        chat_tab, docs_tab, features_tab = st.tabs(["💬 智能对话", "📚 文档管理", "🔧 功能列表"])
        
        with chat_tab:
            render_chat_interface()
        
        with docs_tab:
            render_document_management()
        
        with features_tab:
            render_feature_list()
    
    with col2:
        # 信息面板
        render_info_panel()

def render_chat_interface():
    """渲染聊天界面 - 现代化版本"""
    # 聊天标题和统计
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("### 💬 智能对话")
    with col2:
        if st.session_state.messages:
            st.metric("对话轮数", len([m for m in st.session_state.messages if m['role'] == 'user']))
    with col3:
        if st.button("🔄 重新开始", help="开始新的对话", key="restart_chat_btn"):
            st.session_state.messages = []
            st.rerun()
    
    # 聊天历史容器 - 增大高度改进滚动体验
    chat_container = st.container(height=600)
    with chat_container:
        if not st.session_state.messages:
            # 欢迎消息
            st.markdown("""
            <div style="
                text-align: center;
                padding: 3rem 2rem;
                background: linear-gradient(135deg, #f8f9fa, #ffffff);
                border-radius: 16px;
                border: 2px dashed rgba(102, 126, 234, 0.3);
                margin: 2rem 0;
            ">
                <h3 style="color: #667eea; margin-bottom: 1rem;">👋 欢迎使用企业智能知识助手！</h3>
                <p style="color: #666; font-size: 1.1em; margin-bottom: 1.5rem;">我可以帮您查询企业政策、安排会议、处理文档等</p>
                <p style="color: #888; font-size: 0.9em;">💡 点击下方的预设问题开始对话，或直接输入您的问题</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for message in st.session_state.messages:
                render_message(message)
            
            # 自动滚动到底部
            if st.session_state.get('auto_scroll', False):
                st.markdown('<div id="scroll-target"></div>', unsafe_allow_html=True)
                st.markdown("""
                <script>
                    setTimeout(function() {
                        var element = document.getElementById('scroll-target');
                        if (element) {
                            element.scrollIntoView({behavior: 'smooth'});
                        }
                    }, 100);
                </script>
                """, unsafe_allow_html=True)
                st.session_state.auto_scroll = False
    
    # 预设问题 - 卡片式设计
    if not st.session_state.messages or len(st.session_state.messages) < 2:
        st.markdown("#### 💡 热门问题")
        example_questions = [
            {"text": "公司的年假政策是什么？", "icon": "🏖️", "category": "人事政策"},
            {"text": "公司的请假流程是怎么样的？", "icon": "📋", "category": "流程指南"},
            {"text": "新员工如何申请虚拟机？", "icon": "💻", "category": "技术支持"},
            {"text": "帮我安排下周的会议", "icon": "📅", "category": "日程管理"}
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(
                    f"{question['icon']} {question['text']}",
                    key=f"example_{i}",
                    use_container_width=True,
                    help=f"类别: {question['category']}"
                ):
                    send_message(question['text'])
    
    # 聊天输入区域 - 改进设计
    st.markdown("---")
    
    # 输入提示和快捷操作
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        input_mode = st.selectbox(
            "输入模式",
            ["💬 普通对话", "📚 文档查询", "🛠️ 工具调用"],
            label_visibility="collapsed"
        )
    
    with col3:
        if st.button("🎲 随机问题", help="生成一个随机问题", key="random_question_btn"):
            random_questions = [
                "介绍一下公司的组织架构",
                "如何申请出差报销？",
                "公司有哪些培训课程？",
                "办公设备如何申请？",
                "员工福利有哪些？"
            ]
            import random
            random_question = random.choice(random_questions)
            send_message(random_question)
    
    # 主输入区域
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_area(
                "输入您的问题:",
                height=120,
                placeholder="请输入您想了解的问题...\n\n💡 提示：您可以询问企业政策、申请流程、技术支持等问题",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # 垂直对齐
            submitted = st.form_submit_button(
                "🚀\n发送",
                use_container_width=True,
                type="primary"
            )
            
            # 快捷键提示
            st.caption("💡 Ctrl+Enter 快速发送")
        
        if submitted and user_input.strip():
            send_message(user_input.strip())

def send_message(message: str):
    """发送消息"""
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": message})
    
    # 添加思考状态消息
    thinking_msg = {"role": "thinking", "content": "🤖 AI助手正在思考..."}
    st.session_state.messages.append(thinking_msg)
    
    # 使用spinner显示加载状态，而不是立即rerun
    with st.spinner("AI助手正在思考..."):
        # 准备请求数据
        request_data = {
            "query": message,
            "conversation_history": [
                {"query": msg["content"] if msg["role"] == "user" else "", 
                 "response": msg["content"] if msg["role"] == "assistant" else ""}
                for msg in st.session_state.messages[-10:]  # 保留最近10轮对话
                if msg["role"] in ["user", "assistant"]
            ],
            "session_id": st.session_state.session_id
        }
        
        # 发送请求
        response = make_api_request("/chat", method="POST", data=request_data)
        
        # 移除思考状态消息
        st.session_state.messages = [msg for msg in st.session_state.messages if msg.get("role") != "thinking"]
        
        if response:
            # 添加助手回复
            assistant_response = response.get("response", "")
            if assistant_response.strip():
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": assistant_response,
                    "open_domain_fallback": response.get("open_domain_fallback", False),
                    "provenance": response.get("provenance"),
                    "disclaimers": response.get("disclaimers") or []
                })
            
            # 显示工具执行结果
            actions_performed = response.get("actions_performed", [])
            if actions_performed:
                # 格式化工具执行结果
                tool_content = "### 🔧 工具执行详情\n\n"
                for i, action in enumerate(actions_performed, 1):
                    action_type = action.get("type", "未知操作")
                    action_status = action.get("status", "unknown")
                    action_result = action.get("result", {})
                    action_error = action.get("error", "")
                    
                    status_emoji = "✅" if action_status == "completed" else "❌" if action_status == "failed" else "⚠️"
                    
                    tool_content += f"**{i}. {action_type}操作** {status_emoji}\n\n"
                    
                    if action_status == "completed" and action_result:
                        # 处理成功的操作结果
                        if isinstance(action_result, dict):
                            # 特殊处理复合操作结果
                            if "composite_results" in action_result:
                                composite_results = action_result["composite_results"]
                                tool_content += f"**复合操作包含 {len(composite_results)} 个子操作：**\n\n"
                                for j, comp_result in enumerate(composite_results, 1):
                                    category = comp_result.get("category", "未知")
                                    result_data = comp_result.get("result", {})
                                    tool_content += f"  {j}. **{category}工具**\n"
                                    
                                    if result_data.get("success"):
                                        if category == "calendar":
                                            event_data = result_data.get("result", {})
                                            if event_data.get("action") == "create":
                                                event_info = event_data.get("event", {})
                                                tool_content += f"     - 会议标题: {event_info.get('title', '未设置')}\n"
                                                tool_content += f"     - 开始时间: {event_info.get('start_time', '未设置')}\n"
                                                tool_content += f"     - 结束时间: {event_info.get('end_time', '未设置')}\n"
                                                tool_content += f"     - 地点: {event_info.get('location', '未设置')}\n"
                                        elif category == "email":
                                            email_data = result_data.get("result", {})
                                            tool_content += f"     - 收件人: {', '.join(email_data.get('to_addresses', []))}\n"
                                            tool_content += f"     - 主题: {email_data.get('subject', '未设置')}\n"
                                    else:
                                        tool_content += f"     - 状态: 执行失败\n"
                                    tool_content += "\n"
                            else:
                                # 普通操作结果
                                for key, value in action_result.items():
                                    if key not in ["composite_results"]:
                                        tool_content += f"- **{key}**: {value}\n"
                        else:
                            tool_content += f"```\n{str(action_result)}\n```\n"
                    elif action_status == "failed" and action_error:
                        tool_content += f"**错误信息**: {action_error}\n\n"
                    
                    tool_content += "\n---\n\n"
                
                st.session_state.messages.append({
                    "role": "tool",
                    "content": tool_content
                })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "抱歉，我现在无法处理您的请求。请稍后再试。"
            })
    
    # 设置自动滚动标记
    st.session_state.auto_scroll = True
    st.rerun()

def render_document_management():
    """渲染文档管理界面 - 现代化版本"""
    # 标题和统计
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("### 📚 文档管理")
    with col2:
        if st.button("🔄 刷新列表", use_container_width=True, key="refresh_docs_btn"):
            st.rerun()
    with col3:
        if st.button("📊 查看统计", use_container_width=True, key="view_stats_docs_btn"):
            documents = make_api_request("/documents")
            if documents:
                stats = documents.get("vector_store_stats", {})
                st.session_state.doc_stats = stats
    
    # 文档上传区域 - 改进设计
    st.markdown("#### 📤 上传新文档")
    
    # 上传区域容器
    upload_container = st.container()
    with upload_container:
        # 拖拽上传提示 - 增强版
        st.markdown("""
        <div style="
            border: 3px dashed #667eea;
            border-radius: 20px;
            padding: 3rem 2rem;
            text-align: center;
            background: linear-gradient(135deg, #f8f9fa, #ffffff);
            margin: 1rem 0;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        " 
        onmouseover="this.style.borderColor='#4f46e5'; this.style.backgroundColor='#f0f9ff'; this.style.transform='scale(1.02)'"
        onmouseout="this.style.borderColor='#667eea'; this.style.backgroundColor=''; this.style.transform='scale(1)'">
            <div style="
                font-size: 3rem;
                margin-bottom: 1rem;
                animation: bounce 2s infinite;
            ">📁</div>
            <h3 style="color: #667eea; margin-bottom: 1rem; font-weight: 600;">拖拽文件到此处</h3>
            <p style="color: #666; margin-bottom: 1rem; font-size: 1.1em;">或点击下方按钮选择文件</p>
            <div style="
                display: flex;
                justify-content: center;
                gap: 2rem;
                margin: 1.5rem 0;
                flex-wrap: wrap;
            ">
                <div style="text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📕</div>
                    <span style="color: #666; font-size: 0.9em;">PDF</span>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📘</div>
                    <span style="color: #666; font-size: 0.9em;">DOCX</span>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📄</div>
                    <span style="color: #666; font-size: 0.9em;">TXT</span>
                </div>
            </div>
            <p style="color: #888; font-size: 0.9em; margin-bottom: 0;">最大文件大小: 10MB</p>
        </div>
        
        <style>
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% {
                transform: translateY(0);
            }
            40% {
                transform: translateY(-10px);
            }
            60% {
                transform: translateY(-5px);
            }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 文件上传器
        uploaded_file = st.file_uploader(
            "选择文件",
            type=['pdf', 'docx', 'txt'],
            help="支持PDF、DOCX、TXT格式，最大10MB",
            label_visibility="collapsed"
        )
        
        # 上传按钮和进度
        if uploaded_file is not None:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.markdown(f"**📄 {uploaded_file.name}**")
                st.caption(f"大小: {uploaded_file.size / 1024:.1f} KB")
            
            with col2:
                if st.button("🚀 开始上传", use_container_width=True, type="primary", key="upload_docs_btn"):
                    # 创建进度条
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # 模拟上传进度
                        for i in range(101):
                            progress_bar.progress(i)
                            if i < 30:
                                status_text.text("📤 正在上传文件...")
                            elif i < 70:
                                status_text.text("🔍 正在分析文档...")
                            elif i < 90:
                                status_text.text("🧠 正在构建向量索引...")
                            else:
                                status_text.text("✅ 处理完成！")
                            time.sleep(0.02)
                        
                        # 实际上传
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        result = make_api_request("/documents/upload", method="POST", files=files)
                        
                        if result:
                            st.success(f"✅ 文档上传成功！")
                            st.info(f"📝 上传ID: `{result['upload_id']}`")
                            st.balloons()  # 庆祝动画
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("❌ 文档上传失败")
                    
                    except Exception as e:
                        st.error(f"❌ 上传过程中出现错误: {str(e)}")
            
            with col3:
                st.markdown("**文件预览**")
                if uploaded_file.type == "text/plain":
                    # 保存当前位置，读取预览后恢复
                    current_position = uploaded_file.tell() if hasattr(uploaded_file, 'tell') else 0
                    try:
                        preview_content = uploaded_file.read(200)
                        if isinstance(preview_content, bytes):
                            preview = preview_content.decode('utf-8', errors='ignore')
                        else:
                            preview = str(preview_content)[:200]
                        # 重置文件指针到开始位置
                        if hasattr(uploaded_file, 'seek'):
                            uploaded_file.seek(0)
                        st.text_area("内容预览", preview, height=100, disabled=True)
                    except Exception as e:
                        st.caption(f"预览失败: {str(e)}")
                        # 确保文件指针重置
                        if hasattr(uploaded_file, 'seek'):
                            uploaded_file.seek(0)
                else:
                    st.info("二进制文件")
    
    st.markdown("---")
    
    # 已上传文档列表 - 改进版本
    st.markdown("#### 📋 文档库")
    
    # 获取文档列表
    documents = make_api_request("/documents")
    
    if documents and documents.get("documents"):
        docs = documents["documents"]
        
        # 文档统计卡片
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>📄 总文档数</h4>
                <p style="font-size: 2em; font-weight: 700; color: #667eea; margin: 0;">{}</p>
            </div>
            """.format(len(docs)), unsafe_allow_html=True)
        
        with col2:
            total_size = sum(doc.get('file_size', 0) for doc in docs) / (1024 * 1024)  # MB
            st.markdown("""
            <div class="metric-card">
                <h4>💾 总大小</h4>
                <p style="font-size: 2em; font-weight: 700; color: #28a745; margin: 0;">{:.1f}MB</p>
            </div>
            """.format(total_size), unsafe_allow_html=True)
        
        with col3:
            file_types = {}
            for doc in docs:
                ext = doc.get('file_extension', 'unknown')
                file_types[ext] = file_types.get(ext, 0) + 1
            most_common = max(file_types.items(), key=lambda x: x[1]) if file_types else ('无', 0)
            st.markdown("""
            <div class="metric-card">
                <h4>📊 主要格式</h4>
                <p style="font-size: 1.5em; font-weight: 700; color: #ffc107; margin: 0;">{}</p>
                <p style="margin: 0; color: #666; font-size: 0.9em;">{} 个文件</p>
            </div>
            """.format(most_common[0].upper(), most_common[1]), unsafe_allow_html=True)
        
        with col4:
            stats = documents.get("vector_store_stats", {})
            vector_count = stats.get("total_documents", 0)
            st.markdown("""
            <div class="metric-card">
                <h4>🧠 向量数据</h4>
                <p style="font-size: 2em; font-weight: 700; color: #dc3545; margin: 0;">{}</p>
            </div>
            """.format(vector_count), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 文档列表表格 - 美化版本
        df = pd.DataFrame(docs)
        df["upload_time"] = pd.to_datetime(df["upload_time"], unit='s')
        df["file_size"] = df["file_size"].apply(lambda x: f"{x/1024:.1f} KB")
        df["upload_date"] = df["upload_time"].dt.strftime('%Y-%m-%d')
        df["upload_time_formatted"] = df["upload_time"].dt.strftime('%H:%M:%S')
        
        # 添加文件类型图标
        def get_file_icon(ext):
            icons = {
                'pdf': '📕',
                'docx': '📘', 
                'txt': '📄',
                'doc': '📘'
            }
            return icons.get(ext.lower(), '📄')
        
        df["file_icon"] = df["file_extension"].apply(get_file_icon)
        df["display_name"] = df["file_icon"] + " " + df["filename"]
        
        # 显示表格
        st.dataframe(
            df[["display_name", "file_size", "upload_date", "upload_time_formatted"]],
            use_container_width=True,
            column_config={
                "display_name": st.column_config.TextColumn("📁 文件名", width="large"),
                "file_size": st.column_config.TextColumn("💾 大小", width="small"),
                "upload_date": st.column_config.TextColumn("📅 日期", width="medium"),
                "upload_time_formatted": st.column_config.TextColumn("🕒 时间", width="small")
            },
            hide_index=True
        )
        
    else:
        # 空状态展示
        st.markdown("""
        <div style="
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, #f8f9fa, #ffffff);
            border-radius: 16px;
            border: 2px dashed rgba(102, 126, 234, 0.3);
            margin: 2rem 0;
        ">
            <h3 style="color: #667eea; margin-bottom: 1rem;">📭 文档库为空</h3>
            <p style="color: #666; font-size: 1.1em; margin-bottom: 1.5rem;">还没有上传任何文档</p>
            <p style="color: #888; font-size: 0.9em;">💡 上传文档后，AI助手就能基于您的企业知识回答问题了</p>
        </div>
        """, unsafe_allow_html=True)

def render_feature_list():
    """渲染功能列表界面 - 简化版本"""
    st.markdown("### 📋 可用功能列表")
    
    # 获取可用工具
    tools = make_api_request("/tools")
    
    if tools and tools.get("tools"):
        tools_data = tools["tools"]
        
        # 将嵌套的工具结构转换为平面列表
        available_tools = []
        for category, category_tools in tools_data.items():
            for tool_name, tool_schema in category_tools.items():
                tool_info = {
                    "name": tool_name,
                    "category": category,
                    "description": tool_schema.get("description", ""),
                    "schema": tool_schema
                }
                available_tools.append(tool_info)
        
        # 工具分类
        tool_categories = {
            "📧 邮件工具": [t for t in available_tools if t.get("category") == "email"],
            "📅 日历工具": [t for t in available_tools if t.get("category") == "calendar"],
            "📁 文件工具": [t for t in available_tools if t.get("category") == "file"],
            "🔧 其他工具": []
        }
        
        # 将未分类的工具放入"其他工具"
        categorized_tools = set()
        for category_tools in tool_categories.values():
            categorized_tools.update(t["name"] for t in category_tools)
        
        tool_categories["🔧 其他工具"] = [t for t in available_tools if t["name"] not in categorized_tools]
        
        # 使用说明
        st.info("💡 **使用说明**: 您可以在对话中直接描述需要执行的操作，系统会自动调用相应的工具。例如：'帮我搜索项目中的配置文件' 或 '读取README.md文件的内容'")
        
        # 显示工具分类
        for category, category_tools in tool_categories.items():
            if not category_tools:
                continue
                
            st.markdown(f"#### {category}")
            
            for tool in category_tools:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{tool.get('name', '未知工具')}**")
                        st.markdown(f"*{tool.get('description', '暂无描述')}*")
                    
                    with col2:
                        st.markdown(f"<span style='color: #28a745; font-size: 1.2em;'>🟢 可用</span>", unsafe_allow_html=True)
                    
                    # 显示参数信息
                    schema = tool.get("schema", {})
                    if schema.get("properties"):
                        with st.expander(f"📝 参数说明", expanded=False):
                            properties = schema["properties"]
                            required_params = schema.get("required", [])
                            
                            for param_name, param_info in properties.items():
                                param_type = param_info.get("type", "string")
                                param_desc = param_info.get("description", "")
                                is_required = param_name in required_params
                                required_text = " (必需)" if is_required else " (可选)"
                                
                                st.markdown(f"- **{param_name}** ({param_type}){required_text}: {param_desc}")
                    
                    st.markdown("---")
    
    else:
        st.error("❌ 无法获取工具列表，请检查后端服务是否正常运行")
    
    # 使用示例
    st.markdown("### 💬 对话使用示例")
    
    examples = [
        "🔍 帮我搜索项目中包含'config'的文件",
        "📖 读取README.md文件的内容",
        "✍️ 创建一个新的文本文件，内容是'Hello World'",
        "📧 发送一封邮件给team@company.com",
        "📅 创建一个明天下午2点的会议"
    ]
    
    for example in examples:
        st.markdown(f"- {example}")
    
    st.info("💡 **提示**: 直接在上方的智能对话界面中输入您的需求，系统会自动识别并执行相应的工具！")

def render_info_panel():
    """渲染信息面板"""
    st.subheader("ℹ️ 信息面板")
    
    # 会话信息
    st.write("**🔗 会话信息:**")
    st.write(f"- 会话ID: `{st.session_state.session_id}`")
    st.write(f"- 消息数量: {len(st.session_state.messages)}")
    st.write(f"- 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.divider()
    
    # 使用说明
    st.write("**📖 使用说明:**")
    st.markdown("""
    1. **智能对话**: 直接提问获取企业知识
    2. **文档管理**: 上传企业文档构建知识库
    3. **工具中心**: 使用各种自动化工具
    4. **系统监控**: 查看系统运行状态
    """)
    
    st.divider()
    
    # 快速操作
    st.write("**⚡ 快速操作:**")
    if st.button("🔄 刷新页面", use_container_width=True, key="refresh_info_panel_btn"):
        st.rerun()
    
    if st.button("📊 查看统计", use_container_width=True, key="view_stats_info_panel_btn"):
        status = make_api_request("/status")
        if status:
            st.session_state.system_status = status
            st.success("状态已更新！")

if __name__ == "__main__":
    main()