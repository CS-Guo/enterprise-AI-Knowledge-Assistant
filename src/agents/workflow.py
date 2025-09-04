from typing import Dict, Any, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode  # 未使用，暂时注释
import logging
from uuid import uuid4

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """Agent状态定义"""
    query: str                           # 用户查询
    intent_analysis: Dict[str, Any]      # 意图分析结果
    documents: List[str]                 # 检索到的文档
    context: str                         # 整理后的上下文
    response: str                        # 最终回答
    actions: List[Dict[str, Any]]        # 执行的动作
    conversation_history: List[Dict]     # 对话历史
    error: str                           # 错误信息
    iteration_count: int                 # 迭代次数
    tool_result: Dict[str, Any]          # 工具执行结果（成功时写入）
    request_id: str                      # 请求ID，用于贯穿全链路日志


def create_workflow():
    """创建LangGraph工作流"""
    
    async def query_analysis_node(state: AgentState) -> AgentState:
        """查询分析节点"""
        try:
            # 生成贯穿全链路的请求ID
            if not state.get("request_id"):
                state["request_id"] = str(uuid4())
            
            from .knowledge_agent import KnowledgeAgent
            agent = KnowledgeAgent()
            
            intent_analysis = await agent.analyze_query_intent(state["query"])
            state["intent_analysis"] = intent_analysis
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            
            logger.info(f"[{state['request_id']}] 查询分析完成: {intent_analysis}")
            return state
        except Exception as e:
            # 不中断流程，使用回退意图分析，避免后续节点因缺少category报错
            logger.error(f"查询分析节点错误，使用回退策略: {e}")
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            # 回退：简单规则确定意图
            query = state.get("query", "")
            query_lower = query.lower()
            category = "general"
            if any(k in query for k in ["年假", "休假", "请假", "福利", "薪资", "社保", "人事"]):
                category = "hr"
            elif any(k in query for k in ["接口", "报错", "部署", "数据库", "代码", "技术", "bug"]):
                category = "tech"
            state["intent_analysis"] = {
                "intent_type": "question",
                "confidence": 0.5,
                "entities": [],
                "requires_tools": False,
                "tool_category": "none",
                "category": category,
                "action_needed": "",
                "original_query": query,
                "fallback": True
            }
            return state

    async def document_retrieval_node(state: AgentState) -> AgentState:
        """文档检索节点"""
        try:
            from ..rag.retriever import DocumentRetriever
            retriever = DocumentRetriever()
            
            query = state.get("query", "")
            intent = state.get("intent_analysis", {})
            category = intent.get("category", "general")
            
            # 基于意图调整检索策略（容错处理）
            if category == "hr":
                documents = await retriever.retrieve_documents(query, filter_category="hr")
            elif category == "tech":
                documents = await retriever.retrieve_documents(query, filter_category="tech")
            else:
                documents = await retriever.retrieve_documents(query)
            
            state["documents"] = documents
            logger.info(f"检索到 {len(documents)} 个相关文档")
            return state
        except Exception as e:
            logger.error(f"文档检索节点错误: {e}")
            state["error"] = str(e)
            state["documents"] = []
            return state
    
    async def context_assembly_node(state: AgentState) -> AgentState:
        """上下文组装节点"""
        try:
            documents = state["documents"]
            query = state["query"]
            
            # 组装上下文
            context_parts = []
            for i, doc in enumerate(documents[:5]):  # 最多使用5个文档
                context_parts.append(f"文档 {i+1}:\n{doc}\n")
            
            context = "\n".join(context_parts)
            if not context.strip():
                context = "未找到相关文档信息。"
            
            # 如果前序存在工具失败，附加失败摘要，便于后续生成友好解释
            actions = state.get("actions", [])
            failed = [a for a in actions if a.get("status") == "failed" and a.get("error")]
            if failed:
                failure_summary = "\n".join(
                    f"- {a.get('type', 'unknown')}: {a.get('error')}" for a in failed
                )
                context += f"\n\n[工具执行失败信息]\n{failure_summary}"
            
            state["context"] = context
            logger.info("上下文组装完成")
            return state
        except Exception as e:
            logger.error(f"上下文组装节点错误: {e}")
            state["error"] = str(e)
            state["context"] = "上下文组装失败。"
            return state
    
    async def response_generation_node(state: AgentState) -> AgentState:
        """响应生成节点"""
        try:
            from .knowledge_agent import KnowledgeAgent
            agent = KnowledgeAgent()
            
            query = state["query"]
            context = state["context"]
            history = state.get("conversation_history", [])
            
            response = await agent.generate_response(query, context, history)
            state["response"] = response
            
            logger.info("响应生成完成")
            return state
        except Exception as e:
            logger.error(f"响应生成节点错误: {e}")
            state["error"] = str(e)
            state["response"] = "抱歉，响应生成失败。"
            return state
    
    async def action_execution_node(state: AgentState) -> AgentState:
        """动作执行节点"""
        try:
            intent = state["intent_analysis"]
            
            if intent.get("requires_tools", False):
                # 实现MCP工具调用
                from .knowledge_agent import KnowledgeAgent
                agent = KnowledgeAgent()
                
                # 执行工具
                tool_result = await agent.execute_tool(intent)
                
                # 敏感字段脱敏
                def _redact(v: Any) -> Any:
                    if isinstance(v, dict):
                        out = {}
                        for k, val in v.items():
                            if k.lower() in {"password", "sender_password", "api_key", "token", "secret", "authorization"}:
                                out[k] = "***"
                            else:
                                out[k] = _redact(val)
                        return out
                    if isinstance(v, list):
                        return [_redact(x) for x in v]
                    return v
                
                if tool_result["success"]:
                    action = {
                        "type": intent.get("tool_category", "unknown"),
                        "status": "completed",
                        "result": tool_result["result"],
                        "tool_params": _redact(tool_result.get("tool_params", {})),
                    }
                    logger.info(f"工具执行成功: {action['type']}")
                    logger.debug(f"工具执行结果结构: {action['result']}")
                else:
                    action = {
                        "type": intent.get("tool_category", "unknown"),
                        "status": "failed",
                        "error": tool_result["error"],
                    }
                    logger.error(f"工具执行失败: {tool_result['error']}")
                    logger.debug(f"工具执行失败结构: {tool_result}")
                
                state["actions"] = [action]
                
                # 保存工具执行结果到状态中，等待上下文整合（仅成功时）
                if tool_result["success"]:
                    state["tool_result"] = tool_result
                else:
                    # 确保失败时不残留旧的成功结果
                    if "tool_result" in state:
                        del state["tool_result"]
            else:
                state["actions"] = []
            
            return state
        except Exception as e:
            logger.error(f"动作执行节点错误: {e}")
            state["error"] = str(e)
            state["actions"] = [{
                "type": "error",
                "status": "failed",
                "error": str(e)
            }]
            return state
    
    async def context_integration_node(state: AgentState) -> AgentState:
        """上下文整合节点 - 在工具执行后进行模型整合"""
        try:
            from .knowledge_agent import KnowledgeAgent
            agent = KnowledgeAgent()
            intent = state["intent_analysis"]
            tool_result = state.get("tool_result")
            
            if tool_result is not None:
                tool_category = intent.get("tool_category", "")
                context = state.get("context", "")
                documents = state.get("documents", [])
                
                # 使用新的智能上下文整合方法
                detailed_response = await agent.integrate_tool_result_with_context(
                    state["query"],
                    tool_result,
                    tool_category,
                    context,
                    documents
                )
                
                state["response"] = detailed_response
            
            return state
        except Exception as e:
            logger.error(f"上下文整合节点错误: {e}")
            state["error"] = str(e)
            return state
    
    def should_continue_after_response(state: AgentState) -> str:
        """决定响应生成后是否需要工具调用"""
        if state.get("error"):
            return "error_handling"
        
        intent = state.get("intent_analysis", {})
        if intent.get("requires_tools", False) and not state.get("actions"):
            return "action_execution"
        
        return "end"
    
    def should_continue_after_analysis(state: AgentState) -> str:
        """决定意图分析后的路由"""
        if state.get("error"):
            return "error_handling"
        
        intent = state.get("intent_analysis", {})
        # 如果需要工具调用，直接跳转到工具执行，跳过文档检索
        if intent.get("requires_tools", False):
            return "action_execution"
        
        # 否则进行正常的文档检索流程
        return "document_retrieval"
    
    def after_context_assembly(state: AgentState) -> str:
        """决定上下文组装后的路由：若存在tool_result则进行上下文整合，否则直接生成响应。"""
        if state.get("error"):
            return "error_handling"
        if state.get("tool_result") is not None:
            return "context_integration"
        return "response_generation"
    
    async def error_handling_node(state: AgentState) -> AgentState:
        """错误处理节点"""
        error = state.get("error", "未知错误")
        state["response"] = f"处理请求时发生错误: {error}"
        logger.error(f"工作流错误: {error}")
        return state
    
    # 构建工作流图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("document_retrieval", document_retrieval_node)
    workflow.add_node("context_assembly", context_assembly_node)
    workflow.add_node("response_generation", response_generation_node)
    workflow.add_node("action_execution", action_execution_node)
    workflow.add_node("context_integration", context_integration_node)
    workflow.add_node("error_handling", error_handling_node)
    
    # 设置入口点
    workflow.set_entry_point("query_analysis")
    
    # 添加边
    workflow.add_edge("document_retrieval", "context_assembly")
    # workflow.add_edge("context_assembly", "response_generation")  # 替换为条件路由
    
    # 添加条件边
    workflow.add_conditional_edges(
        "query_analysis",
        should_continue_after_analysis,
        {
            "document_retrieval": "document_retrieval",
            "action_execution": "action_execution",
            "error_handling": "error_handling"
        }
    )
    
    workflow.add_conditional_edges(
        "response_generation",
        should_continue_after_response,
        {
            "action_execution": "action_execution",
            "error_handling": "error_handling",
            "end": END
        }
    )
    
    # 上下文组装后的条件路由：有工具结果则去整合，否则直接生成响应
    workflow.add_conditional_edges(
        "context_assembly",
        after_context_assembly,
        {
            "context_integration": "context_integration",
            "response_generation": "response_generation",
            "error_handling": "error_handling"
        }
    )
    
    # 工具路径：先工具执行，再检索与上下文组装，最后整合
    # workflow.add_edge("action_execution", "context_integration")  # 改为执行后进行检索
    workflow.add_edge("action_execution", "document_retrieval")
    workflow.add_edge("context_integration", END)
    workflow.add_edge("error_handling", END)
    
    return workflow.compile()