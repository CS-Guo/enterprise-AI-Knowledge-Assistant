from typing import Dict, Any, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode  # 未使用，暂时注释
import logging
from uuid import uuid4
from config.settings import settings
from .planner_agent import PlannerAgent
from .critic_agent import CriticAgent

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
    # 新增：开放域回退标记与来源/免责声明
    open_domain_fallback: bool
    provenance: str
    disclaimers: str
    # 新增：审阅建议与下一步动作
    critic_review: Dict[str, Any]
    # 新增：来源清单（从文档字符串中解析）
    sources: List[Dict[str, Any]]
    # 新增：会话记忆摘要（从 conversation_history 提炼）
    memory_summary: str


def create_workflow():
    """创建LangGraph工作流"""
    planner = PlannerAgent()
    critic = CriticAgent()

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

    async def planning_node(state: AgentState) -> AgentState:
        """规划节点：给出策略与步骤提示（不改变核心RAG/回退决策，仅提供hint）。"""
        try:
            intent = state.get("intent_analysis", {})
            plan = await planner.plan(state.get("query", ""), intent)
            state["plan"] = plan
            logger.info(f"规划完成: {plan['strategy']}")
            return state
        except Exception as e:
            logger.error(f"规划节点错误: {e}")
            state["plan"] = {"strategy": {}, "steps": []}
            return state

    async def document_retrieval_node(state: AgentState) -> AgentState:
        """文档检索节点"""
        try:
            from ..rag.retriever import DocumentRetriever
            retriever = DocumentRetriever()

            query = state.get("query", "")
            intent = state.get("intent_analysis", {})
            category = intent.get("category", "general")

            # 若 planner 要求使用工具优先，此处仍先进行检索，但可降低权重（后续节点使用）
            plan_strategy = (state.get("plan", {}) or {}).get("strategy", {})
            state["planner_suggest_use_tools"] = bool(plan_strategy.get("use_tools"))

            # 基于意图调整检索策略（容错处理），统一使用带重排序的检索以提升相关性
            if category == "hr":
                documents = await retriever.retrieve_with_rerank(query, filter_category="hr")
            elif category == "tech":
                documents = await retriever.retrieve_with_rerank(query, filter_category="tech")
            else:
                documents = await retriever.retrieve_with_rerank(query)

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
            intent = state.get("intent_analysis", {})
            category = intent.get("category", "general")

            # 计算是否需要开放域回退
            open_fallback = False
            if (len(documents) < settings.min_rag_hits \
                and settings.allow_open_domain_fallback \
                and category in set(settings.open_domain_allowed_categories)):
                open_fallback = True

            # 如果已有成功的工具执行结果，则不启用开放域回退，优先基于工具结果与上下文生成
            tr = state.get("tool_result")
            if tr and tr.get("success"):
                open_fallback = False

            # 根据 planner 策略进行约束或放宽
            plan_strategy = (state.get("plan", {}) or {}).get("strategy", {})
            if plan_strategy.get("disallow_open_domain"):
                open_fallback = False
            elif plan_strategy.get("force_open_domain") and len(documents) < settings.min_rag_hits:
                # 仅在文档命中不足时考虑强制开放域
                open_fallback = True

            # 结合 planner 策略：若明确建议工具优先，且无文档命中，可在后续优先工具
            if state.get("planner_suggest_use_tools") and len(documents) == 0:
                state["prefer_tools_due_to_plan"] = True

            # 组装上下文 + 解析来源
            context_parts = []
            sources: List[Dict[str, Any]] = []

            # 注入会话记忆（短期记忆）：从最近若干轮对话提炼摘要
            try:
                history = state.get("conversation_history", []) or []
                MAX_TURNS = 6
                selected = history[-MAX_TURNS:]
                mem_lines: List[str] = []
                for m in selected:
                    if not isinstance(m, dict):
                        continue
                    role = (m.get("role") or m.get("speaker") or "").lower()
                    text = m.get("content") or m.get("text") or m.get("message") or ""
                    text = str(text)
                    if not text:
                        continue
                    if len(text) > 200:
                        text = text[:200] + "..."
                    if role in ("user", "human"):
                        mem_lines.append(f"用户: {text}")
                    elif role in ("assistant", "ai", "bot"):
                        mem_lines.append(f"助手: {text}")
                    else:
                        mem_lines.append(f"消息: {text}")
                memory_summary = "\n".join(mem_lines)
                if len(memory_summary) > 800:
                    memory_summary = memory_summary[:800] + "..."
                if memory_summary:
                    context_parts.append(f"[会话记忆]\n{memory_summary}")
                state["memory_summary"] = memory_summary
            except Exception:
                state["memory_summary"] = ""

            # 插入 planner 提示，帮助生成遵循策略
            plan = state.get("plan", {}) or {}
            strategy = plan.get("strategy", {})
            steps = plan.get("steps", [])
            plan_tip = []
            if strategy:
                plan_tip.append("[Planner 策略摘要]")
                plan_tip.append(
                    f"use_rag={strategy.get('use_rag')}, use_tools={strategy.get('use_tools')}, "
                    f"disallow_open_domain={strategy.get('disallow_open_domain')}, force_open_domain={strategy.get('force_open_domain')}"
                )
            if steps:
                plan_tip.append("[Planner 建议步骤]")
                for s in steps[:3]:
                    plan_tip.append(f"- {s}")
            if plan_tip:
                context_parts.append("\n".join(plan_tip))

            for i, doc in enumerate(documents[:5]):  # 最多使用5个文档
                context_parts.append(f"文档 {i+1}:\n{doc}\n")
                # 从富文本内容中解析来源元数据（兼容 retriever.enriched_content 模式）
                try:
                    filename = None
                    category_m = None
                    sim = None
                    for line in (doc.splitlines() if isinstance(doc, str) else []):
                        line = line.strip()
                        if line.startswith("文档来源:"):
                            filename = line.split(":", 1)[1].strip()
                        elif line.startswith("文档类别:"):
                            category_m = line.split(":", 1)[1].strip()
                        elif line.startswith("相似度:"):
                            sim_str = line.split(":", 1)[1].strip()
                            try:
                                sim = float(sim_str)
                            except Exception:
                                sim = None
                    sources.append({
                        "index": i + 1,
                        "filename": filename or "未知",
                        "category": category_m or "未知",
                        "similarity": sim
                    })
                except Exception:
                    sources.append({"index": i + 1, "filename": "未知", "category": "未知", "similarity": None})

            # 整合工具执行结果（若有）
            tool_result = state.get("tool_result")
            if tool_result and tool_result.get("success"):
                # 控制上下文大小，截断
                import json
                tool_snippet = json.dumps(tool_result.get("result", {}), ensure_ascii=False)
                if len(tool_snippet) > 800:
                    tool_snippet = tool_snippet[:800] + "..."
                context_parts.append(f"[工具执行结果]\n类别: {tool_result.get('tool_category')}\n内容: {tool_snippet}")

            context = "\n".join(context_parts)

            if open_fallback:
                state["open_domain_fallback"] = True
                state["provenance"] = "public"
                state["disclaimers"] = "以下回答基于通用公开知识（非公司文档），可能与公司最新政策不一致，仅供参考。"
                # 为开放域回答设置前置信息，但不覆盖已拼接的文档与工具结果
                preface = "无内部文档命中。允许基于通用公开知识回答。"
                context = f"{preface}\n\n{context}" if context else preface

            # 如果前序存在工具失败，附加失败摘要，便于后续生成友好解释
            actions = state.get("actions", [])
            failed = [a for a in actions if a.get("status") == "failed" and a.get("error")]
            if failed:
                failure_summary = "\n".join(
                    f"- {a.get('type', 'unknown')}: {a.get('error')}" for a in failed
                )
                context += f"\n\n[工具执行失败信息]\n{failure_summary}"

            # 保存上下文与来源
            state["context"] = context
            state["sources"] = sources
            logger.info("上下文组装完成")
            return state
        except Exception as e:
            logger.error(f"上下文组装节点错误: {e}")
            state["error"] = str(e)
            state["context"] = ""
            state["sources"] = []
            state["memory_summary"] = ""
            return state

    async def response_generation_node(state: AgentState) -> AgentState:
        """响应生成节点（支持基于审阅建议的二次生成）"""
        try:
            from .knowledge_agent import KnowledgeAgent
            agent = KnowledgeAgent()

            query = state["query"]
            base_context = state.get("context", "")
            history = state.get("conversation_history", [])
            allow_open = bool(state.get("open_domain_fallback", False))
            disclaimers = state.get("disclaimers", "")

            # 将上一次审阅的 rework 指南注入上下文，辅助二次生成
            review = state.get("critic_review", {}) or {}
            rework_guidance = review.get("rework_guidance", "")
            context = base_context
            if rework_guidance:
                context = f"{base_context}\n\n[审阅建议]\n{rework_guidance}" if base_context else f"[审阅建议]\n{rework_guidance}"

            response = await agent.generate_response(query, context, history, allow_open_domain=allow_open, disclaimers=disclaimers)
            state["response"] = response

            # 仅在结尾追加不可见水印，避免影响以特定前缀开头的断言
            try:
                from config.settings import settings
                if settings.watermark_enabled and isinstance(state["response"], str) and state["response"]:
                    # 使用零宽空格与零宽不连字作为签名片段
                    watermark = "\u200b\u200c\u200b"  # zero-width space, zero-width non-joiner, zero-width space
                    state["response"] = state["response"] + watermark
            except Exception:
                # 即便水印失败也不影响主流程
                pass

            logger.info("响应生成完成")
            return state
        except Exception as e:
            logger.error(f"响应生成节点错误: {e}")
            state["error"] = str(e)
            state["response"] = "抱歉，响应生成失败。"
            return state

    async def critic_review_node(state: AgentState) -> AgentState:
        """批判审阅节点：生成后进行校对并给出下一步动作"""
        try:
            review = await critic.review(state)
            state["critic_review"] = review
            # 若审阅包含脱敏后的文本，优先使用脱敏结果
            redacted_resp = (review or {}).get("redacted_response")
            if isinstance(redacted_resp, str) and redacted_resp and redacted_resp != state.get("response", ""):
                state["response"] = redacted_resp
            # 若发现问题但可自动修复的（如缺少免责声明），直接在此补强
            if (not review.get("approved", True)) and state.get("open_domain_fallback"):
                if review and review.get("issues"):
                    tail = "\n\n【提示】此回答可能包含开放域信息，实际以公司最新制度为准。"
                    if tail not in state.get("response", ""):
                        state["response"] = (state.get("response", "") or "") + tail
                # 若无阻断型安全标记（毒性/PII未处理），则在开放域场景直接结束，避免返工循环
                flags = (review or {}).get("flags", {}) or {}
                has_blocking = bool(flags.get("toxicity")) or (bool(flags.get("pii")) and not bool(flags.get("pii_redacted")))
                if not has_blocking:
                    review["next_action"] = "finish"
                    # 若仅为提示类问题，视为通过
                    review["approved"] = True if len(review.get("issues", [])) == 0 else review.get("approved", True)
            # 封闭域且存在文档命中但回答未含引用标记时，自动追加来源尾注
            if (not state.get("open_domain_fallback")) and state.get("documents"):
                resp = state.get("response", "") or ""
                if ("来源" not in resp) and ("文档" not in resp):
                    sources = state.get("sources", []) or []
                    if sources:
                        refs = "; ".join(
                            f"[{s.get('index')}] {s.get('filename', '未知')}" for s in sources[:5]
                        )
                        citation_tail = f"\n\n来源：{refs}"
                        if citation_tail not in resp:
                            state["response"] = resp + citation_tail
                        # 既然已自动补齐引用，避免进入返工循环
                        review["next_action"] = "finish"
                        review["approved"] = True if len(review.get("issues", [])) == 0 else review.get("approved", True)
            # 基于审阅结果推进迭代计数与必要的状态重置（避免在路由器中变更状态）
            next_action = (review or {}).get("next_action", "finish")
            if next_action != "finish":
                state["iteration_count"] = int(state.get("iteration_count", 0) or 0) + 1
                if next_action == "rework_retrieval":
                    # 重新检索前清空旧文档与上下文
                    state["documents"] = []
                    state["context"] = ""
            logger.info("批判审阅完成")
            return state
        except Exception as e:
            logger.error(f"批判审阅节点错误: {e}")
            state["critic_review"] = {"approved": True, "issues": [], "suggestions": [], "next_action": "finish"}
            return state

    async def action_execution_node(state: AgentState) -> AgentState:
        """动作执行节点"""
        try:
            intent = state["intent_analysis"]
            query_text = state.get("query", "")

            # 是否应当执行工具：意图、planner 建议或启发式关键词
            def _should_use_tools() -> bool:
                if intent.get("requires_tools", False):
                    return True
                if state.get("planner_suggest_use_tools"):
                    return True
                # 任务型强触发关键词
                ql = query_text.lower()
                keywords = [
                    "安排会议", "创建会议", "会议", "日程", "预约", "日历",
                    "发送邮件", "发邮件", "email", "邮件通知"
                ]
                return any(k.lower() in ql for k in keywords)

            if not _should_use_tools():
                return state

            # 若意图未提供 tool_category，则基于查询启发式推断
            tool_category = (intent or {}).get("tool_category") or "none"
            if tool_category == "none" or not tool_category:
                lc = []
                ql = query_text.lower()
                if any(k in ql for k in ["安排会议", "创建会议", "会议", "日程", "预约", "日历"]):
                    lc.append("calendar")
                if any(k in ql for k in ["发送邮件", "发邮件", "email", "邮件通知"]):
                    lc.append("email")
                if any(k in ql for k in ["文件", "查找文件", "读取文件", "目录"]):
                    lc.append("file")
                tool_category = "|".join(lc) if lc else "none"
                # 回写到意图，供日志及后续节点参考
                intent["tool_category"] = tool_category

            if tool_category == "none":
                # 没有可执行工具，则直接返回
                return state

            # 实现 MCP 工具调用
            from .knowledge_agent import KnowledgeAgent
            agent = KnowledgeAgent()

            # 保证 original_query 存在
            if not intent.get("original_query"):
                intent["original_query"] = query_text

            # 执行工具
            tool_result = await agent.execute_tool(intent)

            # 敏感字段脱敏
            def _redact(params):
                if not isinstance(params, dict):
                    return params
                p = dict(params)
                if "sender_password" in p:
                    p["sender_password"] = "***"
                return p

            # 新增：处理需要确认的返回
            if tool_result.get("requires_confirmation"):
                action = {
                    "type": intent.get("tool_category", "unknown"),
                    "status": "needs_confirmation",
                    "reason": tool_result.get("reason", "该操作需要用户确认"),
                    "tool_category": tool_result.get("tool_category"),
                    "tool_name": tool_result.get("tool_name"),
                    "tool_params": _redact(tool_result.get("tool_params", {})),
                    "attempts": tool_result.get("attempts"),
                }
                state["actions"] = [action]
                # 将原始结果也挂到状态，供下游（如UI）消费
                state["tool_result"] = tool_result
                return state

            if tool_result.get("success"):
                action = {
                    "type": intent.get("tool_category", "unknown"),
                    "status": "completed",
                    "result": tool_result["result"],
                    "tool_params": _redact(tool_result.get("tool_params", {})),
                    "attempts": tool_result.get("attempts"),
                }
                logger.info(f"工具执行成功: {action['type']}")
                logger.debug(f"工具执行结果结构: {action['result']}")
            else:
                err_msg = tool_result.get("error") or tool_result.get("reason") or "工具执行失败"
                action = {
                    "type": intent.get("tool_category", "unknown"),
                    "status": "failed",
                    "error": err_msg,
                    "attempts": tool_result.get("attempts"),
                }
                logger.error(f"工具执行失败: {err_msg}")
                logger.debug(f"工具执行失败结构: {tool_result}")

            # 写回状态
            state.setdefault("actions", [])
            state["actions"].append(action)
            if tool_result.get("success"):
                state["tool_result"] = tool_result
            return state
        except Exception as e:
            logger.error(f"动作执行节点错误: {e}")
            state["error"] = str(e)
            return state

    async def response_generation_node_dup(state: AgentState) -> AgentState:
        """响应生成节点"""
        try:
            agent = KnowledgeAgent()
            query = state.get("query", "")
            context = state.get("context", "")
            history = state.get("conversation_history", [])
            allow_open = bool(state.get("open_domain_fallback", False))
            disclaimers = state.get("disclaimers", "")

            # 若审阅提出返工建议，附加至上下文供生成模型参考
            review = state.get("critic_review", {}) or {}
            base_context = context
            rework_guidance = review.get("rework_guidance", "")
            if rework_guidance:
                context = f"{base_context}\n\n[审阅建议]\n{rework_guidance}" if base_context else f"[审阅建议]\n{rework_guidance}"

            response = await agent.generate_response(query, context, history, allow_open_domain=allow_open, disclaimers=disclaimers)
            state["response"] = response

            logger.info("响应生成完成")
            return state
        except Exception as e:
            logger.error(f"响应生成节点错误: {e}")
            state["error"] = str(e)
            state["response"] = "抱歉，响应生成失败。"
            return state

    async def critic_review_node_dup(state: AgentState) -> AgentState:
        """批判审阅节点：生成后进行校对并给出下一步动作"""
        try:
            review = await critic.review(state)
            state["critic_review"] = review
            # 若审阅包含脱敏后的文本，优先使用脱敏结果
            redacted_resp = (review or {}).get("redacted_response")
            if isinstance(redacted_resp, str) and redacted_resp and redacted_resp != state.get("response", ""):
                state["response"] = redacted_resp
            # 若发现问题但可自动修复的（如缺少免责声明），直接在此补强
            if (not review.get("approved", True)) and state.get("open_domain_fallback"):
                if review and review.get("issues"):
                    tail = "\n\n【提示】此回答可能包含开放域信息，实际以公司最新制度为准。"
                    if tail not in state.get("response", ""):
                        state["response"] = (state.get("response", "") or "") + tail
            # 封闭域且存在文档命中但回答未含引用标记时，自动追加来源尾注
            if (not state.get("open_domain_fallback")) and state.get("documents"):
                resp = state.get("response", "") or ""
                if ("来源" not in resp) and ("文档" not in resp):
                    sources = state.get("sources", []) or []
                    if sources:
                        refs = "; ".join(
                            f"[{s.get('index')}] {s.get('filename', '未知')}" for s in sources[:5]
                        )
                        citation_tail = f"\n\n来源：{refs}"
                        if citation_tail not in resp:
                            state["response"] = resp + citation_tail
                        # 既然已自动补齐引用，避免进入返工循环
                        review["next_action"] = "finish"
                        review["approved"] = True if len(review.get("issues", [])) == 0 else review.get("approved", True)
            # 基于审阅结果推进迭代计数与必要的状态重置（避免在路由器中变更状态）
            next_action = (review or {}).get("next_action", "finish")
            if next_action != "finish":
                state["iteration_count"] = int(state.get("iteration_count", 0) or 0) + 1
                if next_action == "rework_retrieval":
                    # 重新检索前清空旧文档与上下文
                    state["documents"] = []
                    state["context"] = ""
            logger.info("批判审阅完成")
            return state
        except Exception as e:
            logger.error(f"批判审阅节点错误: {e}")
            state["critic_review"] = {"approved": True, "issues": [], "suggestions": [], "next_action": "finish"}
            return state

    # 构建图
    workflow = StateGraph(AgentState)

    # 节点注册
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("document_retrieval", document_retrieval_node)
    workflow.add_node("context_assembly", context_assembly_node)
    workflow.add_node("response_generation", response_generation_node)
    workflow.add_node("critic_review", critic_review_node)
    workflow.add_node("action_execution", action_execution_node)

    # 边注册：标准RAG闭环 + 规划/审阅
    workflow.add_edge("planning", "query_analysis")
    workflow.add_edge("query_analysis", "document_retrieval")

    # 条件路由：检索后根据意图/规划决定是否先走工具
    def route_after_retrieval(state: AgentState) -> str:
        intent = state.get("intent_analysis", {}) or {}
        # 1) 明确需要工具
        if intent.get("requires_tools"):
            return "tools"
        # 2) 规划建议优先使用工具
        if state.get("planner_suggest_use_tools"):
            return "tools"
        # 3) 基于启发式的强触发关键词（会议/邮件等任务型）
        q = (state.get("query") or "").lower()
        heuristic_tool_keywords = [
            "安排会议", "创建会议", "会议", "日程", "预约", "日历",
            "发送邮件", "发邮件", "email", "邮件通知"
        ]
        if any(k.lower() in q for k in heuristic_tool_keywords):
            return "tools"
        return "context"

    workflow.add_conditional_edges(
        "document_retrieval",
        route_after_retrieval,
        {
            "tools": "action_execution",
            "context": "context_assembly",
        },
    )

    # 工具执行后统一进入上下文组装 -> 生成 -> 审阅
    workflow.add_edge("action_execution", "context_assembly")
    workflow.add_edge("context_assembly", "response_generation")
    workflow.add_edge("response_generation", "critic_review")

    # 审阅后的闭环路由：根据 critic 输出的 next_action 返工或结束
    def route_after_review(state: AgentState) -> str:
        # 迭代上限守卫（此处只读，不修改状态）
        iter_count = int(state.get("iteration_count", 0) or 0)
        if iter_count >= settings.max_iterations:
            return "end"
        review = state.get("critic_review", {}) or {}
        action = review.get("next_action", "finish")
        if action == "rework_response":
            return "response_generation"
        if action == "rework_retrieval":
            return "document_retrieval"
        if action == "rework_tools":
            return "action_execution"
        return "end"

    workflow.add_conditional_edges(
        "critic_review",
        route_after_review,
        {
            "response_generation": "response_generation",
            "document_retrieval": "document_retrieval",
            "action_execution": "action_execution",
            "end": END,
        },
    )

    # 入口与结束
    workflow.set_entry_point("planning")

    return workflow.compile()