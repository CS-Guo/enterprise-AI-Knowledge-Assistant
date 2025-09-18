import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PlannerAgent:
    """轻量规划智能体：基于意图与类别，给出后续协作策略与步骤。
    目标：尽量不引入额外LLM调用，使用启发式规则产出 plan。
    """

    async def plan(self, query: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        category = (intent or {}).get("category", "general")
        requires_tools = bool((intent or {}).get("requires_tools", False))
        tool_category = (intent or {}).get("tool_category", "none")
        intent_type = (intent or {}).get("intent_type", "question")

        steps = []
        strategy: Dict[str, Any] = {
            "use_rag": True,
            "use_tools": False,
            "force_open_domain": False,
            "disallow_open_domain": False,
            "rationale": "",
        }

        # 工具优先策略
        if requires_tools and tool_category and tool_category != "none":
            strategy["use_tools"] = True
            steps.append("分析参数并调用工具链")

        # 类别驱动的RAG策略
        if category in {"hr", "policy"}:
            strategy["use_rag"] = True
            steps.append("检索与重排序公司文档以构建证据")
        elif category in {"tech"}:
            strategy["use_rag"] = True
            steps.append("优先技术知识文档检索与重排序")
        else:
            # general 类别默认允许开放域作为兜底
            strategy["use_rag"] = True
            steps.append("尝试检索文档，不足时考虑开放域兜底")

        # 针对纯闲聊/百科问答的开放域倾向
        if intent_type == "question" and category == "general" and not requires_tools:
            strategy["force_open_domain"] = False  # 仍让RAG先试，命中不足时回退
            steps.append("若文档命中不足，允许开放域回答并标注来源与免责声明")

        # 对HR/Policy或明确需要工具的任务，禁用开放域回退，避免不合规或虚构
        if category in {"hr", "policy"} or requires_tools:
            strategy["disallow_open_domain"] = True
            steps.append("禁用开放域兜底，避免不合规或虚构内容")

        # 最后响应生成
        steps.append("综合证据/工具结果生成回答，必要时添加免责声明")

        strategy["rationale"] = (
            f"intent_type={intent_type}, category={category}, tools={tool_category}"
        )

        plan = {
            "steps": steps,
            "strategy": strategy,
            "intent_snapshot": intent,
            "query": query,
        }
        logger.info(f"Planner 生成计划: {plan['strategy']}")
        return plan