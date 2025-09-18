import logging
from typing import Dict, Any, List
import re

logger = logging.getLogger(__name__)

class CriticAgent:
    """轻量批判智能体：对生成的回答进行规则化审阅，给出问题与建议修正。
    不额外调用LLM，避免时延增长。
    """

    # 轻量敏感信息与不当用语检测词典/模式（可按需扩展）
    _TOXIC_TERMS = {
        "傻逼", "去死", "垃圾", "蠢", "废物",
    }
    _REGEX_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    _REGEX_PHONE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")  # 中国大陆手机号
    _REGEX_IDCARD = re.compile(r"(?<![0-9A-Za-z])[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?![0-9A-Za-z])")
    _REGEX_CARD = re.compile(r"(?<!\d)\d{13,19}(?!\d)")  # 潜在银行卡号/长数字串

    def _detect_toxicity(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        return any(term in t for term in (w.lower() for w in self._TOXIC_TERMS))

    def _redact_pii(self, text: str) -> Dict[str, Any]:
        """返回脱敏后的文本与是否命中标记。"""
        if not text:
            return {"changed": False, "text": text, "hits": []}
        hits: List[str] = []
        redacted = text
        # 邮箱
        if self._REGEX_EMAIL.search(redacted):
            hits.append("email")
            redacted = self._REGEX_EMAIL.sub("[已脱敏:email]", redacted)
        # 手机
        if self._REGEX_PHONE.search(redacted):
            hits.append("phone")
            redacted = self._REGEX_PHONE.sub(lambda m: f"[已脱敏:phone ****{m.group(1)[-2:]}]", redacted)
        # 身份证
        if self._REGEX_IDCARD.search(redacted):
            hits.append("id")
            redacted = self._REGEX_IDCARD.sub("[已脱敏:id]", redacted)
        # 银行卡/长数字串（保守处理）
        if self._REGEX_CARD.search(redacted):
            hits.append("card")
            redacted = self._REGEX_CARD.sub("[已脱敏:card]", redacted)
        return {"changed": redacted != text, "text": redacted, "hits": hits}

    async def review(self, state: Dict[str, Any]) -> Dict[str, Any]:
        response: str = state.get("response", "") or ""
        context: str = state.get("context", "") or ""
        intent: Dict[str, Any] = state.get("intent_analysis", {}) or {}
        documents: List[str] = state.get("documents", []) or []
        open_fallback = bool(state.get("open_domain_fallback", False))
        actions: List[Dict[str, Any]] = state.get("actions", []) or []

        issues: List[str] = []
        suggestions: List[str] = []

        # 0) 轻量安全检查：不当用语与PII
        toxic = self._detect_toxicity(response)
        if toxic:
            issues.append("回答可能包含不当用语")
            suggestions.append("请使用更为专业与中性的表述，避免攻击性词汇")
        pii_result = self._redact_pii(response)
        if pii_result.get("changed"):
            suggestions.append("检测到可能的个人敏感信息，已进行脱敏处理")

        # 1) 安全一致性：开放域回答时需有免责声明
        if open_fallback:
            disclaimers = state.get("disclaimers", "")
            if not disclaimers or "公开" not in disclaimers:
                issues.append("开放域回答缺少充分的免责声明")
                suggestions.append("在回答尾部追加开放域免责声明，提示与公司最新政策可能不一致")

        # 2) 证据可追溯：若非开放域且引用了文档，应包含引用片段或编号
        if not open_fallback and documents:
            if "文档" not in response and "来源" not in response:
                suggestions.append("在回答中加入简短的文档引用标记（如：参见文档1/2）")

        # 3) 格式与清晰度
        if len(response.strip()) < 8:
            issues.append("回答过短")
            suggestions.append("补充关键要点，至少给出2-3条分点说明")

        # 4) 工具执行情况与意图一致性
        requires_tools = bool(intent.get("requires_tools", False))
        tool_success = any(a.get("status") == "completed" for a in actions)
        tool_failed = any(a.get("status") == "failed" for a in actions)

        # 5) 检索命中情况
        no_docs = len(documents) == 0

        # 决策 next_action（finish / rework_response / rework_retrieval / rework_tools）
        next_action = "finish"

        # 优先保证工具要求
        if requires_tools and not tool_success:
            next_action = "rework_tools"
            if tool_failed:
                suggestions.append("上一次工具调用失败，请检查参数并重试（时间、人名、邮箱等关键字段是否缺失）")
            else:
                suggestions.append("需要调用相应工具完成任务，请先进行参数抽取与校验后再调用")
        # 若需要证据但未体现引用标记，优先修正文案
        elif (not open_fallback and documents) and ("文档" not in response and "来源" not in response):
            next_action = "rework_response"
        # 若回答明显过短或不完整，修正文案（仅在有文档或开放域时触发；无文档且非开放域时直接通过，避免死循环）
        elif len(response.strip()) < 30:
            if open_fallback or (not open_fallback and documents):
                next_action = "rework_response"
            else:
                next_action = "finish"
        # 若既无文档也无成功工具，则直接结束，避免无意义的反复重检索（HR等受限域可能不允许开放域回退）
        elif no_docs and not tool_success and not open_fallback:
            next_action = "finish"
        else:
            next_action = "finish"

        result: Dict[str, Any] = {
            "issues": issues,
            "suggestions": suggestions,
            "approved": next_action == "finish" and len(issues) == 0,
            "next_action": next_action,
            "rework_guidance": "\n".join(f"- {s}" for s in suggestions) if suggestions else "",
            "flags": {
                "toxicity": bool(toxic),
                "pii": bool(pii_result.get("changed")),
            },
        }
        if pii_result.get("changed"):
            result["redacted_response"] = pii_result.get("text", response)
        return result