import logging
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from config.settings import settings

logger = logging.getLogger(__name__)

class KnowledgeAgent:
    """企业知识助手的核心Agent类"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key,
            base_url=settings.base_url
        )
        # 初始化工具
        self._init_tools()
    
    def _init_tools(self):
        """初始化MCP工具"""
        try:
            from ..mcp.file_tools import FileTools
            from ..mcp.email_tools import EmailTools
            from ..mcp.calendar_tools import CalendarTools
            
            self.file_tools = FileTools()
            self.email_tools = EmailTools()
            self.calendar_tools = CalendarTools()
            
            self.tools_map = {
                "file": self.file_tools,
                "email": self.email_tools,
                "calendar": self.calendar_tools
            }
            logger.info("MCP工具初始化成功")
        except Exception as e:
            logger.error(f"MCP工具初始化失败: {e}")
            self.tools_map = {}
    
    async def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """分析用户查询的意图"""
        system_prompt = """你是一个查询意图分析专家。请严格按照以下要求分析用户输入，并只输出一个JSON对象（不要额外解释、前后缀、代码块围栏）。
        
        你需要完成：
        - 判断该请求是提问/任务/搜索/求助
        - 是否需要调用工具；如需要，指出哪些工具（支持复合，使用“|”分隔）
        - 提取关键实体（如人名、时间、邮箱、文件名等）
        - 推断业务类别（hr/tech/policy/general）
        
        返回字段：
        {
            "intent_type": "question|task|search|help",
            "confidence": 0.0-1.0,
            "entities": ["关键实体1", "关键实体2"],
            "requires_tools": true/false,
            "tool_category": "file|email|calendar|none" 或 复合如 "calendar|email",
            "category": "hr|tech|policy|general",
            "action_needed": "需要执行的动作简述"
        }
        
        规则：
        - 仅输出有效JSON；不要输出任何解释性文字
        - 无法确定是否需要工具时，优先 requires_tools=false
        - 同时需要多个工具时用“|”连接，例如 "calendar|email"。
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"请分析这个查询: {query}")
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            # 尝试解析JSON响应
            import json
            import re
            
            # 提取JSON部分
            json_match = re.search(r'\{[^}]*\}', response.content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    result["original_query"] = query
                    return result
                except json.JSONDecodeError:
                    pass
            
            # 如果JSON解析失败，使用关键词匹配
            return self._fallback_intent_analysis(query)
            
        except Exception as e:
            logger.error(f"查询意图分析失败: {e}")
            return self._fallback_intent_analysis(query)
    
    def _fallback_intent_analysis(self, query: str) -> Dict[str, Any]:
        """备用意图分析方法"""
        query_lower = query.lower()
        
        # 工具调用关键词检测
        file_keywords = ['搜索文件', '查找文件', '读取文件', '创建文件', '文件', '目录']
        email_keywords = ['发送邮件', '邮件', '发邮件', '发送', 'email', '通知']
        calendar_keywords = ['安排会议', '创建会议', '会议', '日程', '预约', '约会', '时间安排']
        
        requires_tools = False
        tool_categories = []
        intent_type = "question"
        
        # 检测各种工具需求
        if any(keyword in query_lower for keyword in file_keywords):
            requires_tools = True
            tool_categories.append("file")
            intent_type = "task"
        
        if any(keyword in query_lower for keyword in email_keywords):
            requires_tools = True
            tool_categories.append("email")
            intent_type = "task"
            
        if any(keyword in query_lower for keyword in calendar_keywords):
            requires_tools = True
            tool_categories.append("calendar")
            intent_type = "task"
        
        # 确定最终的tool_category
        if len(tool_categories) > 1:
            tool_category = "|".join(tool_categories)
        elif len(tool_categories) == 1:
            tool_category = tool_categories[0]
        else:
            tool_category = "none"
        
        return {
            "intent_type": intent_type,
            "confidence": 0.7 if requires_tools else 0.8,
            "entities": [],
            "requires_tools": requires_tools,
            "tool_category": tool_category,
            "category": "general",
            "action_needed": query if requires_tools else "",
            "original_query": query
        }
    
    async def generate_response(self, query: str, context: str, conversation_history: List[Dict] = None, allow_open_domain: bool = False, disclaimers: str = "") -> str:
        """基于上下文生成回答；支持在无命中时开放域回退"""
        if conversation_history is None:
            conversation_history = []
            
        # 构建对话历史
        history_text = ""
        for item in conversation_history[-3:]:  # 只保留最近3轮对话
            history_text += f"用户: {item.get('query', '')}\n助手: {item.get('response', '')}\n\n"
        
        if allow_open_domain:
            system_prompt = f"""你是一个专业的企业知识助手。当前无内部文档命中，允许基于通用公开知识回答，但必须遵守以下限制：
1) 禁止编造公司内部政策、流程、数据；
2) 如问题明显涉及公司特定信息，应拒答并给出引导；
3) 对通用/公开领域问题，给出简洁权威的解释，并在结尾附上“来源：公开知识（非公司文档）”；

对话历史：
{history_text}

请遵循以下原则：
- 先给结论，再解释；
- 用中文、专业且自然；
- 可提出1个澄清问题（可选）。
{('附加说明：' + disclaimers) if disclaimers else ''}
"""
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
        else:
            system_prompt = f"""你是一个专业的企业知识助手。请基于以下上下文回答用户问题：

上下文信息：
{context}

对话历史：
{history_text}

请遵循以下原则：
1. 先给出简明结论；若上下文不足以回答，请坦诚说明，并提出最多1个澄清问题（可选）。
2. 仅依据提供的上下文与对话历史，不要编造信息。
3. 用中文、专业且自然。
4. 如能从上下文提炼出处，请在结尾以“参考：<来源/标题>”列出最多3条。
{('附加说明：' + disclaimers) if disclaimers else ''}
"""
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
        
        try:
            response = await self.llm.ainvoke(messages)
            content = response.content
            # 开放域回答时强制附加来源提示
            if allow_open_domain:
                if "公开知识" not in content:
                    content += "\n\n来源：公开知识（非公司文档）"
            return content
        except Exception as e:
            logger.error(f"响应生成失败: {e}")
            return "抱歉，我在处理您的请求时遇到了问题。请稍后重试。"
    
    # --- 安全策略与脱敏工具函数 ---
    def _is_category_whitelisted(self, category: str) -> bool:
        wl = set(getattr(settings, "tool_whitelist_categories", []) or [])
        return category in wl

    def _is_tool_blacklisted(self, category: str, tool_name: str) -> bool:
        bl = set(getattr(settings, "tool_blacklist", []) or [])
        return f"{category}:{tool_name}" in bl

    def _needs_confirmation(self, category: str, tool_name: str, params: Dict[str, Any]) -> bool:
        # 类别-工具级确认
        by_cat = getattr(settings, "tool_confirm_required", {}) or {}
        needs = tool_name in (by_cat.get(category, []) or [])
        if needs:
            return True
        # 具体动作级确认
        by_actions = getattr(settings, "tool_confirm_actions", {}) or {}
        action_map = (by_actions.get(category, {}) or {}).get(tool_name, [])
        action = (params or {}).get("action")
        if action and action_map and action in action_map:
            return True
        return False

    def _redact_sensitive(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return params
        p = dict(params)
        if "sender_password" in p:
            p["sender_password"] = "***"
        return p
    
    async def execute_tool(self, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """根据意图分析结果执行相应的工具"""
        tool_category = intent_analysis.get("tool_category", "none")
        query = intent_analysis.get("original_query", "")
        
        # 处理复合工具类型（如 calendar|email）
        if "|" in tool_category:
            try:
                composite = await self._execute_composite_tools(tool_category, query, intent_analysis)
                return composite
            except Exception as e:
                logger.error(f"复合工具执行失败: {e}")
                return {"success": False, "error": str(e), "tool_category": tool_category}
        
        if tool_category == "none" or tool_category not in self.tools_map:
            return {
                "success": False,
                "error": f"不支持的工具类型: {tool_category}",
                "tool_category": tool_category
            }
        
        # 白名单校验
        if not self._is_category_whitelisted(tool_category):
            return {
                "success": False,
                "error": f"工具类别未授权: {tool_category}",
                "tool_category": tool_category
            }
        
        try:
            # 根据查询内容确定具体的工具和参数
            tool_params = await self._extract_tool_parameters(query, tool_category)
            
            # 若未给出具体工具名，按类别设默认
            if "tool_name" not in tool_params:
                default_tool = {
                    "file": "file_search",
                    "email": "email_send",
                    "calendar": "calendar_event",
                }.get(tool_category)
                if default_tool:
                    tool_params["tool_name"] = default_tool
            
            tool_instance = self.tools_map[tool_category]
            tool_name = tool_params.get("tool_name")
            if not tool_name:
                raise ValueError("缺少必要的tool_name参数")
            
            # 黑名单拦截
            if self._is_tool_blacklisted(tool_category, tool_name):
                return {
                    "success": False,
                    "error": f"工具被禁用: {tool_category}:{tool_name}",
                    "tool_category": tool_category,
                    "tool_name": tool_name
                }
            # 确认拦截（不执行，返回待确认）
            if self._needs_confirmation(tool_category, tool_name, tool_params):
                return {
                    "success": False,
                    "requires_confirmation": True,
                    "reason": "该操作需要用户确认",
                    "tool_category": tool_category,
                    "tool_name": tool_name,
                    "tool_params": self._redact_sensitive(tool_params),
                }
            
            # 邮件发送在本地开发环境下启用 dry-run
            if tool_category == "email":
                sender_email = tool_params.get("sender_email")
                sender_password = tool_params.get("sender_password")
                if not sender_email or not sender_password:
                    # 模拟发送成功，返回可追溯结构
                    simulated = {
                        "status": "dry_run",
                        "to_addresses": tool_params.get("to_addresses", []),
                        "subject": tool_params.get("subject", ""),
                        "body": tool_params.get("body", ""),
                        "note": "开发环境未配置SMTP凭据，已模拟发送。"
                    }
                    return {
                        "success": True,
                        "result": simulated,
                        "tool_category": tool_category,
                        "tool_params": {k: v for k, v in tool_params.items() if k not in {"sender_password"}}
                    }
            
            result = await tool_instance.execute_tool(tool_name, **{k: v for k, v in tool_params.items() if k != "tool_name"})
            
            # 规范化失败场景：部分底层工具返回 False
            if result is False or result is None:
                raise RuntimeError(f"底层工具执行失败或无结果: {tool_name}")
            
            logger.info(f"工具执行成功: {tool_category}, 参数: {tool_params}")
            return {
                "success": True,
                "result": result,
                "tool_category": tool_category,
                "tool_params": {k: v for k, v in tool_params.items() if k != "sender_password"}
            }
        except Exception as e:
            logger.error(f"工具执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_category": tool_category
            }

    # --- 新增：参数抽取与复合工具执行 ---
    async def _extract_tool_parameters(self, query: str, tool_category: str) -> Dict[str, Any]:
        """基于启发式从自然语言中抽取工具参数（简化实现）。"""
        import re
        from datetime import datetime, timedelta
        params: Dict[str, Any] = {}
        q = query.strip()
        
        if tool_category == "calendar":
            # 识别人名（示例：和张三/与张三/找张三/与XXX的会议）
            name_match = re.search(r"(?:和|与|找)([\u4e00-\u9fa5]{2,4})", q)
            person = name_match.group(1) if name_match else "对方"
            title = f"与{person}会议"
            # 识别时间（明天/后天 + 上午/下午 + X点）
            day_offset = 0
            if "后天" in q:
                day_offset = 2
            elif "明天" in q:
                day_offset = 1
            # 小时
            hour = None
            hour_match = re.search(r"(\d{1,2})点", q)
            if hour_match:
                hour = int(hour_match.group(1))
            # 上下午
            is_pm = ("下午" in q) or ("pm" in q.lower())
            if hour is not None and is_pm and hour < 12:
                hour += 12
            if hour is None:
                hour = 10  # 默认10点
            # 构造起止时间（默认60分钟）
            start_dt = (datetime.now() + timedelta(days=day_offset)).replace(hour=hour, minute=0, second=0, microsecond=0)
            end_dt = start_dt + timedelta(minutes=60)
            params.update({
                "tool_name": "calendar_event",
                "action": "create",
                "title": title,
                "start_time": start_dt.isoformat(),
                "end_time": end_dt.isoformat(),
                "attendees": [],
                "description": f"自动创建：{q}",
            })
        elif tool_category == "email":
            # 识别收件人邮箱（简单正则）
            import re as _re
            emails = _re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", q)
            to_addresses = emails if emails else ["zhangsan@example.com"]
            # 主题与正文（若缺失，后续可由模板工具生成）
            subject = "会议通知"
            body = q
            params.update({
                "tool_name": "email_send",
                "to_addresses": to_addresses,
                "subject": subject,
                "body": body,
                # 发件人凭据留空 -> 触发 dry-run
                "sender_email": "",
                "sender_password": "",
            })
        elif tool_category == "file":
            params.update({
                "tool_name": "file_search",
                "directory": "./",
                "filename_pattern": "*",
                "recursive": True,
            })
        return params

    async def _execute_composite_tools(self, tool_category: str, query: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """执行复合工具链（例如 calendar|email）。返回聚合结果。"""
        chain = [c.strip() for c in tool_category.split("|") if c.strip()]
        steps: List[Dict[str, Any]] = []
        context: Dict[str, Any] = {}
        
        for cat in chain:
            if cat not in self.tools_map:
                raise ValueError(f"未知工具类别: {cat}")
            # 白名单校验
            if not self._is_category_whitelisted(cat):
                steps.append({"category": cat, "status": "blocked", "reason": f"工具类别未授权: {cat}"})
                continue

            tool_params = await self._extract_tool_parameters(query, cat)
            # 若已创建日历事件，将其信息传递给后续邮件模板
            if cat == "email" and context.get("calendar_event"):
                event = context["calendar_event"].get("event", {})
                # 先尝试模板生成
                try:
                    template_vars = {
                        "meeting_title": event.get("title", "会议"),
                        "meeting_time": f"{event.get('start_time', '')} - {event.get('end_time', '')}",
                        "meeting_location": event.get("location", "线上/待定"),
                        "agenda": "沟通项目事项",
                    }
                    tmpl = await self.email_tools.execute_tool("email_template", template_name="meeting_invite", template_vars=template_vars)
                    if tmpl:
                        tool_params["subject"] = tmpl.get("subject", tool_params.get("subject"))
                        tool_params["body"] = tmpl.get("body", tool_params.get("body"))
                except Exception as e:
                    logger.warning(f"邮件模板生成失败，使用原文：{e}")
            
            tool_name = tool_params.get("tool_name")
            if not tool_name:
                steps.append({"category": cat, "status": "failed", "error": "缺少tool_name"})
                continue
            # 黑名单与确认拦截
            if self._is_tool_blacklisted(cat, tool_name):
                steps.append({"category": cat, "status": "blocked", "reason": f"工具被禁用: {cat}:{tool_name}"})
                continue
            if self._needs_confirmation(cat, tool_name, tool_params):
                steps.append({
                    "category": cat,
                    "status": "needs_confirmation",
                    "tool_name": tool_name,
                    "params": self._redact_sensitive(tool_params),
                    "reason": "该操作需要用户确认"
                })
                continue
            
            result = await self.tools_map[cat].execute_tool(tool_name, **{k: v for k, v in tool_params.items() if k != "tool_name"})
            if result is False or result is None:
                steps.append({"category": cat, "status": "failed", "error": f"{tool_name} 执行失败"})
                # 不中断，继续尝试后续步骤
                continue
            
            steps.append({"category": cat, "status": "ok", "result": result})
            if cat == "calendar":
                context["calendar_event"] = result
            elif cat == "email":
                context["email_send"] = result
            elif cat == "file":
                context["file"] = result
        
        # 汇总
        success_any = any(s.get("status") == "ok" for s in steps)
        summary = {
            "executed": chain,
            "steps": steps,
            "context": context,
        }
        return {"success": success_any, "result": summary, "tool_category": tool_category}
