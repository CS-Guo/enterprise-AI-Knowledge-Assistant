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
    
    async def generate_response(self, query: str, context: str, conversation_history: List[Dict] = None) -> str:
        """基于上下文生成回答"""
        if conversation_history is None:
            conversation_history = []
            
        # 构建对话历史
        history_text = ""
        for item in conversation_history[-3:]:  # 只保留最近3轮对话
            history_text += f"用户: {item.get('query', '')}\n助手: {item.get('response', '')}\n\n"
        
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
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"响应生成失败: {e}")
            return "抱歉，我在处理您的请求时遇到了问题。请稍后重试。"
    
    async def execute_tool(self, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """根据意图分析结果执行相应的工具"""
        tool_category = intent_analysis.get("tool_category", "none")
        query = intent_analysis.get("original_query", "")
        
        # 处理复合工具类型（如 calendar|email）
        if "|" in tool_category:
            return await self._execute_composite_tools(tool_category, query, intent_analysis)
        
        if tool_category == "none" or tool_category not in self.tools_map:
            return {
                "success": False,
                "error": f"不支持的工具类型: {tool_category}",
                "tool_category": tool_category
            }
        
        try:
            # 根据查询内容确定具体的工具和参数
            tool_params = await self._extract_tool_parameters(query, tool_category)
            
            # 执行工具
            tool_instance = self.tools_map[tool_category]
            # 确保传递tool_name参数
            tool_name = tool_params.get("tool_name")
            if not tool_name:
                raise ValueError(f"缺少必要的tool_name参数")
            
            # 新增：邮件模板+发送链式处理（单类别执行场景）
            if tool_category == "email":
                template_info = tool_params.get("template")
                if isinstance(template_info, dict):
                    try:
                        template_result = await tool_instance.execute_tool("email_template", **template_info)
                        if template_result:
                            # 仅在主题/正文缺失时由模板填充
                            if not tool_params.get("subject"):
                                tool_params["subject"] = template_result.get("subject")
                            if not tool_params.get("body"):
                                tool_params["body"] = template_result.get("body")
                    except Exception as e:
                        logger.error(f"邮件模板生成失败: {e}")
                        # 不中断流程，继续尝试发送默认内容
            
            result = await tool_instance.execute_tool(tool_name, **{k: v for k, v in tool_params.items() if k != "tool_name"})
            
            logger.info(f"工具执行成功: {tool_category}, 参数: {tool_params}")
            return {
                "success": True,
                "result": result,
                "tool_category": tool_category,
                "tool_params": tool_params
            }
            
        except Exception as e:
            logger.error(f"工具执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_category": tool_category
            }
    
    async def _execute_composite_tools(self, tool_category: str, query: str, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """执行复合工具操作（如 calendar|email）"""
        try:
            tool_categories = [cat.strip() for cat in tool_category.split("|")]
            results = []
            all_success = True
            meeting_info = None  # 存储会议信息用于邮件生成
            
            # 顺序执行每个工具
            for category in tool_categories:
                if category not in self.tools_map:
                    logger.error(f"不支持的工具类别: {category}")
                    all_success = False
                    continue
                
                # 特殊处理：如果是邮件工具且之前创建了会议，使用会议信息生成邮件
                if category == "email" and meeting_info:
                    tool_params = await self._extract_email_parameters_with_meeting(query, meeting_info)
                else:
                    # 提取该工具的参数
                    tool_params = await self._extract_tool_parameters(query, category)
                
                # 特殊处理：如果是多会议创建
                if category == "calendar" and "_multiple_meetings" in tool_params:
                    # 安全地移除特殊标记并创建副本
                    meetings = tool_params.pop("_multiple_meetings")  # 移除特殊标记
                    tool_instance = self.tools_map[category]
                    
                    # 创建多个会议
                    meeting_results = []
                    for meeting_params in meetings:
                        # 移除特殊标记（如果存在）并创建参数的副本
                        clean_params = {k: v for k, v in meeting_params.items() if not k.startswith("_")}
                        # 确保传递tool_name参数
                        tool_name = clean_params.get("tool_name")
                        if not tool_name:
                            raise ValueError(f"缺少必要的tool_name参数")
                            
                        meeting_result = await tool_instance.execute_tool(tool_name, **{k: v for k, v in clean_params.items() if k != "tool_name"})
                        meeting_results.append(meeting_result)
                        
                        # 保存第一个会议信息用于邮件
                        if not meeting_info and meeting_result.get("success"):
                            meeting_info = meeting_result.get("result", {}).get("event", {}).copy() if meeting_result.get("result", {}).get("event") else {}
                    
                    # 合并多个会议的结果（创建副本避免循环引用）
                    successful_meetings = []
                    for r in meeting_results:
                        if r.get("success"):
                            event = r.get("result", {}).get("event", {})
                            if event:
                                successful_meetings.append(event.copy())
                    
                    result = {
                        "success": all(r.get("success", False) for r in meeting_results),
                        "result": {
                            "action": "create_multiple",
                            "meetings": successful_meetings,
                            "total_created": len(successful_meetings)
                        }
                    }
                    
                    # 为邮件生成准备会议信息（创建新字典避免循环引用）
                    if successful_meetings:
                        meeting_info = {
                            "meetings": [m.copy() for m in successful_meetings],
                            "total_created": len(successful_meetings)
                        }
                else:
                    # 执行单个工具
                    tool_instance = self.tools_map[category]
                    
                    # 新增：在复合场景中支持邮件模板+发送链
                    if category == "email":
                        template_info = tool_params.get("template")
                        if isinstance(template_info, dict):
                            try:
                                template_result = await tool_instance.execute_tool("email_template", **template_info)
                                if template_result:
                                    if not tool_params.get("subject"):
                                        tool_params["subject"] = template_result.get("subject")
                                    if not tool_params.get("body"):
                                        tool_params["body"] = template_result.get("body")
                            except Exception as e:
                                logger.error(f"复合链路中邮件模板生成失败: {e}")
                                # 忽略模板失败，继续发送默认内容
                    
                    # 确保传递tool_name参数
                    tool_name = tool_params.get("tool_name")
                    if not tool_name:
                        raise ValueError(f"缺少必要的tool_name参数")
                        
                    result = await tool_instance.execute_tool(tool_name, **{k: v for k, v in tool_params.items() if k != "tool_name"})
                    
                    # 如果是日历工具且创建成功，保存会议信息
                    if category == "calendar" and result.get("success") and result.get("result", {}).get("action") == "create":
                        meeting_info = result.get("result", {}).get("event", {})
                
                results.append({
                    "category": category,
                    "result": result,
                    "tool_params": tool_params
                })
                
                logger.info(f"复合工具执行: {category} 完成")
            
            return {
                "success": all_success,
                "result": {
                    "composite_results": results,
                    "total_tools": len(tool_categories),
                    "executed_tools": len(results)
                },
                "tool_category": tool_category,
                "tool_params": {"composite": True}
            }
            
        except Exception as e:
            logger.error(f"复合工具执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_category": tool_category
            }
    
    async def _extract_tool_parameters(self, query: str, tool_category: str) -> Dict[str, Any]:
        """智能提取工具参数（统一入口），并进行必要的兜底与规范化"""
        try:
            params = await self._intelligent_parameter_extraction(query, tool_category)
        except Exception as e:
            logger.error(f"智能参数提取失败，使用回退方案: {e}")
            params = await self._fallback_parameter_extraction(query, tool_category)
        
        if not isinstance(params, dict):
            params = {}
        
        # 各类别兜底处理
        if tool_category == "calendar":
            params.setdefault("tool_name", "calendar_event")
            params["action"] = params.get("action", "create")
        elif tool_category == "email":
            params.setdefault("tool_name", "email_send")
            to_addr = params.get("to_addresses")
            if isinstance(to_addr, str):
                params["to_addresses"] = [to_addr]
            if params.get("sender_email") in (None, "", "__AUTO__"):
                params["sender_email"] = settings.email_sender
            if params.get("sender_password") in (None, "", "__AUTO__"):
                params["sender_password"] = settings.email_password
        elif tool_category == "file":
            params.setdefault("tool_name", "file_search")
            if params.get("tool_name") == "file_search":
                params.setdefault("directory", "./")
                params.setdefault("filename_pattern", "*")
                params.setdefault("recursive", True)
        
        return params
    
    async def _extract_email_parameters_with_meeting(self, query: str, meeting_info: Dict[str, Any]) -> Dict[str, Any]:
        """根据会议信息生成更贴合场景的邮件主题与正文"""
        import re
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, query)
        to_email = emails[0] if emails else settings.default_recipient
        
        # 支持多会议聚合
        if isinstance(meeting_info, dict) and meeting_info.get("meetings"):
            meetings = meeting_info.get("meetings", [])
            subject = f"会议邀请：共{len(meetings)}个会议安排"
            body_lines = [
                "尊敬的收件人：\n",
                "您好！诚挚邀请您参加以下会议：\n"
            ]
            from datetime import datetime
            for i, m in enumerate(meetings, 1):
                title = m.get("title", f"会议{i}")
                loc = m.get("location", "会议室")
                desc = m.get("description", "")
                st = m.get("start_time", "")
                et = m.get("end_time", "")
                try:
                    if st:
                        st_dt = datetime.fromisoformat(st.replace('T', ' '))
                        st_fmt = st_dt.strftime('%Y年%m月%d日 %H:%M')
                    else:
                        st_fmt = "待定"
                    if et:
                        et_dt = datetime.fromisoformat(et.replace('T', ' '))
                        et_fmt = et_dt.strftime('%H:%M')
                    else:
                        et_fmt = "待定"
                except Exception:
                    st_fmt, et_fmt = st, et
                body_lines.append(f"\n{i}. {title}\n   时间：{st_fmt} - {et_fmt}\n   地点：{loc}\n")
                if desc:
                    body_lines.append(f"   描述：{desc}\n")
            body_lines.append("\n请您合理安排时间。如无法参加，请尽快邮件告知。\n\n此致\n敬礼")
            body = "".join(body_lines)
        else:
            # 单会议
            m = meeting_info or {}
            title = m.get("title", "会议")
            loc = m.get("location", "待定")
            desc = m.get("description", "")
            st = m.get("start_time", "")
            et = m.get("end_time", "")
            from datetime import datetime
            try:
                if st:
                    st_dt = datetime.fromisoformat(st.replace('T', ' '))
                    st_fmt = st_dt.strftime('%Y年%m月%d日 %H:%M')
                else:
                    st_fmt = "待定"
                if et:
                    et_dt = datetime.fromisoformat(et.replace('T', ' '))
                    et_fmt = et_dt.strftime('%H:%M')
                else:
                    et_fmt = "待定"
            except Exception:
                st_fmt, et_fmt = st, et
            subject = f"会议邀请：{title}"
            body = (
                "尊敬的收件人：\n\n"
                "您好！诚挚邀请您参加以下会议：\n\n"
                f"会议主题：{title}\n"
                f"会议时间：{st_fmt} - {et_fmt}\n"
                f"会议地点：{loc}\n"
                + (f"会议描述：{desc}\n" if desc else "") +
                "\n请您准时参加。如有疑问或无法出席，请及时回复此邮件。\n\n此致\n敬礼"
            )
        
        return {
            "tool_name": "email_send",
            "to_addresses": [to_email],
            "subject": subject,
            "body": body,
            "sender_email": settings.email_sender,
            "sender_password": settings.email_password
        }
    
    async def integrate_tool_result_with_context(self, query: str, tool_result: Dict[str, Any], tool_category: str, context: str = "", documents: List[Dict] = None) -> str:
        """结合工具结果与上下文，生成自然语言回复；失败时回退到各自的生成器"""
        try:
            import json
            sys_prompt = (
                "你是一个智能助手。基于用户请求、工具执行结果与上下文，生成清晰、友好、可靠的回复。\n"
                "- 若执行成功：总结已完成的事项并呈现关键结果。\n"
                "- 若执行失败：解释原因并给出可行的下一步建议。\n"
                "- 根据工具类型(calendar/email/file/composite)选择合适的表述方式。"
            )
            user_prompt = (
                f"用户请求：{query}\n\n"
                f"工具类型：{tool_category}\n\n"
                f"工具结果：\n{json.dumps(tool_result, ensure_ascii=False, indent=2)}\n\n"
                f"上下文：{context or '无'}\n\n"
                f"相关文档：\n{json.dumps(documents or [], ensure_ascii=False, indent=2) if documents else '无'}\n\n"
                "请生成一个完整、自然的回复。"
            )
            messages = [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
            resp = await self.llm.ainvoke(messages)
            return resp.content
        except Exception as e:
            logger.error(f"上下文整合失败: {e}")
            # 回退
            if tool_category == "composite":
                return await self.generate_composite_tool_response(query, tool_result)
            if tool_category == "calendar":
                result = tool_result.get("result", {})
                if isinstance(result, list):
                    return await self.generate_calendar_list_response(query, result)
                event = result.get("event") if isinstance(result, dict) else None
                if event:
                    return await self.generate_calendar_response(query, event)
            return await self.generate_tool_response(query, tool_result)
    
    async def _intelligent_parameter_extraction(self, query: str, tool_category: str) -> Dict[str, Any]:
        """使用LLM智能提取参数"""
        if tool_category == "calendar":
            system_prompt = """你是一个智能日程助手。请从用户的自然语言描述中严格提取会议信息，并仅返回一个JSON对象。
            
            提取要点：
            - 操作类型：create（创建）或 list（查询）
            - 标题、开始时间、结束时间、地点、描述、参与者邮箱列表
            
            时间解析规则：
            - “明天”表示明天的日期；“后天”表示后天；“今天”表示当天
            - “下午3点”→ 15:00；“上午10点”→ 10:00
            - 只提到时间未提到日期，默认明天
            - 未给结束时间，默认时长1小时
            
            返回JSON（仅限一个对象，不要包含任何额外文本或Markdown）：
            - 创建：{
                "tool_name": "calendar_event",
                "action": "create",
                "title": "...",
                "start_time": "YYYY-MM-DDTHH:MM:SS",
                "end_time": "YYYY-MM-DDTHH:MM:SS",
                "location": "...",
                "description": "...",
                "attendees": ["a@xx.com", "b@yy.com"]
              }
            - 查询：{
                "tool_name": "calendar_event",
                "action": "list",
                "date_filter": "YYYY-MM-DD"
              }
            """
            
            user_prompt = f"用户输入：{query}\n\n请按上述要求仅返回一个JSON对象。"
            
        elif tool_category == "email":
            system_prompt = """你是一个智能邮件助手。请从用户的自然语言描述中严格提取邮件发送参数，并仅返回一个JSON对象。
            
            你需要在两种模式中做出选择：
            1) 直接发送：当用户已给出明确的主题/正文时
            2) 模板+发送：当用户明确提及“会议邀请/任务提醒/报告摘要”等模板化场景时，返回一个包含 template 的对象，后续将先生成模板再发送
            
            提取要点：
            - 收件人邮箱列表 to_addresses（数组）
            - 邮件主题 subject（字符串，可省略：当使用模板时由模板生成）
            - 邮件正文 body（字符串，可省略：当使用模板时由模板生成）
            - 可选模板：template = { "template_name": "meeting_invite|task_reminder|report_summary", "template_vars": { ... } }
            
            注意：发件人邮箱和密码由系统自动注入，你可以：
            - 省略 sender_email 与 sender_password 字段；或
            - 将 sender_email 与 sender_password 设置为 "__AUTO__"
            
            返回JSON（仅限一个对象，不要包含任何额外文本或Markdown）：
            {
              "tool_name": "email_send",
              "to_addresses": ["a@xx.com"],
              "subject": "...（可省略，若使用模板）",
              "body": "...（可省略，若使用模板）",
              "sender_email": "__AUTO__",
              "sender_password": "__AUTO__",
              "template": {
                 "template_name": "meeting_invite|task_reminder|report_summary",
                 "template_vars": {"k": "v"}
              }
            }
            
            当不需要模板时，请省略 template 字段。
            """
            
            user_prompt = f"用户输入：{query}\n\n请按上述要求仅返回一个JSON对象。"
            
        elif tool_category == "file":
            system_prompt = """你是一个智能文件助手。请从用户的自然语言描述中识别文件操作类型并提取参数，仅返回一个JSON对象。
            
            支持的操作与返回格式：
            - 搜索：{
                "tool_name": "file_search",
                "directory": "搜索目录（必填）",
                "filename_pattern": "文件名通配（可选，默认*）",
                "file_extension": "扩展名（可选）",
                "recursive": true  // 是否递归（可选，默认true）
              }
            - 读取：{
                "tool_name": "file_read",
                "file_path": "文件路径（必填）",
                "encoding": "utf-8" // 可选
              }
            - 写入：{
                "tool_name": "file_write",
                "file_path": "文件路径（必填）",
                "content": "要写入的内容（必填）",
                "encoding": "utf-8", // 可选
                "append": false        // 可选
              }
            
            要求：
            - tool_name 必须是 file_search | file_read | file_write 之一
            - 仅返回一个JSON对象，不要包含任何额外文本或Markdown
            """
            
            user_prompt = f"用户输入：{query}\n\n请按上述要求仅返回一个JSON对象。"
        else:
            return {}
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # 添加超时设置，避免长时间等待
        import asyncio
        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke(messages), 
                timeout=30.0  # 30秒超时
            )
        except asyncio.TimeoutError:
            logger.error("LLM调用超时，使用回退方案")
            raise ValueError("LLM调用超时")
        
        # 解析LLM返回的JSON
        import json
        import re
        
        # 提取JSON部分
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response.content)
        if json_match:
            json_str = json_match.group()
            try:
                params = json.loads(json_str)
                
                # 处理时间格式
                if tool_category == "calendar" and params.get("action") == "create":
                    params = await self._process_calendar_time(params, query)
                
                # 为邮件工具自动注入发件人信息与默认收件人
                if tool_category == "email":
                    # 默认收件人
                    if not params.get("to_addresses") or not isinstance(params.get("to_addresses"), list) or len(params.get("to_addresses")) == 0:
                        params["to_addresses"] = [settings.default_recipient]
                    # 注入发件人信息
                    if params.get("sender_email") in (None, "", "__AUTO__"):
                        params["sender_email"] = settings.email_sender
                    if params.get("sender_password") in (None, "", "__AUTO__"):
                        params["sender_password"] = settings.email_password
                
                return params
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {e}")
                raise
        else:
            raise ValueError("未找到有效的JSON响应")
    
    async def _process_calendar_time(self, params: Dict[str, Any], query: str) -> Dict[str, Any]:
        """处理日历时间参数"""
        from datetime import datetime, timedelta
        import re
        
        # 如果时间格式不正确，尝试智能解析
        start_time = params.get("start_time", "")
        end_time = params.get("end_time", "")
        
        # 解析相对时间
        now = datetime.now()
        
        # 处理"明天"、"后天"等
        if "明天" in query:
            target_date = now + timedelta(days=1)
        elif "后天" in query:
            target_date = now + timedelta(days=2)
        elif "今天" in query:
            target_date = now
        else:
            target_date = now + timedelta(days=1)  # 默认明天
        
        # 处理时间
        time_patterns = [
            (r'(\d{1,2})点', lambda m: int(m.group(1))),
            (r'下午(\d{1,2})点', lambda m: int(m.group(1)) + 12 if int(m.group(1)) < 12 else int(m.group(1))),
            (r'上午(\d{1,2})点', lambda m: int(m.group(1))),
            (r'(\d{1,2}):(\d{2})', lambda m: int(m.group(1)) + (12 if "下午" in query and int(m.group(1)) < 12 else 0))
        ]
        
        hour = 14  # 默认下午2点
        minute = 0
        
        for pattern, extractor in time_patterns:
            match = re.search(pattern, query)
            if match:
                if ":(" in pattern:
                    hour = extractor(match)
                    minute = int(match.group(2))
                else:
                    hour = extractor(match)
                break
        
        # 构建时间字符串
        start_dt = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        end_dt = start_dt + timedelta(hours=1)  # 默认1小时会议
        
        params["start_time"] = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
        params["end_time"] = end_dt.strftime("%Y-%m-%dT%H:%M:%S")
        
        return params
    
    async def _fallback_parameter_extraction(self, query: str, tool_category: str) -> Dict[str, Any]:
        """回退的简单参数提取方法"""
        query_lower = query.lower()
        
        if tool_category == "calendar":
            # 判断是查询还是创建操作
            if any(keyword in query_lower for keyword in ["查看", "查询", "显示", "列出", "我的日程", "日程安排"]):
                from datetime import datetime
                today = datetime.now().strftime("%Y-%m-%d")
                return {
                    "tool_name": "calendar_event",
                    "action": "list",
                    "date_filter": today
                }
            else:
                # 检查是否有多余会议
                import re
                from datetime import datetime, timedelta
                
                # 尝试提取多个会议信息
                meetings = []
                
                # 检查是否明确提到多个会议
                if any(keyword in query for keyword in ["两个会议", "2个会议", "多个会议", "几个会议"]):
                    # 尝试提取具体的会议信息
                    time_patterns = [
                        r'(\d{1,2})点',
                        r'下午(\d{1,2})点',
                        r'上午(\d{1,2})点',
                        r'(\d{1,2}):(\d{2})'
                    ]
                    
                    times_found = []
                    for pattern in time_patterns:
                        matches = re.findall(pattern, query)
                        for match in matches:
                            if isinstance(match, tuple):
                                hour = int(match[0])
                                minute = int(match[1]) if len(match) > 1 else 0
                            else:
                                hour = int(match)
                                minute = 0
                            
                            # 处理下午时间
                            if "下午" in query and hour < 12:
                                hour += 12
                            
                            times_found.append((hour, minute))
                    
                    # 如果找到多个时间，创建多个会议
                    if len(times_found) >= 2:
                        tomorrow = datetime.now() + timedelta(days=1)
                        for i, (hour, minute) in enumerate(times_found[:2]):  # 最多处理2个会议
                            start_dt = tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
                            end_dt = start_dt + timedelta(hours=1)
                            
                            meetings.append({
                                "tool_name": "calendar_event",
                                "action": "create",
                                "title": f"会议{i+1}",
                                "start_time": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                                "end_time": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                                "description": f"第{i+1}个会议",
                                "location": "会议室",
                                "attendees": [settings.default_recipient]
                            })
                    
                    # 如果成功解析出多个会议，返回第一个会议的参数
                    # 多会议创建将在_execute_composite_tools中处理
                    if meetings:
                        first_meeting = meetings[0].copy()  # 创建副本避免循环引用
                        # 创建meetings的深拷贝，避免循环引用
                        meetings_copy = [m.copy() for m in meetings]
                        first_meeting["_multiple_meetings"] = meetings_copy  # 保存所有会议信息
                        return first_meeting
                
                # 默认创建单个会议
                tomorrow = datetime.now() + timedelta(days=1)
                return {
                    "tool_name": "calendar_event",
                    "action": "create",
                    "title": "会议",
                    "start_time": tomorrow.strftime("%Y-%m-%dT14:00:00"),
                    "end_time": tomorrow.strftime("%Y-%m-%dT15:00:00"),
                    "description": "会议安排",
                    "location": "会议室",
                    "attendees": [settings.default_recipient]
                }
        
        elif tool_category == "email":
            import re
            email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            emails = re.findall(email_pattern, query)
            to_email = emails[0] if emails else settings.default_recipient
            
            return {
                "tool_name": "email_send",
            "to_addresses": [to_email],
            "subject": "会议通知",
            "body": "会议通知\n\n尊敬的收件人：\n\n特此通知，您被邀请参加会议。\n\n请准时参加。如有任何疑问，请及时回复此邮件。\n\n谢谢！",
            "sender_email": settings.email_sender,
            "sender_password": settings.email_password
            }
        
        elif tool_category == "file":
            # 根据关键词判断操作类型
            import re
            if any(k in query_lower for k in ["搜索", "查找", "寻找"]):
                # 文件搜索
                return {
                    "tool_name": "file_search",
                    "directory": "./",
                    "filename_pattern": "*",
                    "recursive": True
                }
            if any(k in query_lower for k in ["读取", "查看", "显示", "打开"]):
                # 文件读取，尝试提取路径
                file_patterns = [
                    r'([\w\-\.]+\.[a-zA-Z]{2,4})',        # 简单文件名
                    r'([\w\-\./ ]+\.[a-zA-Z]{2,4})'       # 带路径文件名
                ]
                file_path = "./example.txt"
                for pattern in file_patterns:
                    matches = re.findall(pattern, query)
                    if matches:
                        file_path = matches[0]
                        break
                return {
                    "tool_name": "file_read",
                    "file_path": file_path,
                    "encoding": "utf-8"
                }
            if any(k in query_lower for k in ["写入", "保存", "创建"]):
                # 文件写入，尝试提取路径与内容
                file_patterns = [
                    r'([\w\-\.]+\.[a-zA-Z]{2,4})',
                    r'([\w\-\./ ]+\.[a-zA-Z]{2,4})'
                ]
                file_path = "./output.txt"
                for pattern in file_patterns:
                    matches = re.findall(pattern, query)
                    if matches:
                        file_path = matches[0]
                        break
                # 简单提取“内容：xxx”样式
                content_match = re.search(r'内容[:：]\s*(.+)$', query)
                content = content_match.group(1) if content_match else "这是自动生成的内容。"
                return {
                    "tool_name": "file_write",
                    "file_path": file_path,
                    "content": content,
                    "encoding": "utf-8",
                    "append": False
                }
        
        return {}
    
    async def generate_calendar_response(self, query: str, event_info: Dict[str, Any]) -> str:
        """生成日历事件创建的详细回复"""
        try:
            system_prompt = """你是一个专业的日程助手。用户刚刚创建了一个日历事件，请基于以下结构生成友好、详细的确认回复：
            
            回复要求：
            1. 结论（一句话确认已安排）
            - 会议详情（标题、时间、地点、参与者等）
            - 贴心提醒（如提前准备、到场时间等，选填）
            请避免出现“工具执行”等技术性表述。
            """
            
            user_prompt = f"""用户请求：{query}
            
            创建的会议信息：
            - 会议ID：{event_info.get('id', '')}
            - 标题：{event_info.get('title', '')}
            - 开始时间：{event_info.get('start_time', '')}
            - 结束时间：{event_info.get('end_time', '')}
            - 地点：{event_info.get('location', '')}
            - 描述：{event_info.get('description', '')}
            - 参与者：{', '.join(event_info.get('attendees', []))}
            - 创建时间：{event_info.get('created_at', '')}
            
            请生成一个详细的确认回复。"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"生成日历回复失败: {e}")
            return f"会议已成功创建！\n\n会议详情：\n标题：{event_info.get('title', '')}\n时间：{event_info.get('start_time', '')} - {event_info.get('end_time', '')}\n地点：{event_info.get('location', '')}\n参与者：{', '.join(event_info.get('attendees', []))}"
    
    async def generate_tool_response(self, query: str, tool_result: Dict[str, Any]) -> str:
        """生成其他工具执行的详细回复"""
        try:
            system_prompt = """你是一个智能助手。用户刚刚执行了一个操作，请生成一个友好、详细的确认回复。
            
            回复要求：
            1. 确认操作已成功完成
            2. 展示操作的详细结果
            3. 语气友好、专业
            4. 不要提及"工具执行"等技术术语
            """
            
            user_prompt = f"""用户请求：{query}
            
            操作结果：{tool_result.get('result', '')}
            操作类型：{tool_result.get('tool_category', '')}
            
            请生成一个详细的确认回复。"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
             logger.error(f"生成工具回复失败: {e}")
             return f"操作已成功完成！\n\n结果：{tool_result.get('result', '')}"
    
    async def generate_calendar_list_response(self, query: str, events: List[Dict[str, Any]]) -> str:
        """生成日历查询结果的详细回复"""
        try:
            system_prompt = """你是一个专业的日程助手。用户刚刚查询了日程安排，请生成一个友好、详细的回复。
            
            回复要求：
            1. 如果有日程，清晰地展示所有日程信息
            2. 如果没有日程，友好地告知用户
            3. 按时间顺序排列日程
            4. 语气友好、专业
            5. 不要提及"工具执行"等技术术语
            6. 可以提供一些贴心的建议
            """
            
            events_text = ""
            if events:
                events_text = "\n\n查询到的日程安排：\n"
                for i, event in enumerate(events, 1):
                    events_text += f"\n{i}. {event.get('title', '未命名会议')}\n"
                    events_text += f"   时间：{event.get('start_time', '')} - {event.get('end_time', '')}\n"
                    events_text += f"   地点：{event.get('location', '未指定')}\n"
                    if event.get('description'):
                        events_text += f"   描述：{event.get('description')}\n"
                    if event.get('attendees'):
                        events_text += f"   参与者：{', '.join(event.get('attendees', []))}\n"
            else:
                events_text = "\n\n暂无日程安排。"
            
            user_prompt = f"""用户请求：{query}
            
            查询结果：{events_text}
            
            请生成一个详细的回复。"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"生成日历查询回复失败: {e}")
            if events:
                result = f"为您查询到 {len(events)} 个日程安排：\n\n"
                for i, event in enumerate(events, 1):
                    result += f"{i}. {event.get('title', '未命名会议')}\n"
                    result += f"   时间：{event.get('start_time', '')} - {event.get('end_time', '')}\n"
                    result += f"   地点：{event.get('location', '未指定')}\n\n"
                return result
            else:
                return "您当前没有日程安排。"
    
    async def generate_composite_tool_response(self, query: str, tool_result: Dict[str, Any]) -> str:
        """生成复合工具执行的详细回复"""
        try:
            system_prompt = """你是一个智能助手。用户刚刚执行了一个包含多个操作的复合任务，请生成一个友好、详细的确认回复。
            
            回复要求：
            1. 确认所有操作都已成功完成
            2. 按顺序展示每个操作的详细结果
            3. 语气友好、专业
            4. 不要提及"工具执行"等技术术语
            5. 如果是日历+邮件的组合，要说明会议已创建并通知已发送
            """
            
            # 构建操作结果描述
            results_text = ""
            composite_results = tool_result.get("result", {}).get("composite_results", [])
            
            for i, result in enumerate(composite_results, 1):
                category = result.get("category", "")
                result_data = result.get("result", {})
                
                if category == "calendar":
                    if result_data.get("success"):
                        event_data = result_data.get("result", {})
                        if event_data.get("action") == "create":
                            event_info = event_data.get("event", {})
                            results_text += f"\n{i}. 会议创建成功：\n"
                            results_text += f"   标题：{event_info.get('title', '')}\n"
                            results_text += f"   时间：{event_info.get('start_time', '')} - {event_info.get('end_time', '')}\n"
                            results_text += f"   地点：{event_info.get('location', '')}\n"
                elif category == "email":
                    if result_data.get("success"):
                        email_data = result_data.get("result", {})
                        results_text += f"\n{i}. 邮件发送成功：\n"
                        results_text += f"   收件人：{email_data.get('to_addresses', [])}\n"
                        results_text += f"   主题：{email_data.get('subject', '')}\n"
            
            user_prompt = f"""用户请求：{query}
            
            执行结果：{results_text}
            总共执行了 {len(composite_results)} 个操作
            
            请生成一个详细的确认回复。"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"生成复合工具回复失败: {e}")
            # 生成简单的回复
            composite_results = tool_result.get("result", {}).get("composite_results", [])
            simple_response = "操作已成功完成！\n\n"
            
            for i, result in enumerate(composite_results, 1):
                category = result.get("category", "")
                if category == "calendar":
                    simple_response += f"{i}. 会议已成功创建\n"
                elif category == "email":
                    simple_response += f"{i}. 邮件通知已发送\n"
            
            return simple_response
