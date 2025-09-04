# src/mcp/email_tools.py
import smtplib
import ssl
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
import logging
from .base_tool import BaseMCPTool

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class EmailSendTool(BaseMCPTool):
    """邮件发送工具"""
    
    def __init__(self, smtp_server: str = "smtp.qq.com", smtp_port: int = 587):
        super().__init__(
            name="email_send",
            description="发送电子邮件"
        )
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to_addresses": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "收件人邮箱列表"
                },
                "subject": {
                    "type": "string",
                    "description": "邮件主题"
                },
                "body": {
                    "type": "string",
                    "description": "邮件正文"
                },
                "cc_addresses": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "抄送邮箱列表"
                },
                "sender_email": {
                    "type": "string",
                    "description": "发件人邮箱"
                },
                "sender_password": {
                    "type": "string",
                    "description": "发件人邮箱密码或应用密码"
                }
            },
            "required": ["to_addresses", "subject", "body", "sender_email", "sender_password"]
        }
    
    async def execute(self, to_addresses: List[str], subject: str, body: str,
                     sender_email: str, sender_password: str, 
                     cc_addresses: Optional[List[str]] = None) -> Dict[str, Any]:
        """执行邮件发送"""
        # 创建邮件对象
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = ', '.join(to_addresses)
        msg['Subject'] = subject
        
        if cc_addresses:
            msg['Cc'] = ', '.join(cc_addresses)
        
        # 添加邮件正文
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # 发送邮件
        try:
            # 使用 SMTP 和 STARTTLS，端口587
            context = ssl.create_default_context()
            server = smtplib.SMTP(self.smtp_server, 587, timeout=30)
            server.starttls(context=context)

            try:
                logging.debug("尝试登录SMTP服务器...")
                server.login(sender_email, sender_password)
                logging.debug("SMTP服务器登录成功。")
            except smtplib.SMTPAuthenticationError as e:
                logging.error(f"SMTP认证失败: {e}")
                try:
                    server.quit()
                except:
                    pass
                return False
            except Exception as e:
                logging.error(f"邮件发送失败: 登录时发生未知错误: {type(e).__name__}: {e}")
                try:
                    server.quit()
                except:
                    pass
                return False
            
            all_recipients = to_addresses + (cc_addresses or [])
            try:
                server.sendmail(sender_email, all_recipients, msg.as_string())
                logging.debug("邮件发送成功。")
                result = {
                    "status": "sent",
                    "to_addresses": to_addresses,
                    "cc_addresses": cc_addresses or [],
                    "subject": subject,
                    "message_id": msg.get('Message-ID')
                }
                # 手动关闭连接，忽略关闭时的错误
                try:
                    server.quit()
                except:
                    pass
                return result
            except Exception as send_error:
                logging.error(f"发送邮件时出错: {type(send_error).__name__}: {send_error}")
                try:
                    server.quit()
                except:
                    pass
                return False
            
        except smtplib.SMTPAuthenticationError as e:
            logging.error(f"SMTP认证失败: {e}")
            return False
        except smtplib.SMTPConnectError as e:
            logging.error(f"SMTP连接错误: {e}")
            return False
        except smtplib.SMTPException as e:
            logging.error(f"SMTP错误: {e}")
            return False
        except Exception as e:
            logging.error(f"邮件发送失败: {type(e).__name__}: {e}")
            return False

class EmailTemplateTool(BaseMCPTool):
    """邮件模板工具"""
    
    def __init__(self):
        super().__init__(
            name="email_template",
            description="根据模板生成邮件内容"
        )
        self.templates = {
            "meeting_invite": {
                "subject": "会议邀请：{meeting_title}",
                "body": """您好，

邀请您参加以下会议：

会议主题：{meeting_title}
会议时间：{meeting_time}
会议地点：{meeting_location}
会议议程：{agenda}

请确认您的参会情况。

谢谢！
"""
            },
            "task_reminder": {
                "subject": "任务提醒：{task_title}",
                "body": """您好，

提醒您有一个任务需要处理：

任务名称：{task_title}
截止时间：{due_date}
任务描述：{description}
优先级：{priority}

请及时处理。

谢谢！
"""
            },
            "report_summary": {
                "subject": "报告摘要：{report_title}",
                "body": """您好，

以下是 {report_title} 的摘要：

{summary_content}

详细报告请查看附件或联系我获取。

谢谢！
"""
            }
        }
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "template_name": {
                    "type": "string",
                    "enum": list(self.templates.keys()),
                    "description": "邮件模板名称"
                },
                "template_vars": {
                    "type": "object",
                    "description": "模板变量字典"
                }
            },
            "required": ["template_name", "template_vars"]
        }
    
    async def execute(self, template_name: str, template_vars: Dict[str, Any]) -> Dict[str, Any]:
        """执行邮件模板生成"""
        if template_name not in self.templates:
            raise ValueError(f"未知的邮件模板: {template_name}")
        
        template = self.templates[template_name]
        
        try:
            subject = template["subject"].format(**template_vars)
            body = template["body"].format(**template_vars)
            
            return {
                "template_name": template_name,
                "subject": subject,
                "body": body,
                "variables_used": list(template_vars.keys())
            }
        except KeyError as e:
            raise ValueError(f"模板变量缺失: {e}")

class EmailTools:
    """邮件工具集合"""
    
    def __init__(self):
        self.tools = {
            "email_send": EmailSendTool(),
            "email_template": EmailTemplateTool()
        }
    
    def get_all_tools(self) -> Dict[str, BaseMCPTool]:
        """获取所有邮件工具"""
        return self.tools
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """执行指定的邮件工具"""
        if tool_name not in self.tools:
            raise ValueError(f"未知的邮件工具: {tool_name}")
        
        return await self.tools[tool_name].safe_execute(**kwargs)