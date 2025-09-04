# -*- coding: utf-8 -*-
"""
邮件工具测试脚本（安全版）

- 验证邮件模板工具生成主题与正文
- 验证邮件发送工具的参数模式（不进行真实外发）
- 使用同步测试包装异步逻辑，避免依赖 pytest-asyncio
"""
import os
import sys
import asyncio

# 确保项目根目录加入模块搜索路径，避免需要 PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mcp.email_tools import EmailSendTool, EmailTemplateTool


async def _async_email_tools_checks():
    # 测试邮件模板工具
    template_tool = EmailTemplateTool()
    template_result = await template_tool.execute(
        template_name="meeting_invite",
        template_vars={
            "meeting_title": "项目启动会议",
            "meeting_time": "2023-11-15 14:00",
            "meeting_location": "会议室A",
            "agenda": "1. 项目介绍\n2. 任务分配\n3. 时间安排",
        },
    )

    assert isinstance(template_result, dict)
    assert "subject" in template_result and template_result["subject"]
    assert "body" in template_result and template_result["body"]

    # 测试邮件发送工具（仅校验参数模式，不实际发送）
    send_tool = EmailSendTool(smtp_server="smtp.qq.com", smtp_port=587)
    schema = send_tool.get_parameters_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema and isinstance(schema["properties"], dict)
    # 关键字段存在
    for key in ["to_addresses", "subject", "body", "sender_email", "sender_password"]:
        assert key in schema["properties"], f"参数模式缺少字段: {key}"


def test_email_tools():
    """同步包装的测试入口，避免额外插件依赖。"""
    asyncio.run(_async_email_tools_checks())