#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面测试所有MCP工具的功能（安全版）

- 仅验证模板/参数/本地文件操作，不进行真实外发或网络请求
- 使用同步包装异步测试，避免依赖 pytest-asyncio
"""
import os
import sys
import asyncio
import tempfile

# 确保项目根目录加入模块搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mcp.email_tools import EmailSendTool, EmailTemplateTool
from src.mcp.calendar_tools import CalendarEventTool, SchedulingTool
from src.mcp.file_tools import FileSearchTool, FileReadTool, FileWriteTool


class ToolTester:
    def __init__(self):
        self.test_results = []
        self.temp_dir = None

    def log_test(self, test_name: str, success: bool, message: str = ""):
        status = "✅ 通过" if success else "❌ 失败"
        result = f"{status} {test_name}"
        if message:
            result += f" - {message}"
        print(result)
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
        })

    async def test_email_tools(self):
        # 模板
        try:
            template_tool = EmailTemplateTool()
            template_result = await template_tool.execute(
                template_name="meeting_invite",
                template_vars={
                    "meeting_title": "测试会议",
                    "meeting_time": "2024-01-15 14:00",
                    "meeting_location": "测试会议室",
                    "agenda": "1. 测试议程\n2. 功能验证",
                },
            )
            ok = bool(template_result and template_result.get("subject") and template_result.get("body"))
            self.log_test("邮件模板生成", ok, f"主题: {template_result.get('subject', '')}")
        except Exception as e:
            self.log_test("邮件模板生成", False, f"异常: {e}")

        # 发送工具参数模式校验
        try:
            send_tool = EmailSendTool(smtp_server="smtp.qq.com", smtp_port=587)
            schema = send_tool.get_parameters_schema()
            ok = bool(schema and isinstance(schema, dict) and "properties" in schema)
            if ok:
                needed = ["to_addresses", "subject", "body", "sender_email", "sender_password"]
                ok = all(k in schema["properties"] for k in needed)
            self.log_test("邮件发送工具初始化", ok, "参数模式检查")
        except Exception as e:
            self.log_test("邮件发送工具初始化", False, f"异常: {e}")

    async def test_calendar_tools(self):
        calendar_tool = CalendarEventTool()
        try:
            create_result = await calendar_tool.execute(
                action="create",
                title="测试会议",
                start_time="2024-01-15T14:00:00",
                end_time="2024-01-15T15:00:00",
                description="这是一个测试事件",
                location="测试会议室",
            )
            ok = bool(create_result and create_result.get("status") == "created")
            self.log_test("日历事件创建", ok)
            if ok:
                event_id = create_result.get("event_id")
                list_result = await calendar_tool.execute(action="list")
                self.log_test("日历事件查询", bool(list_result and "events" in list_result))
                if event_id:
                    update_result = await calendar_tool.execute(action="update", event_id=event_id, title="更新后的测试会议")
                    self.log_test("日历事件更新", bool(update_result and update_result.get("status") == "updated"))
                    delete_result = await calendar_tool.execute(action="delete", event_id=event_id)
                    self.log_test("日历事件删除", bool(delete_result and delete_result.get("status") == "deleted"))
        except Exception as e:
            self.log_test("日历事件操作", False, f"异常: {e}")

        try:
            scheduling_tool = SchedulingTool()
            schedule_result = await scheduling_tool.execute(
                attendees=["user1@example.com", "user2@example.com"],
                preferred_date="2024-01-15",
                duration_minutes=60,
            )
            ok = bool(schedule_result and "available_slots" in schedule_result)
            self.log_test("会议调度", ok)
        except Exception as e:
            self.log_test("会议调度", False, f"异常: {e}")

    async def test_file_tools(self):
        self.temp_dir = tempfile.mkdtemp()
        test_file_path = os.path.join(self.temp_dir, "test_file.txt")
        test_content = "这是一个测试文件\n包含多行内容\n用于验证文件工具功能"
        try:
            write_tool = FileWriteTool()
            write_result = await write_tool.execute(file_path=test_file_path, content=test_content)
            self.log_test("文件写入", bool(write_result and "file_path" in write_result))
        except Exception as e:
            self.log_test("文件写入", False, f"异常: {e}")

        try:
            read_tool = FileReadTool()
            read_result = await read_tool.execute(file_path=test_file_path)
            self.log_test("文件读取", bool(read_result and read_result.get("content") == test_content))
        except Exception as e:
            self.log_test("文件读取", False, f"异常: {e}")

        try:
            search_tool = FileSearchTool()
            search_result = await search_tool.execute(directory=self.temp_dir, filename_pattern="*.txt")
            ok = bool(search_result and "files_found" in search_result and len(search_result["files_found"]) > 0)
            self.log_test("文件搜索", ok)
        except Exception as e:
            self.log_test("文件搜索", False, f"异常: {e}")

    def cleanup(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def print_summary(self):
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["success"])
        failed = total - passed
        print("\n== 测试总结 ==")
        print(f"总数: {total} 通过: {passed} 失败: {failed}")
        if failed:
            print("失败详情:")
            for r in self.test_results:
                if not r["success"]:
                    print(f"- {r['test']}: {r['message']}")


async def _async_run_all():
    tester = ToolTester()
    try:
        await tester.test_email_tools()
        await tester.test_calendar_tools()
        await tester.test_file_tools()
    finally:
        tester.cleanup()
        tester.print_summary()


def test_all_tools():
    """同步包装，pytest 直接运行。"""
    asyncio.run(_async_run_all())


if __name__ == "__main__":
    asyncio.run(_async_run_all())