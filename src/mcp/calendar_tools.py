# src/mcp/calendar_tools.py
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_tool import BaseMCPTool
import json

class CalendarEventTool(BaseMCPTool):
    """日历事件工具"""
    
    def __init__(self):
        super().__init__(
            name="calendar_event", 
            description="创建、查询或修改日历事件"
        )
        # 简化实现，使用内存存储
        self.events = []
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "update", "delete"],
                    "description": "操作类型"
                },
                "event_id": {
                    "type": "string",
                    "description": "事件ID（更新或删除时需要）"
                },
                "title": {
                    "type": "string",
                    "description": "事件标题"
                },
                "start_time": {
                    "type": "string",
                    "description": "开始时间 (ISO格式)"
                },
                "end_time": {
                    "type": "string", 
                    "description": "结束时间 (ISO格式)"
                },
                "description": {
                    "type": "string",
                    "description": "事件描述"
                },
                "location": {
                    "type": "string",
                    "description": "事件地点"
                },
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "参会人员邮箱列表"
                },
                "date_filter": {
                    "type": "string",
                    "description": "日期过滤器 (查询时使用，格式: YYYY-MM-DD)"
                }
            },
            "required": ["action"]
        }
    
    async def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """执行日历操作"""
        if action == "create":
            required = ["title", "start_time", "end_time"]
            for r in required:
                if r not in kwargs:
                    raise ValueError(f"Missing required parameter for create: {r}")
            return await self._create_event(**kwargs)
        elif action == "list":
            return await self._list_events(**kwargs)
        elif action == "update":
            if "event_id" not in kwargs:
                raise ValueError(f"Missing required parameter for update: event_id")
            return await self._update_event(**kwargs)
        elif action == "delete":
            if "event_id" not in kwargs:
                raise ValueError(f"Missing required parameter for delete: event_id")
            return await self._delete_event(**kwargs)
        else:
            raise ValueError(f"不支持的操作: {action}")
    
    async def _create_event(self, title: str, start_time: str, end_time: str,
                           description: Optional[str] = None, location: Optional[str] = None,
                           attendees: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """创建日历事件"""
        event_id = f"event_{len(self.events) + 1}_{datetime.now().timestamp()}"
        
        event = {
            "id": event_id,
            "title": title,
            "start_time": start_time,
            "end_time": end_time,
            "description": description or "",
            "location": location or "",
            "attendees": attendees or [],
            "created_at": datetime.now().isoformat()
        }
        
        self.events.append(event)
        
        return {
            "action": "create",
            "event_id": event_id,
            "event": event,
            "status": "created"
        }
    
    async def _list_events(self, date_filter: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """列出日历事件"""
        events = self.events.copy()
        
        if date_filter:
            # 简单的日期过滤
            events = [
                event for event in events
                if event["start_time"].startswith(date_filter)
            ]
        
        return {
            "action": "list",
            "events": events,
            "total_count": len(events),
            "date_filter": date_filter
        }
    
    async def _update_event(self, event_id: str, **kwargs) -> Dict[str, Any]:
        """更新日历事件"""
        event = next((e for e in self.events if e["id"] == event_id), None)
        
        if not event:
            raise ValueError(f"事件不存在: {event_id}")
        
        # 更新事件字段
        updatable_fields = ["title", "start_time", "end_time", "description", "location", "attendees"]
        updated_fields = []
        
        for field in updatable_fields:
            if field in kwargs:
                event[field] = kwargs[field]
                updated_fields.append(field)
        
        event["updated_at"] = datetime.now().isoformat()
        
        return {
            "action": "update",
            "event_id": event_id,
            "updated_fields": updated_fields,
            "event": event,
            "status": "updated"
        }
    
    async def _delete_event(self, event_id: str, **kwargs) -> Dict[str, Any]:
        """删除日历事件"""
        event = next((e for e in self.events if e["id"] == event_id), None)
        
        if not event:
            raise ValueError(f"事件不存在: {event_id}")
        
        self.events.remove(event)
        
        return {
            "action": "delete",
            "event_id": event_id,
            "status": "deleted"
        }

class SchedulingTool(BaseMCPTool):
    """智能排程工具"""
    
    def __init__(self):
        super().__init__(
            name="smart_scheduling",
            description="智能安排会议时间"
        )
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "参会人员邮箱列表"
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "会议时长（分钟）",
                    "default": 60
                },
                "preferred_date": {
                    "type": "string",
                    "description": "首选日期 (YYYY-MM-DD)"
                },
                "time_range": {
                    "type": "object",
                    "properties": {
                        "start_hour": {"type": "integer", "minimum": 0, "maximum": 23},
                        "end_hour": {"type": "integer", "minimum": 0, "maximum": 23}
                    },
                    "description": "时间范围"
                }
            },
            "required": ["attendees", "preferred_date"]
        }
    
    async def execute(self, attendees: List[str], preferred_date: str, 
                     duration_minutes: int = 60, time_range: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """执行智能排程"""
        # 简化实现：生成可用时间段
        if time_range is None:
            time_range = {"start_hour": 9, "end_hour": 17}  # 默认工作时间
        
        # 生成可能的时间段
        available_slots = []
        start_hour = time_range["start_hour"]
        end_hour = time_range["end_hour"]
        
        # 每30分钟一个时间段
        current_hour = start_hour
        while current_hour < end_hour:
            for minutes in [0, 30]:
                if current_hour * 60 + minutes + duration_minutes <= end_hour * 60:
                    start_time = f"{preferred_date}T{current_hour:02d}:{minutes:02d}:00"
                    end_dt = datetime.fromisoformat(start_time) + timedelta(minutes=duration_minutes)
                    end_time = end_dt.isoformat()
                    
                    available_slots.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration_minutes": duration_minutes
                    })
            
            current_hour += 1
        
        # 推荐最佳时间段（简化：选择上午的第一个可用时段）
        recommended_slot = next(
            (slot for slot in available_slots if slot["start_time"].split("T")[1] < "12:00:00"),
            available_slots[0] if available_slots else None
        )
        
        return {
            "attendees": attendees,
            "preferred_date": preferred_date,
            "available_slots": available_slots[:5],  # 返回前5个可用时段
            "recommended_slot": recommended_slot,
            "total_available": len(available_slots)
        }

class CalendarTools:
    """日历工具集合"""
    
    def __init__(self):
        self.tools = {
            "calendar_event": CalendarEventTool(),
            "smart_scheduling": SchedulingTool()
        }
    
    def get_all_tools(self) -> Dict[str, BaseMCPTool]:
        """获取所有日历工具"""
        return self.tools
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """执行指定的日历工具"""
        if tool_name not in self.tools:
            raise ValueError(f"未知的日历工具: {tool_name}")
        
        return await self.tools[tool_name].safe_execute(**kwargs)