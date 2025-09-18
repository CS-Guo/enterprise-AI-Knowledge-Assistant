#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端 /chat 接口测试（离线、无外部依赖）

覆盖策略：
- HR 类别且无文档命中 -> Planner 禁止开放域回退
- general 类别且无文档命中 -> 允许开放域回退

通过 monkeypatch 替换：
- KnowledgeAgent.analyze_query_intent（避免外部LLM）
- KnowledgeAgent.generate_response（避免外部LLM）
- DocumentRetriever.retrieve_with_rerank（避免向量检索触发外部嵌入）
"""
import os
import sys
from typing import Dict, Any, List

# 确保项目根目录加入模块搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.agents.knowledge_agent import KnowledgeAgent
from src.rag.retriever import DocumentRetriever


@pytest.fixture
def client():
    # 使用 FastAPI TestClient 同步调用接口
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize(
    "intent_mock, query, expect_open, expect_category",
    [
        (
            # HR 类别：不允许开放域回退
            {
                "intent_type": "question",
                "confidence": 0.95,
                "entities": [],
                "requires_tools": False,
                "tool_category": "none",
                "category": "hr",
                "action_needed": "",
            },
            "我想咨询年假政策，多少天，如何申请？",
            False,
            "hr",
        ),
        (
            # general 类别：允许开放域回退
            {
                "intent_type": "question",
                "confidence": 0.9,
                "entities": [],
                "requires_tools": False,
                "tool_category": "none",
                "category": "general",
                "action_needed": "",
            },
            "什么是向量数据库，简单解释一下原理及应用场景",
            True,
            "general",
        ),
    ],
)
def test_chat_open_domain_policy_offline(monkeypatch, client: TestClient, intent_mock: Dict[str, Any], query: str, expect_open: bool, expect_category: str):
    # 1) 避免外部嵌入：检索返回空，强制触发回退判断路径
    async def fake_retrieve_with_rerank(self, query: str, filter_category: str = None) -> List[str]:
        return []

    monkeypatch.setattr(DocumentRetriever, "retrieve_with_rerank", fake_retrieve_with_rerank)

    # 2) 避免外部LLM：意图分析固定输出
    async def fake_analyze(self, q: str) -> Dict[str, Any]:
        intent = dict(intent_mock)
        intent["original_query"] = q
        return intent

    monkeypatch.setattr(KnowledgeAgent, "analyze_query_intent", fake_analyze)

    # 3) 避免外部LLM：响应生成直接返回固定文案
    async def fake_generate(self, query: str, context: str, conversation_history=None, allow_open_domain: bool = False, disclaimers: str = "") -> str:
        # 模拟根据 allow_open_domain 追加提示，便于断言
        prefix = "[开放域] " if allow_open_domain else "[受限域] "
        return prefix + "测试回答：" + (query[:20] if isinstance(query, str) else "")

    monkeypatch.setattr(KnowledgeAgent, "generate_response", fake_generate)

    # 4) 调用 /chat 接口
    resp = client.post(
        "/api/v1/chat",
        json={
            "query": query,
            "conversation_history": [],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # 5) 断言：策略与回退
    assert isinstance(data, dict)
    assert data.get("intent_analysis", {}).get("category") == expect_category
    assert data.get("open_domain_fallback") is expect_open

    # HR 不允许开放域 -> 无免责声明；general 允许 -> 有免责声明
    if expect_open:
        assert isinstance(data.get("disclaimers"), str) and len(data.get("disclaimers")) > 0
        # 响应文本由 fake_generate 决定，应包含“开放域”提示
        assert data.get("response", "").startswith("[开放域]")
    else:
        assert data.get("disclaimers", "") in ("", None)
        assert data.get("response", "").startswith("[受限域]")

    # 文档为0命中
    assert data.get("documents_used") == []


if __name__ == "__main__":
    # 允许独立运行
    pytest.main([__file__])