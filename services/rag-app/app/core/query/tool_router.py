import json
from typing import Any, Dict, Optional


SEARCH_DATABASE_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_database",
        "description": "Run hybrid legal/regulation database retrieval before asking the user for clarification.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's question or a concise retrieval query preserving legal subjects and agencies.",
                },
                "reason": {
                    "type": "string",
                    "description": "Brief reason for probing the database.",
                },
            },
            "required": ["query"],
        },
    },
}


async def route_with_search_database_tool(runtime: Any, query: str) -> Dict[str, Any]:
    if not bool(getattr(runtime.config, "ENABLE_LLM_TOOL_ROUTER", True)):
        return {"tool_called": False, "reason": "disabled"}
    client = getattr(runtime, "llm_client", None)
    if not client or not client.available():
        return {"tool_called": False, "reason": "llm_unavailable"}

    system_prompt = (
        "你是法规 RAG 的路由模型，只负责决定是否调用工具。不要输出意图分类 JSON。\n"
        "你只有一个工具 search_database，它代表后端混合检索（向量、词法、rerank、证据选择）。\n"
        "绝对规则：你没有最终回答权、没有拒绝权、没有澄清权。\n"
        "你绝对禁止自行拒绝用户请求，绝对禁止输出类似“无法回答”“不能处理”“不属于范围”“请补充信息”的结论。\n"
        "拒绝、空结果判断、澄清请求只能由 search_database 执行后的后端流程决定。\n"
        "规则：\n"
        "1. 只要问题可能需要法规库证据，或你对是否可回答不确定，先调用 search_database。\n"
        "2. 不要在调用 search_database 前向用户澄清。\n"
        "3. 行政机关、执法机关、监管部门名称必须保留在检索语义中。\n"
        "4. 只有当 search_database 执行后没有结果，系统才会向用户发起澄清。\n"
        "5. 如果问题看起来范围不明、主体不明、可能无结果、可能被拒绝，也必须先调用 search_database 试探。\n"
        "6. 只有在问题明显与法规库检索完全无关时，才可以不调用工具；即便如此也不得拒绝用户，只能保持路由中立。"
    )
    system_prompt = (
        "你是法规 RAG 的路由模型，只负责决定是否调用工具。不要输出意图分类 JSON。\n"
        "你只有一个工具 search_database，它代表后端混合检索（向量、词法、rerank、证据选择）。\n"
        "绝对规则：你没有最终回答权、没有拒绝权、没有澄清权。\n"
        "你绝对禁止自行拒绝用户请求，绝对禁止输出类似“无法回答”“不能处理”“不属于范围”“请补充信息”的结论。\n"
        "拒绝、空结果判断、澄清请求只能由 search_database 执行后的后端流程决定。\n"
        "规则：\n"
        "1. 只要问题可能需要法规库证据，或你对是否可回答不确定，先调用 search_database。\n"
        "2. 不要在调用 search_database 前向用户澄清。\n"
        "3. 行政机关、执法机关、监管部门名称必须保留在检索语义中。\n"
        "4. 只有当 search_database 执行后没有结果，系统才会向用户发起澄清。\n"
        "5. 如果问题看起来范围不明、主体不明、可能无结果、可能被拒绝，也必须先调用 search_database 试探。\n"
        "6. 兜底行为：只有当用户的输入是纯粹的日常问候（如“你好”、“在吗”）或完全脱离法规领域的闲聊时，"
        "你才可以不调用工具，此时请用一句简短、礼貌的话回复（例如：“您好，请问有什么法规查询需求？”）。"
        "除此之外的任何问题，一律调用 search_database 工具。"
    )
    tool_choice: Any = "auto" if _is_pure_router_chitchat(query) else {
        "type": "function",
        "function": {"name": "search_database"},
    }
    payload = client.build_payload(
        system_prompt,
        f"用户问题：{query}",
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(getattr(runtime.config, "LLM_TOOL_ROUTER_MAX_TOKENS", 120)),
        presence_penalty=0.0,
        extra={
            "tools": [SEARCH_DATABASE_TOOL],
            "tool_choice": tool_choice,
        },
    )
    timeout = max(1, int(getattr(runtime.config, "LLM_TOOL_ROUTER_TIMEOUT", 8)))
    try:
        data = await client.chat_response(payload, timeout=timeout)
    except Exception as exc:
        return {"tool_called": False, "reason": "router_error", "error": type(exc).__name__}

    tool_call = _first_tool_call(data, "search_database")
    if not tool_call:
        router_text = _first_message_text(data)[:240]
        if not _is_pure_router_chitchat(query):
            return {
                "tool_called": True,
                "tool_name": "search_database",
                "arguments": {
                    "query": query,
                    "reason": "router_no_tool_call_fallback",
                },
                "query": query,
                "reason": "router_no_tool_call_fallback",
                "router_text": router_text,
            }
        return {
            "tool_called": False,
            "reason": "no_tool_call",
            "router_text": router_text,
        }
    arguments = _tool_arguments(tool_call)
    tool_query = str(arguments.get("query") or query).strip() or query
    if not str(arguments.get("query") or "").strip():
        arguments = {**arguments, "query": tool_query}
    return {
        "tool_called": True,
        "tool_name": "search_database",
        "arguments": arguments,
        "query": tool_query,
        "reason": str(arguments.get("reason") or "llm_requested_database_probe").strip(),
    }


def _first_tool_call(data: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    choices = data.get("choices") or []
    if not isinstance(choices, list) or not choices:
        return None
    message = (choices[0] or {}).get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        return None
    for call in tool_calls:
        function = (call or {}).get("function") or {}
        if str(function.get("name") or "") == name:
            return call
    return None


def _is_pure_router_chitchat(query: str) -> bool:
    text = str(query or "").strip()
    if not text:
        return True
    compact = "".join(text.split()).lower()
    greetings = {
        "你好",
        "您好",
        "在吗",
        "在不在",
        "hello",
        "hi",
        "hey",
    }
    if compact in greetings:
        return True
    legal_markers = [
        "法",
        "法规",
        "条例",
        "规定",
        "办法",
        "规章",
        "处罚",
        "职责",
        "义务",
        "许可",
        "备案",
        "公安",
        "执法",
        "监管",
        "管理",
        "第",
        "条",
        "款",
        "项",
    ]
    if any(marker in text for marker in legal_markers):
        return False
    chitchat_markers = [
        "今天天气",
        "讲个笑话",
        "你是谁",
        "陪我聊",
        "随便聊",
    ]
    return any(marker in text for marker in chitchat_markers)


def _tool_arguments(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    function = (tool_call or {}).get("function") or {}
    raw = function.get("arguments") or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw or "{}"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _first_message_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not isinstance(choices, list) or not choices:
        return ""
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content") if isinstance(message, dict) else ""
    return str(content or "").strip()
