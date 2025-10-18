# src/graph.py
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .assistants import sales_assistant, sales_tools, support_assistant, support_tools
from .state import State
from .tools import create_tool_node_with_fallback


def after_sales_tool(state: dict) -> dict:
    tool_msg = state["messages"][-1]
    if (
        isinstance(tool_msg, ToolMessage)
        and getattr(tool_msg, "name", None) == "RouteToCustomerSupport"
    ):
        return {"dialog_state": "customer_support"}
    return {}


def after_support_tool(state: dict) -> dict:
    tool_msg = state["messages"][-1]
    if (
        isinstance(tool_msg, ToolMessage)
        and getattr(tool_msg, "name", None) == "EscalateToHuman"
    ):
        # Parse severity and summary from the content
        content = tool_msg.content
        severity = "unknown"
        summary = "No summary provided"
        try:
            parts = content.split(" ")
            for part in parts:
                if part.startswith("severity="):
                    severity = part.split("=")[1].strip("'")
                elif part.startswith("summary="):
                    summary_parts = content.split("summary=")[1].strip()
                    if summary_parts.startswith("'"):
                        summary = summary_parts.split("'")[1]
                    else:
                        summary = summary_parts
        except Exception:
            summary = content

        return {
            "need_human_approval": {
                "tool_call_id": getattr(tool_msg, "tool_call_id", None),
                "severity": severity,
                "summary": summary,
            }
        }
    return {}


from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt


def human_approval(state: State) -> dict:
    approval_data = state.get("need_human_approval")
    if not approval_data:
        return {}

    # Interrupt to get human input (e.g., supervisor's response)
    human_input = interrupt(
        {
            "question": "Supervisor input required",
            "severity": approval_data.get("severity", "unknown"),
            "summary": approval_data.get("summary", ""),
        }
    )

    return {
        "messages": [
            HumanMessage(content=f"[SUPERVISOR RESPONSE] {human_input}"),
        ],
        "need_human_approval": None,
    }


def build_graph(return_builder=False):
    builder = StateGraph(State)
    builder.add_node("sales_rep", sales_assistant)
    builder.add_node("customer_support", support_assistant)
    builder.add_node("sales_tools", create_tool_node_with_fallback(sales_tools))
    builder.add_node("support_tools", create_tool_node_with_fallback(support_tools))
    builder.add_node("after_sales_tool", after_sales_tool)
    builder.add_node("after_support_tool", after_support_tool)
    builder.add_node("human_approval", human_approval)

    def route_start(state: dict) -> str:
        dialog = state.get("dialog_state", [])
        return (
            "customer_support"
            if dialog and dialog[-1] == "customer_support"
            else "sales_rep"
        )

    def route_sales(state: dict) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and getattr(last_msg, "tool_calls", None):
            return "sales_tools"
        return END

    def route_support(state: dict) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and getattr(last_msg, "tool_calls", None):
            return "support_tools"
        return END

    def route_after_sales_tool(state: dict) -> str:
        dialog = state.get("dialog_state", [])
        if dialog and dialog[-1] == "customer_support":
            return "customer_support"
        return "sales_rep"

    def route_after_support_tool(state: dict) -> str:
        if state.get("need_human_approval"):
            return "human_approval"

        last_msg = state["messages"][-1]
        if (
            isinstance(last_msg, AIMessage)
            and "[SUPERVISOR RESPONSE]" in last_msg.content
        ):
            return "customer_support"

        return END

    builder.add_conditional_edges(START, route_start, ["sales_rep", "customer_support"])
    builder.add_conditional_edges("sales_rep", route_sales, ["sales_tools", END])
    builder.add_edge("sales_tools", "after_sales_tool")
    builder.add_conditional_edges(
        "after_sales_tool", route_after_sales_tool, ["sales_rep", "customer_support"]
    )
    builder.add_conditional_edges(
        "customer_support", route_support, ["support_tools", END]
    )
    builder.add_edge("support_tools", "after_support_tool")
    builder.add_conditional_edges(
        "after_support_tool",
        route_after_support_tool,
        ["human_approval", "customer_support"],
    )
    builder.add_edge("human_approval", "customer_support")

    if return_builder:
        return builder
    return builder.compile(checkpointer=MemorySaver())


graph = build_graph()
