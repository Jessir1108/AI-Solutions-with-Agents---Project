# src/state.py
from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """
    Updates the dialog state stack.
    """
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    dialog_state: Annotated[
        list[Literal["sales_rep", "customer_support"]], update_dialog_stack
    ]
    need_human_approval: Optional[Dict[str, Any]]
    supervisor_response: Optional[str]
