import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from .prompts import sales_rep_prompt, support_prompt
from .state import State
from .tools import (
    DEFAULT_USER_ID,
    EscalateToHuman,
    RouteToCustomerSupport,
    cart_tool,
    search_tool,
    set_thread_id,
    set_user_id,
    structured_search_tool,
    view_cart,
)

load_dotenv()
import pandas as pd

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or "Empty"
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

sales_tools = [
    RouteToCustomerSupport,
    search_tool,
    structured_search_tool,
    cart_tool,
    view_cart,
]
support_tools = [EscalateToHuman]

sales_runnable = sales_rep_prompt.partial(time=datetime.now) | llm.bind_tools(
    sales_tools
)
support_runnable = support_prompt.partial(time=datetime.now) | llm.bind_tools(
    support_tools
)


async def sales_assistant(
    state: State, config: RunnableConfig, runnable=sales_runnable
) -> dict:
    """Sales assistant that handles product queries and cart operations."""
    set_thread_id(config["configurable"]["thread_id"])
    set_user_id(DEFAULT_USER_ID)
    response = await runnable.ainvoke(state, config=config)
    return {"messages": response}


def support_assistant(state: State, config: RunnableConfig) -> dict:
    set_thread_id(config["configurable"]["thread_id"])
    return {"messages": support_runnable.invoke(state, config=config)}
