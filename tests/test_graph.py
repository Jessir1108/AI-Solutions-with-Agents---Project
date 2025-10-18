from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from src.graph import after_sales_tool
from src import assistants
import asyncio


def test_after_sales_tool_route_detection():
    msg = ToolMessage(
        content="Routing...", name="RouteToCustomerSupport", tool_call_id="route-123"
    )
    result = after_sales_tool({"messages": [msg]})
    assert result == {"dialog_state": "customer_support"}


def test_after_sales_tool_no_routing():
    msg = ToolMessage(
        content="Found products", name="search_tool", tool_call_id="search-456"
    )
    result = after_sales_tool({"messages": [msg]})
    assert result == {}


def test_after_sales_tool_executes():
    msg = ToolMessage(content="routed", name="RouteToCustomerSupport", tool_call_id="1")
    result = after_sales_tool({"messages": [msg]})
    assert isinstance(result, dict)


@patch("src.conversation_runner.graph")
def test_graph_invocation_with_mock(mock_graph):
    from src.conversation_runner import run_single_turn

    mock_snapshot = MagicMock()
    mock_snapshot.values = {
        "messages": [HumanMessage(content="Hello"), AIMessage(content="Hi")],
        "dialog_state": ["sales_rep"],
    }
    mock_snapshot.tasks = []
    mock_graph.get_state.return_value = mock_snapshot
    mock_graph.invoke.return_value = {"messages": [AIMessage(content="Response")]}

    result = run_single_turn("Hello", "test-thread")
    assert "user_input" in result
    assert "response" in result
    assert "current_mode" in result


def test_sales_assistant_is_callable_offline():
    from src import assistants
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableConfig
    from unittest.mock import MagicMock, AsyncMock
    import pytest
    import inspect

    # Setup mock runnable that simulates LangChain response
    mock_runnable = MagicMock()
    # Set up both sync and async methods for the mock
    mock_runnable.invoke.return_value = [AIMessage(content="mock")]
    mock_runnable.ainvoke = AsyncMock(return_value=[AIMessage(content="mock")])

    config = RunnableConfig(configurable={"thread_id": "test-thread"})
    state = {"messages": []}

    # Confirm function accepts the correct signature
    if 'runnable' not in assistants.sales_assistant.__code__.co_varnames:
        pytest.fail("sales_assistant() does not accept a 'runnable' argument — are you using the updated version?")

    # Check if the function is async
    is_async = inspect.iscoroutinefunction(assistants.sales_assistant)
    
    # Try to run the function with the mock
    try:
        if is_async:
            # Run the async function
            result = asyncio.run(assistants.sales_assistant(state, config, runnable=mock_runnable))
        else:
            # Run the sync function
            result = assistants.sales_assistant(state, config, runnable=mock_runnable)
    except Exception as e:
        pytest.fail(f"sales_assistant() raised an exception: {type(e).__name__}: {e}")

    # Validate the result
    if result is None:
        pytest.fail("sales_assistant() returned None — is it implemented and returning a result?")
    if not isinstance(result, dict):
        pytest.fail(f"Expected result to be a dict, got {type(result)}: {result}")
    if "messages" not in result:
        pytest.fail(f"Result is missing 'messages' key: {result}")
    if not isinstance(result["messages"], list):
        pytest.fail(f"'messages' is not a list: {result}")
    if not result["messages"] or not isinstance(result["messages"][0], AIMessage):
        pytest.fail(f"'messages' must contain AIMessage objects, got: {result['messages']}")