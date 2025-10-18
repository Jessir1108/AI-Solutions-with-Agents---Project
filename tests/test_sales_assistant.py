import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from src.assistants import sales_assistant


def test_sales_assistant_with_mocked_llm():
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = AIMessage(
        content="Hello! How can I help you today?"
    )
    # Add async version for ainvoke
    mock_runnable.ainvoke = AsyncMock(return_value=[AIMessage(
        content="Hello! How can I help you today?"
    )])
    
    config = RunnableConfig(configurable={"thread_id": "test-thread"})
    state = {"messages": [HumanMessage(content="Hi there")]}
    
    # Run the async function
    result = asyncio.run(sales_assistant(state, config, runnable=mock_runnable))
    
    assert isinstance(result, dict)
    assert "messages" in result


@patch("src.assistants.set_thread_id")
@patch("src.assistants.set_user_id")
def test_sales_assistant_sets_context_properly(mock_set_user, mock_set_thread):
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = AIMessage(content="Response")
    # Add async version for ainvoke
    mock_runnable.ainvoke = AsyncMock(return_value=[AIMessage(content="Response")])
    
    config = RunnableConfig(configurable={"thread_id": "test-thread-456"})
    state = {"messages": []}
    
    # Run the async function
    asyncio.run(sales_assistant(state, config, runnable=mock_runnable))
    
    mock_set_thread.assert_called_once_with("test-thread-456")
    mock_set_user.assert_called_once()


def test_sales_assistant_is_callable():
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = AIMessage(content="Hey there!")
    # Add async version for ainvoke
    mock_runnable.ainvoke = AsyncMock(return_value=[AIMessage(content="Hey there!")])
    
    config = RunnableConfig(configurable={"thread_id": "test"})
    
    # Run the async function
    result = asyncio.run(sales_assistant({"messages": []}, config, runnable=mock_runnable))
    
    assert isinstance(result, dict)
    assert "messages" in result