import os


def test_get_brave_web_search_tool_sync_returns_tools(monkeypatch):
    # Ensure BRAVE_API_KEY exists
    monkeypatch.setenv("BRAVE_API_KEY", "fake-key")

    # Import inside function to so the patch is correctly applied to the env var
    from src import web_search_mcp
    tools = web_search_mcp.get_brave_web_search_tool_sync()

    assert isinstance(tools, list)
    for tool in tools:
        assert hasattr(tool, "name")
    assert any("brave" in tool.name.lower() for tool in tools)
