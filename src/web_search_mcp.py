# src/web_search_mcp.py
import os
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)

async def _load_brave_tool():
    """
    TODO: Implement the MCP client connection for Brave search with proper error handling.
    
    This function should:
    1. Check if BRAVE_API_KEY exists in environment variables
    2. If no API key, return a fallback tool that explains web search is unavailable
    3. If API key exists, create a MultiServerMCPClient with:
       - Command: "npx"
       - Args: ["-y", "@brave/brave-search-mcp-server", "--transport", "stdio", "--brave-api-key", <key>]
       - Transport: "stdio"
    4. Get tools from the client and filter for the tools you want.
    5. Handle any exceptions and return appropriate fallback tools
    
    Returns:
        List of tools (always return a list, even with fallback tools)
    """
    brave_api_key = os.environ.get("BRAVE_API_KEY", "")
    return []

def get_brave_web_search_tool_sync():
    """Safe sync wrapper for Streamlit."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in an event loop → use run_until_complete
        return loop.run_until_complete(_load_brave_tool())
    else:
        return asyncio.run(_load_brave_tool())