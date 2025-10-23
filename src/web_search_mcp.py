import os
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)


async def _load_brave_tool():
    brave_api_key = os.environ.get("BRAVE_API_KEY", "")

    if not brave_api_key:

        @tool
        def brave_web_search_fallback(query: str) -> str:
            return "Web search is unavailable. Please set BRAVE_API_KEY environment variable."

        return [brave_web_search_fallback]

    try:
        client = MultiServerMCPClient(
            {
                "brave": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "@brave/brave-search-mcp-server",
                        "--transport",
                        "stdio",
                        "--brave-api-key",
                        brave_api_key,
                    ],
                    "transport": "stdio",
                }
            }
        )

        tools = await client.get_tools()
        return tools

    except Exception as e:
        logger.error(f"Error loading Brave search tools: {e}")

        @tool
        def brave_web_search_fallback(query: str) -> str:
            return f"Web search is unavailable due to error: {str(e)}"

        return [brave_web_search_fallback]


def get_brave_web_search_tool_sync():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return loop.run_until_complete(_load_brave_tool())
    else:
        return asyncio.run(_load_brave_tool())
