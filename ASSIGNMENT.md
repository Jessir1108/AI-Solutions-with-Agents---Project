# AI-Powered Shopping Assistant

In this project, we will build and deploy an AI-powered conversational shopping assistant using LangGraph and Large Language Models. The system will handle customer inquiries, product searches, cart management, and escalate complex issues to human support when needed. You don't need to fully understand how LLMs work internally because we will cover that in detail later. For now, you can focus on implementing the required functions to make the system work.

The project structure is already defined and you will see the modules already have some code and comments to help you get started.

Below is the full project structure:

```
├── src
│   ├── assistants.py
│   ├── tools.py
│   ├── graph.py
│   ├── state.py
│   ├── prompts.py
│   ├── conversation_runner.py
├── tests
│   ├── test_cart_and_schema.py
│   ├── test_end_to_end.py
│   ├── test_graph.py
│   ├── test_sales_assistant.py
│   ├── test_structured_search.py
│   ├── test_tool_node.py
│   └── test_vector_search.py
├── dataset
│   ├── products.csv
│   ├── orders.csv
│   ├── aisles.csv
│   └── departments.csv
├── app.py
├── download_dataset.py
├── README.md
└── requirements.txt
```

Let's take a quick overview of each module:

- src: Contains the core implementation of the AI shopping assistant system.
    - `src/tools.py`: Implements the tools that the AI agents can use, including product search, cart management, and escalation functions. You must implement the following functions:
        - `structured_search_tool()`: Provides SQL-like filtering over the product catalog with purchase history support.
        - `search_products()`: Performs semantic vector search using embeddings.
        - `search_tool()`: LangChain tool wrapper for vector search.
        - `RouteToCustomerSupport`: Pydantic schema for escalating to support.
        - `create_tool_node_with_fallback()`: Error handling for tool execution.
    - `src/assistants.py`: Contains the AI agent implementations for sales and customer support.
    - `src/graph.py`: Implements the conversation flow using LangGraph.
    - `src/state.py`: Defines the conversation state structure.
    - `src/prompts.py`: Contains the prompts for the AI agents.
    - `src/conversation_runner.py`: Utilities for testing conversations.
- tests: This module contains unit tests and integration tests to validate the system's behavior.
- dataset: Contains the product catalog and order history data.
- app.py: Streamlit web interface for testing the shopping assistant.

Your task will be to complete the corresponding code on those parts marked with `#TODO` across all the modules. You can validate it's working as expected using the already provided tests. We encourage you to read the docstrings carefully as they contain detailed implementation guidance.

**Important**: Before starting, you must build the vector database as the system won't work without it.

## Building the Vector Database

The vector database is essential for semantic product search. You have two options:

### Option A: Google Colab (Recommended)
Building embeddings for thousands of products is computationally intensive. Google Colab provides free access to NVIDIA T4 GPUs which can process the embeddings 10-20x faster than a typical CPU.

1. Open Google Colab and create a new notebook
2. Enable GPU: Go to Runtime → Change runtime type → Hardware accelerator → T4 GPU
3. Clone your repository:
   ```python
   !git clone YOUR_REPO_URL
   %cd your-repo-name
   ```
4. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```
5. Build the vector database:
   ```python
   !python src/build_vector_db.py
   ```
   This should take 5-10 minutes with GPU vs 1-2 hours on CPU.
6. Download the `vector_db` folder and place it in your local project root directory.

### Option B: Local Machine
If you prefer to run locally:
```bash
python src/build_vector_db.py
```

### Downloading the Dataset (Required)

Before building the vector database or running any tests, you must download and extract the dataset.

Run the following script once from the project root:

```bash
python download_dataset.py
```

This will:
- Download the dataset ZIP file from Google Drive
- Unzip it directly into the `dataset/` folder

After running it, you should see files like `products.csv`, `orders.csv`, etc. inside the `dataset/` directory.

### Recommended way to work across all those files

Our recommendation for you about the order in which you should complete these files is the following:

## 1. `src/tools.py`

Inside this module, complete the functions in this order:

1. `RouteToCustomerSupport` class. This is a simple Pydantic BaseModel that defines the schema for escalating to customer support.

2. `search_products()` function. This function performs semantic vector search using the vector database. Read the docstring carefully as it explains the exact functions you need to call and the expected return format.

3. `search_tool()` function. This is a LangChain tool wrapper that formats the results from `search_products()` for the AI agent.

4. `structured_search_tool()` function. This provides SQL-like filtering over the product catalog. The docstring contains detailed examples and implementation hints. Pay attention to the `history_only` parameter and error handling.

5. `create_tool_node_with_fallback()` function. This implements error handling for tool execution using LangChain's fallback mechanism.

You can test your progress by running:
```console
$ pytest tests/test_structured_search.py -v
```

## 2. `src/assistants.py`

Inside this module, complete:

1. `sales_assistant()` function. This function sets up the thread and user context, then invokes the sales agent pipeline. The docstring explains exactly what you need to do, and you can look at the `support_assistant()` function below it for the pattern.

Now run the tests again to check if they are passing correctly.

## 3. Testing and Validation

Run the comprehensive test suite to validate your implementation:

```console
$ pytest tests/ -v
```

You can also test individual functions manually:

```python
from src.conversation_runner import run_single_turn
result = run_single_turn("Hi, I need bananas", "test-thread-123")
print(result)
```

## 4. Try the Web Interface

Once your functions are working and tests are passing:

```bash
streamlit run app.py
```

Test these scenarios:
- Product search: "I need healthy snacks"
- Cart management: "Add some bananas to my cart"
- Escalation: "I want a refund" (triggers human approval)

## 5. Implement web search through Brave Search MCP: https://github.com/brave/brave-search-mcp-server

For the web search exercise, you'll need Node.js installed to run `npx` commands. The MCP server runs via `npx -y @brave/brave-search-mcp-server` with appropriate transport and API key arguments. You'll also need to obtain a Brave Search API key from https://brave.com/search/api and add it to your `.env` file as `BRAVE_API_KEY=your_key_here`. The implementation involves using `langchain-mcp` adapters to connect to the MCP server and integrate the resulting tool into your sales assistant's toolset.

## Key Implementation Notes

When implementing the functions, keep these points in mind:

- **Read the docstrings carefully**: Each function has detailed documentation explaining inputs, outputs, and implementation requirements.
- **Use proper tool calling**: For LangChain tools, always use `.invoke({"param": value})` format, not direct function calls.
- **Handle errors gracefully**: Return appropriate error messages in the expected format rather than raising exceptions.
- **Follow the data flow**: Products have IDs that link to names, cart operations need thread IDs, and vector search returns different data than structured search.
- **Test incrementally**: Run tests after completing each function to catch issues early.

Good luck!
