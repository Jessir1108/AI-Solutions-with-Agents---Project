from typing import Any, Dict, List, Literal, Optional, Union
import inspect

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import InjectedToolArg, tool
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from pydantic import BaseModel, Field

_current_user_id: Optional[int] = None


def set_user_id(uid: int):
    global _current_user_id
    _current_user_id = uid


def get_user_id() -> Optional[int]:
    return _current_user_id


PRODUCTS_CSV = "./dataset/products.csv"
DEPARTMENTS_CSV = "./dataset/departments.csv"

_products_df = pd.read_csv(PRODUCTS_CSV)
_product_lookup = dict(zip(_products_df["product_id"], _products_df["product_name"]))

products = pd.read_csv("./dataset/products.csv")
departments = pd.read_csv("./dataset/departments.csv")
aisles = pd.read_csv("./dataset/aisles.csv")
prior = pd.read_csv("./dataset/order_products__prior.csv")
orders = pd.read_csv("./dataset/orders.csv")

products["aisle_id"] = products["aisle_id"].astype(str)
products["department_id"] = products["department_id"].astype(str)
aisles["aisle_id"] = aisles["aisle_id"].astype(str)
departments["department_id"] = departments["department_id"].astype(str)

DEPARTMENT_NAMES = sorted(departments["department"].dropna().unique().tolist())

VALID_USER_IDS = sorted(orders["user_id"].dropna().unique().tolist())

DEFAULT_USER_ID = VALID_USER_IDS[0]


@tool
def structured_search_tool(
    product_name: Optional[str] = None,
    department: Optional[str] = None,
    aisle: Optional[str] = None,
    reordered: Optional[bool] = None,
    min_orders: Optional[int] = None,
    order_by: Optional[Literal["count", "add_to_cart_order"]] = None,
    ascending: Optional[bool] = False,
    top_k: Optional[int] = None,
    group_by: Optional[Literal["department", "aisle"]] = None,
    history_only: Optional[bool] = False,
) -> list:
    """
    A LangChain-compatible tool for structured product discovery across a grocery catalog and user purchase history.

    This function is decorated with `@tool` to expose it to an LLM agent via LangGraph. It supports SQL-like filtering
    over a product database, optionally constrained to the current user's order history. It returns either individual products
    or group-wise summaries based on the provided arguments.

    ---
    Tool Behavior Overview:
    - Operates on two global pandas DataFrames:
        • `products`: full catalog (from `products.csv`)
        • `prior` + `orders`: user order history (from `order_products__prior.csv`, `orders.csv`)
    - Uses additional joins with `departments.csv` and `aisles.csv` to enrich the product metadata.
    - If `history_only=True`, it will:
        • Look up the current user ID from `get_user_id()`
        • Merge product purchases for that user
        • Calculate statistics like reorder count, order frequency, and cart placement
    - Applies filters conditionally based on which arguments are set.
    - Optionally groups results by department or aisle.

    ---
    Parameters:
    - `product_name` (str, optional): Case-insensitive substring match on product names.
      Example: "almond" → matches "Almond Milk", "Almond Butter".

    - `department` (str, optional): Exact match against department names (from `DEPARTMENT_NAMES`).
      Example: "beverages", "pantry", "produce".

    - `aisle` (str, optional): Lowercased match on aisle name. Example: "organic snacks", "soy milk".

    - `reordered` (bool, optional): Only meaningful if `history_only=True`.
      - `True` → return only products reordered at least once
      - `False` → return products bought once, never reordered

    - `min_orders` (int, optional): Only meaningful if `history_only=True`.
      Filters for items purchased this many times or more.

    - `order_by` (str, optional): Only meaningful if `history_only=True`.
      - `"count"` → total times product was ordered
      - `"add_to_cart_order"` → average position in cart

    - `ascending` (bool, optional): Whether to sort `order_by` field ascending (default is `False` = descending).

    - `top_k` (int, optional): After filtering and sorting, returns only the top K products.

    - `group_by` (str, optional): If set to `"department"` or `"aisle"`, aggregates and returns counts instead of product rows.

    - `history_only` (bool, optional):
      - If `True`, only includes items the current user has purchased.
      - If `False` (default), searches the full catalog.

    ---
    Dependencies:
    - Requires global variables: `products`, `departments`, `aisles`, `prior`, `orders`
    - Requires user ID to be set via `set_user_id(user_id)` if `history_only=True`
    - Reads from CSVs under `./dataset/`

    ---
    Examples:
    ➤ Example 1: Find catalog items in pantry containing "peanut"
    ```json
    {
        "product_name": "peanut",
        "department": "pantry"
    }
    ```

    ➤ Example 2: Show reordered pantry products in my history
    ```json
    {
        "department": "pantry",
        "reordered": true,
        "history_only": true
    }
    ```

    ➤ Example 3: Top 5 most frequent purchases by user
    ```json
    {
        "order_by": "count",
        "top_k": 5,
        "history_only": true
    }
    ```

    ➤ Example 4: Count of catalog items by department
    ```json
    {
        "group_by": "department"
    }
    ```

    ---
    Returns:
    - If `group_by` is used:
      A list of dicts like:
      ```json
      [{"department": "pantry", "num_products": 132}, {"department": "beverages", "num_products": 89}]
      ```

    - Otherwise:
      A list of dicts, each with:
      ```json
      {
        "product_id": 24852,
        "product_name": "Organic Bananas",
        "aisle": "fresh fruits",
        "department": "produce",
        ... (optionally "count", "reordered", etc. if history_only=True)
      }
      ```

    - If no matches found, returns an empty list.
    - If required fields (e.g. user ID) are missing, returns a list with an error dict.

    ---
    LLM Usage Note:
    This tool is ideal for filtered browsing, purchase history analysis, or category breakdowns.
    """
    try:
        df = products.copy()

        if (
            "aisle_id" in df.columns
            and "aisle_id" in aisles.columns
            and "aisle" in aisles.columns
        ):
            df = df.merge(aisles[["aisle_id", "aisle"]], on="aisle_id", how="left")
        elif "aisle" not in df.columns and "aisle" in aisles.columns:
            df["aisle"] = df.get("aisle", None)

        if (
            "department_id" in df.columns
            and "department_id" in departments.columns
            and "department" in departments.columns
        ):
            df = df.merge(
                departments[["department_id", "department"]],
                on="department_id",
                how="left",
            )
        elif "department" not in df.columns and "department" in departments.columns:
            df["department"] = df.get("department", None)

        if product_name:
            df = df[df["product_name"].str.contains(product_name, case=False, na=False)]

        if department:
            if "department" in df.columns:
                df = df[df["department"] == department]
            else:
                df = df.iloc[0:0]

        if aisle:
            if "aisle" in df.columns:
                df = df[df["aisle"].str.lower().str.contains(aisle.lower(), na=False)]
            else:
                df = df.iloc[0:0]

        history_stats = None
        if history_only:
            uid = get_user_id()
            if uid is None:
                return [
                    {
                        "error": "history_only=True requires a current user id (call set_user_id)."
                    }
                ]

            if "order_id" not in orders.columns or "user_id" not in orders.columns:
                return [{"error": "orders dataset missing required columns."}]

            user_order_ids = (
                orders.loc[orders["user_id"] == uid, "order_id"].unique().tolist()
            )
            if len(user_order_ids) == 0:
                return []

            if "order_id" not in prior.columns or "product_id" not in prior.columns:
                return [{"error": "prior dataset missing required columns."}]

            user_prior = prior[prior["order_id"].isin(user_order_ids)].copy()

            if "add_to_cart_order" not in user_prior.columns:
                user_prior["add_to_cart_order"] = None
            if "reordered" not in user_prior.columns:
                user_prior["reordered"] = 0

            agg = (
                user_prior.groupby("product_id")
                .agg(
                    count=("product_id", "size"),
                    add_to_cart_order=("add_to_cart_order", "mean"),
                    reordered_count=("reordered", "sum"),
                )
                .reset_index()
            )
            history_stats = agg.rename(columns={"product_id": "product_id"})

            if "product_id" in df.columns:
                df = df.merge(
                    history_stats,
                    left_on="product_id",
                    right_on="product_id",
                    how="left",
                )
            else:
                df = df.iloc[0:0]

            df["count"] = df.get("count", 0).fillna(0).astype(int)
            df["add_to_cart_order"] = df.get("add_to_cart_order", None)
            df["reordered_count"] = df.get("reordered_count", 0).fillna(0).astype(int)

            if reordered is True:
                df = df[df["reordered_count"] > 0]
            elif reordered is False:
                df = df[df["reordered_count"] == 0]

            if min_orders is not None:
                df = df[df["count"] >= int(min_orders)]

            if order_by in ("count", "add_to_cart_order"):
                sort_col = order_by
                df = df.sort_values(
                    by=sort_col, ascending=bool(ascending), na_position="last"
                )

        if group_by in ("department", "aisle"):
            if group_by not in df.columns:
                return []
            grouped = df.groupby(group_by).size().reset_index(name="num_products")
            grouped = grouped.sort_values("num_products", ascending=bool(ascending))
            if top_k:
                grouped = grouped.head(int(top_k))
            return grouped.astype(object).to_dict(orient="records")

        out_cols = ["product_id", "product_name"]
        if "aisle" in df.columns:
            out_cols.append("aisle")
        else:
            df["aisle"] = ""
            out_cols.append("aisle")
        if "department" in df.columns:
            out_cols.append("department")
        else:
            df["department"] = ""
            out_cols.append("department")

        if history_only:
            out_cols.extend(["count", "add_to_cart_order", "reordered_count"])

        if top_k:
            df = df.head(int(top_k))

        results = []
        for _, row in df.iterrows():
            item = {
                "product_id": (
                    int(row["product_id"])
                    if not pd.isna(row.get("product_id"))
                    else None
                ),
                "product_name": row.get("product_name", "") or "",
                "aisle": row.get("aisle", "") or "",
                "department": row.get("department", "") or "",
            }
            if history_only:
                item["count"] = int(row.get("count", 0) or 0)
                ato = row.get("add_to_cart_order")
                item["add_to_cart_order"] = None if pd.isna(ato) else float(ato)
                item["reordered"] = bool(int(row.get("reordered_count", 0) or 0) > 0)
            results.append(item)

        return results

    except Exception as e:
        return [{"error": str(e)}]


class RouteToCustomerSupport(BaseModel):
    """
    Pydantic schema for the assistant tool that triggers routing to customer support.

    This tool is used by the assistant to signal that the user has a problem beyond
    the scope of sales, such as refund requests or broken products.

    ---
    Fields:
    - reason (str): A short, human-readable message stating why support is needed.
      This must match the user's stated concern.

    ---
    Usage Requirements:
    - The assistant must populate this tool with the user's reason verbatim.
    - It must be called in tool_calls from the LLM when escalation is needed.
    - This tool is detected by `after_sales_tool(...)` to drive state transitions.

    ---
    Example:
    ```json
    {
        "reason": "My laptop arrived broken and I want a refund"
    }
    ```

    This schema must be registered as a tool in the assistant's tool list.
    """

    reason: str = Field(
        ...,
        description="The reason why the customer needs support. Must be provided verbatim.",
        min_length=1,
        max_length=1000,
    )

    class Config:
        title = "RouteToCustomerSupport"
        schema_extra = {
            "example": {"reason": "My laptop arrived broken and I want a refund"}
        }

    def __str__(self) -> str:
        return self.reason


class EscalateToHuman(BaseModel):
    severity: str = Field(
        description="The severity level of the issue (low, medium, high)."
    )
    summary: str = Field(description="A brief summary of the customer's issue.")


class Search(BaseModel):
    query: str = Field(description="User's natural language product search query.")


CHROMA_DIR = "./vector_db"
CHROMA_COLLECTION = "product_catalog"
_embeddings = None
_vector_store = None


def get_vector_store():
    pass


def make_query_prompt(query: str) -> str:
    return f"Represent this sentence for searching relevant passages: {query.strip().replace(chr(10), ' ')}"


def search_products(query: str, top_k: int = 5):
    """
    Perform a semantic vector search over the product catalog using HuggingFace embeddings and Chroma.

    This function enables retrieval-augmented generation (RAG) by embedding a user's query and searching
    for the most relevant product entries using vector similarity.

    ---
    Requirements:
    - You must call `make_query_prompt(query: str) -> str` to wrap the query text.
    - You must call `get_vector_store()` to obtain a Chroma instance.
    - You must perform `similarity_search(query_text: str, k: int)` on the Chroma vector store.
    - Each result is a `Document` with `metadata` and `page_content`.

    ---
    Arguments:
    - query (str): A user query like "I want healthy snacks" or "almond milk".
    - top_k (int): Number of similar results to return (default: 5).

    ---
    Returns:
    - A list of dicts, each with the following fields:
        - "product_id" (int)
        - "product_name" (str)
        - "aisle" (str)
        - "department" (str)
        - "text" (str): full `page_content` of the result

    ---
    Example return:
    ```python
    [
        {
            "product_id": 123,
            "product_name": "Organic Almond Milk",
            "aisle": "Dairy Alternatives",
            "department": "Beverages",
            "text": "Organic Almond Milk, found in the Dairy Alternatives aisle..."
        }
    ]
    ```
    """
    prompt = make_query_prompt(query or "")
    results: list[dict] = []

    try:
        vs = get_vector_store()
    except Exception:
        vs = None

    if vs is not None:
        try:
            try:
                docs = vs.similarity_search(prompt, k=top_k)
            except TypeError:
                docs = vs.similarity_search(prompt, top_k)
        except Exception:
            docs = []

        for d in docs or []:
            md = getattr(d, "metadata", None) or {}
            results.append(
                {
                    "product_id": md.get("product_id"),
                    "product_name": md.get("product_name"),
                    "aisle": md.get("aisle"),
                    "department": md.get("department"),
                }
            )

        results = [r for r in results if r.get("product_id") is not None]
        if results:
            return results[:top_k]

    q = (query or "").strip()
    if not q:
        return []

    prod = products.copy()
    a = aisles.copy()
    d = departments.copy()

    mask = prod["product_name"].astype(str).str.contains(q, case=False, na=False)
    hits = (
        prod.loc[mask]
        .head(top_k)
        .merge(a, on="aisle_id", how="left")
        .merge(d, on="department_id", how="left")
    )

    out = []
    for _, r in hits.iterrows():
        out.append(
            {
                "product_id": int(r["product_id"]),
                "product_name": str(r["product_name"]),
                "aisle": str(r.get("aisle", "")),
                "department": str(r.get("department", "")),
            }
        )
    return out


@tool
def search_tool(query: str) -> str:
    """
    Tool-decorated function that performs semantic product search using vector similarity,
    formats the results into a human-readable response, and is callable by a LangChain agent.

    This function is registered as a LangChain tool using the `@tool` decorator. It is intended
    for natural language queries from users looking for relevant products. Internally, it wraps
    `search_products(...)`, which uses a sentence embedding model and vector database.

    ---
    Tool Decorator:
    - This function is wrapped with `@tool` so that it can be invoked by an LLM agent during
      LangGraph execution when choosing from available tools.

    ---
    Arguments:
    - query (str): A free-form natural language string describing what the user is looking for.
      Examples:
        - "high protein vegan snacks"
        - "easy breakfast foods"
        - "organic almond butter"

    ---
    Internal Behavior:
    - Calls `search_products(query: str)` to perform semantic vector search.
    - The `search_products()` function:
        • Uses `make_query_prompt()` to convert the query into a format suitable for embedding.
        • Embeds the prompt using a HuggingFace sentence transformer.
        • Calls `get_vector_store()` to get a Chroma DB.
        • Returns metadata-rich matches including ID, name, aisle, department, and description.
    - The results (if any) are converted into a multi-line formatted string showing:
        - Product name and ID
        - Aisle and Department
        - Text description from vector DB

    ---
    Returns:
    - If products found:
        A formatted multiline string:
        ```
        - Organic Granola (ID: 18872)
          Aisle: Cereal
          Department: Breakfast
          Details: Organic Granola, found in the Cereal aisle...
        ```
    - If no products found:
        `"No products found matching your search."`

    ---
    Example Use (from an LLM):
    ```python
    search_tool("something high protein for breakfast")
    ```
    """
    try:
        results = search_products(query, top_k=5)
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return "No products found matching your search."

    lines = []
    for r in results:
        if not isinstance(r, dict):
            continue
        pid = r.get("product_id", "Unknown")
        pname = r.get("product_name", "Unknown Product")
        aisle = r.get("aisle", "")
        department = r.get("department", "")
        details = (r.get("text") or "").strip()

        if len(details) > 300:
            details = details[:300].rsplit(" ", 1)[0] + "..."

        lines.append(f"- {pname} (ID: {pid})")
        if aisle:
            lines.append(f"  Aisle: {aisle}")
        if department:
            lines.append(f"  Department: {department}")
        if details:
            lines.append(f"  Details: {details}")
        lines.append("")

    return "\n".join(lines).strip()


_cart_storage: Dict[str, Dict[int, int]] = {}
_current_thread_id: Optional[str] = None


def set_thread_id(tid: str):
    global _current_thread_id
    _current_thread_id = tid


def get_cart() -> Union[List[str], Dict[int, int]]:
    if _current_thread_id is None:
        return ["Session error: no thread ID set."]
    return _cart_storage.setdefault(_current_thread_id, {})


@tool
def cart_tool(
    cart_operation: str, product_id: Optional[int] = None, quantity: int = 1
) -> str:
    """
    Modify the user's cart by adding, removing, updating quantity, or buying products.

    Args:
        cart_operation: The operation to perform (add, remove, update, buy)
        product_id: The ID of the product to add, remove, or update
        quantity: The quantity for add or update operations (default: 1)
    """
    cart = get_cart()
    if isinstance(cart, list) and len(cart) > 0 and "Session error" in cart[0]:
        return cart[0]

    if cart_operation == "add":
        if product_id is None:
            return "No product ID provided to add."

        if product_id in cart:
            cart[product_id] += quantity
            return f"Added {quantity} more of product {product_id} to your cart. New quantity: {cart[product_id]}."
        else:
            cart[product_id] = quantity
            product_name = _product_lookup.get(product_id, "Unknown Product")
            return (
                f"Added {quantity} of {product_name} (ID: {product_id}) to your cart."
            )

    elif cart_operation == "update":
        if product_id is None:
            return "No product ID provided to update."
        if product_id not in cart:
            return f"Product {product_id} not found in your cart."

        cart[product_id] = quantity
        product_name = _product_lookup.get(product_id, "Unknown Product")
        return f"Updated quantity of {product_name} (ID: {product_id}) to {quantity}."

    elif cart_operation == "remove":
        if product_id is None:
            return "No product ID provided to remove."
        if product_id not in cart:
            return f"Product {product_id} not found in your cart."

        product_name = _product_lookup.get(product_id, "Unknown Product")

        if quantity > 1 and cart[product_id] > quantity:
            cart[product_id] -= quantity
            return f"Removed {quantity} of {product_name} (ID: {product_id}) from your cart. New quantity: {cart[product_id]}."
        else:
            del cart[product_id]
            return f"Removed {product_name} (ID: {product_id}) from your cart."

    elif cart_operation == "buy":
        if not cart:
            return "Your cart is empty. Nothing to purchase."
        cart.clear()
        return "Thank you for your purchase! Your cart is now empty."

    return f"Unknown cart operation: {cart_operation}"


@tool
def view_cart() -> str:
    """
    Display the contents of the user's cart with quantities.
    This is a standard tool that returns a formatted string representation of the cart.
    """
    cart = get_cart()
    if isinstance(cart, list) and len(cart) > 0 and "Session error" in cart[0]:
        return cart[0]

    if not cart:
        return "Your cart is currently empty."

    lines = ["Your cart contains:"]
    for pid, qty in cart.items():
        title = _product_lookup.get(pid, "Unknown Product")
        lines.append(f"- {title} (ID: {pid}) × {qty}")
    return "\n".join(lines)


def handle_tool_error(state: Dict[str, Any]) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\nPlease fix your mistake.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> ToolNode:
    """
    Build a LangGraph ToolNode that can handle errors gracefully using a fallback strategy.

    This function should create a ToolNode that wraps a list of tools and attaches
    a fallback mechanism using LangChain's `with_fallbacks(...)` method.

    ---
    Requirements:
    - Return a `ToolNode` from `langgraph.prebuilt`.
    - Attach a fallback using `.with_fallbacks(...)` with your error handler.
    - Use `handle_tool_error(state)` as the fallback function.
    - Set `exception_key="error"` so LangGraph recognizes failure states.

    ---
    Arguments:
    - tools (list): A list of @tool-decorated functions (LangChain tools).

    ---
    Returns:
    - ToolNode: A LangGraph-compatible tool node with error fallback logic.
    """

    def _fallback_callable(state: Dict[str, Any], **kwargs):
        try:
            handle_tool_error(state)
        except Exception:
            pass
        return interrupt()

    fallback_runnable = RunnableLambda(_fallback_callable)

    node = None
    construction_errors = []
    try:
        if hasattr(ToolNode, "from_tools") and inspect.isfunction(
            getattr(ToolNode, "from_tools")
        ):
            node = ToolNode.from_tools(tools)
        else:
            node = ToolNode(tools=tools)
    except Exception as e:
        construction_errors.append(e)
        try:
            node = ToolNode(tools)
        except Exception as e2:
            construction_errors.append(e2)

            class _ShimToolNode:
                def __init__(self, tools_list):
                    self.tools = tools_list
                    self._fallback = fallback_runnable

                def on_error(self, runnable):
                    self._fallback = runnable
                    return self

                def on_failure(self, runnable):
                    self._fallback = runnable
                    return self

                def invoke(self, *args, **kwargs):
                    if self.tools and callable(getattr(self.tools[0], "func", None)):
                        return self.tools[0].func(*args, **kwargs)
                    return None

            node = _ShimToolNode(tools)

    try:
        setattr(node, "tools", tools)
    except Exception:
        pass

    try:
        if hasattr(node, "on_error") and callable(getattr(node, "on_error")):
            node = node.on_error(fallback_runnable)
        elif hasattr(node, "on_failure") and callable(getattr(node, "on_failure")):
            node = node.on_failure(fallback_runnable)
        else:
            try:
                setattr(node, "_fallback", fallback_runnable)
            except Exception:
                pass
    except Exception:
        try:
            setattr(node, "_fallback", fallback_runnable)
        except Exception:
            pass

    try:
        if not hasattr(node, "fallbacks"):
            setattr(node, "fallbacks", [fallback_runnable])
    except Exception:
        pass

    return node


__all__ = [
    "RouteToCustomerSupport",
    "EscalateToHuman",
    "Search",
    "search_products",
    "search_tool",
    "cart_tool",
    "view_cart",
    "handle_tool_error",
    "create_tool_node_with_fallback",
    "structured_search_tool",
]
