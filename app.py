#!/usr/bin/env python
# app.py

import json
import os
import sys
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

# Add the project root to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph.types import Command

# Import from your existing codebase
from src.graph import graph

# Set page configuration
st.set_page_config(
    page_title="Shopping Assistant Demo",
    page_icon="🛒",
    layout="wide",
)

# Function to get price from products.csv
def get_product_price(product_id):
    """Get price from products.csv file or products_with_prices.csv if available"""
    try:
        # First try to load from products_with_prices.csv
        if os.path.exists("./products_with_prices.csv"):
            df = pd.read_csv("./products_with_prices.csv")
            if "price" in df.columns:
                row = df[df["product_id"] == int(product_id)]
                if not row.empty and not pd.isna(row["price"].iloc[0]):
                    return float(row["price"].iloc[0])

        # Then try from regular products.csv
        df = pd.read_csv("./dataset/products.csv")
        if "price" in df.columns:
            row = df[df["product_id"] == int(product_id)]
            if not row.empty and not pd.isna(row["price"].iloc[0]):
                return float(row["price"].iloc[0])

        # Fallback to deterministic price based on product_id
        return float(f"{(int(product_id) % 100) + 0.99:.2f}")
    except Exception as e:
        if st.session_state.get("debug_mode", False):
            print(f"Error getting price: {e}")
        return 0.99  # Default fallback price


# Custom CSS for better styling
st.markdown(
    """
<style>
    .user-bubble {
        background-color: #e6f7ff;
        color: black;  /* Explicit text color */
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 80%;
        margin-left: auto;
        border: 1px solid #ccecff;
    }
    .assistant-bubble {
        background-color: #f0f0f0;
        color: black;  /* Explicit text color */
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 80%;
        margin-right: auto;
        border: 1px solid #e0e0e0;
    }
    .tool-bubble {
        background-color: #fff8e1;
        color: black;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 80%;
        margin-right: auto;
        border: 1px solid #ffe0b2;
        font-family: 'Courier New', monospace;
    }
    .supervisor-bubble {
        background-color: #f5e6ff;
        color: black;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 80%;
        border: 1px solid #e6ccff;
    }
    .chat-header {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .chat-title {
        margin-left: 10px;
        font-weight: bold;
    }
    .debug-panel {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 10px;
        border: 1px solid #eaeaea;
        font-family: monospace;
        font-size: 0.9em;
    }
    .small-text {
        font-size: 0.8em;
        color: #666;
    }
    .cart-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 10px;
        margin: 4px 0;
        background-color: #f8f8f8;
        border-radius: 8px;
        border: 1px solid #e6e6e6;
    }
    .cart-item .product-name {
        flex-grow: 1;
        margin-right: 10px;
        font-size: 0.9em;
        color: #000000;  /* Explicitly set text color to black */
    }
    .cart-item .product-id {
        color: #888;
        font-size: 0.8em;
        font-family: monospace;
    }
    .cart-item .product-quantity {
        color: #444;
        font-size: 0.9em;
        font-weight: bold;
        background-color: #e6e6e6;
        border-radius: 12px;
        padding: 2px 8px;
        margin-left: 8px;
    }
    .cart-item .product-price {
        color: #2a6f3b;
        font-size: 0.9em;
        font-weight: bold;
        padding: 2px 8px;
    }
    .cart-empty {
        text-align: center;
        padding: 15px;
        color: #888;
        font-style: italic;
    }
    .cart-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .cart-toggle {
        cursor: pointer;
        user-select: none;
    }
    .sidebar-section {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
        border: 1px solid #eaeaea;
    }
    .cart-total {
        text-align: right;
        padding: 8px;
        font-weight: bold;
        border-top: 1px solid #e0e0e0;
        margin-top: 8px;
    }
    .price-total {
        color: #2a6f3b;
        font-size: 1.1em;
        font-weight: bold;
        margin-top: 4px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
def init_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_approval" not in st.session_state:
        st.session_state.pending_approval = None
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = "sales_rep"
    if "show_cart" not in st.session_state:
        st.session_state.show_cart = True
    if "cart_items" not in st.session_state:
        st.session_state.cart_items = (
            {}
        )  # Dictionary to store cart items with quantities and prices


init_session()

# Function to format tool calls nicely
def format_tool_call(tool_call):
    tool_name = tool_call.get("name", "unknown")
    args = tool_call.get("args", {})

    # Format arguments nicely
    formatted_args = []
    for key, value in args.items():
        formatted_args.append(f"{key}='{value}'")

    return f"{tool_name}({', '.join(formatted_args)})"


# Function to get the current state
def get_current_state():
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    snapshot = graph.get_state(config).values
    dialog_state = snapshot.get("dialog_state", [])
    current_mode = dialog_state[-1] if dialog_state else "sales_rep"
    return snapshot, current_mode


def direct_cart_update():
    """
    Directly gets the current cart state without going through the LLM.
    This ensures the UI always shows the true cart state.
    """
    # Import the functions we need from tools.py
    # Set the thread ID to ensure we get the right cart
    from src.tools import _product_lookup, get_cart, set_thread_id

    set_thread_id(st.session_state.thread_id)

    # Get the current cart directly
    cart = get_cart()

    # Make sure it's valid
    if isinstance(cart, list) and len(cart) > 0 and "Session error" in cart[0]:
        return

    # Update the UI cart state directly
    cart_items = {}
    for pid, qty in cart.items():
        title = _product_lookup.get(pid, "Unknown Product")
        price = get_product_price(pid)
        cart_items[str(pid)] = {"name": title, "quantity": qty, "price": price}

    # Update the session state
    st.session_state.cart_items = cart_items


def parse_cart_from_tool_message(content):
    """
    Parse the cart information directly from the view_cart tool output.
    This function extracts product IDs, names, quantities, and prices from the tool message.
    """
    cart_items = {}
    if content is None:
        return cart_items

    # Check if cart has items
    if "Your cart contains:" in content:
        lines = content.split("\n")
        for line in lines:
            if line.strip().startswith("- ") and "(ID:" in line and "×" in line:
                try:
                    # Parse line like "- Product Name (ID: 123) × 2"
                    product_part = line.split(" (ID:")[0].strip("- ")
                    product_name = product_part.strip()

                    id_part = line.split("(ID: ")[1].split(")")[0].strip()
                    product_id = id_part

                    quantity_part = line.split("×")[1].strip()
                    quantity = int(quantity_part)

                    # Get the price for this product
                    price = get_product_price(int(product_id))

                    cart_items[product_id] = {
                        "name": product_name,
                        "quantity": quantity,
                        "price": price,
                    }
                except Exception as e:
                    if st.session_state.debug_mode:
                        print(f"Error parsing cart line: {line} - {str(e)}")
    return cart_items

import asyncio
from langgraph.types import Command

def process_user_input():
    if st.session_state.pending_approval:
        return

    user_input = st.session_state.user_input
    if not user_input.strip():
        return

    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Clear the input box
    st.session_state.user_input = ""

    # Process through LangGraph
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    input_state = {"messages": [("user", user_input)]}

    try:
        # Async invoke
        result = asyncio.run(graph.ainvoke(input_state, config))

        # Get updated state
        snapshot = graph.get_state(config)
        state = snapshot.values

        # Check for interrupts (need for human approval)
        interrupt_info = None
        for task in snapshot.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                for intr in task.interrupts:
                    if hasattr(intr, "value") and isinstance(intr.value, dict):
                        interrupt_info = intr.value

        # Update current mode
        dialog_state = state.get("dialog_state", [])
        st.session_state.current_mode = (
            dialog_state[-1] if dialog_state else "sales_rep"
        )

        # Get the latest messages
        messages = state["messages"]

        # Process new messages
        new_messages = [msg for msg in messages if msg not in st.session_state.messages]
        for msg in new_messages:
            msg_type = msg.__class__.__name__
            content = getattr(msg, "content", "")

            if msg_type == "AIMessage":
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        formatted_call = format_tool_call(tc)
                        tool_name = tc.get("name", "unknown_tool")
                        st.session_state.chat_history.append(
                            {
                                "role": "tool_call",
                                "content": formatted_call,
                                "tool_name": tool_name,
                            }
                        )
                else:
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": content,
                            "mode": st.session_state.current_mode,
                        }
                    )
            elif msg_type == "ToolMessage":
                tool_call_id = getattr(msg, "tool_call_id", "unknown_id")
                tool_name = "unknown_tool"

                # Find the corresponding tool call to get the tool name
                for prev_msg in messages:
                    if hasattr(prev_msg, "tool_calls"):
                        for tc in prev_msg.tool_calls:
                            if tc.get("id") == tool_call_id:
                                tool_name = tc.get("name", "unknown_tool")
                                break

                st.session_state.chat_history.append(
                    {"role": "tool_result", "content": content, "tool_name": tool_name}
                )

                # Try to update cart from view_cart if that was the tool used
                if tool_name == "view_cart":
                    cart_items = parse_cart_from_tool_message(content)
                    if cart_items:
                        st.session_state.cart_items = cart_items

        # Store the updated messages
        st.session_state.messages = messages

        # Directly update cart display
        direct_cart_update()

        # Handle interrupts
        if interrupt_info:
            severity = interrupt_info.get("severity", "unknown")
            summary = interrupt_info.get("summary", "No details")
            message = interrupt_info.get("message", "")

            st.session_state.pending_approval = {
                "severity": severity,
                "summary": summary,
                "message": message,
            }

    except Exception as e:
        st.session_state.chat_history.append(
            {"role": "error", "content": f"An error occurred: {str(e)}"}
        )



def view_current_cart():
    """
    Function to explicitly request the cart status by sending a message to view the cart.
    This is a direct, transparent approach with no hidden messages.
    """
    cart_message = "What's in my cart?"
    st.session_state.chat_history.append({"role": "user", "content": cart_message})

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    input_state = {"messages": [("user", cart_message)]}

    try:
        # Async invoke
        result = asyncio.run(graph.ainvoke(input_state, config))

        # Get updated state
        snapshot = graph.get_state(config)
        state = snapshot.values

        # Update current mode
        dialog_state = state.get("dialog_state", [])
        st.session_state.current_mode = (
            dialog_state[-1] if dialog_state else "sales_rep"
        )

        # Get the latest messages
        messages = state["messages"]

        # Process new messages
        new_messages = [msg for msg in messages if msg not in st.session_state.messages]
        for msg in new_messages:
            msg_type = msg.__class__.__name__
            content = getattr(msg, "content", "")

            if msg_type == "AIMessage":
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        formatted_call = format_tool_call(tc)
                        tool_name = tc.get("name", "unknown_tool")
                        st.session_state.chat_history.append(
                            {
                                "role": "tool_call",
                                "content": formatted_call,
                                "tool_name": tool_name,
                            }
                        )
                else:
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": content,
                            "mode": st.session_state.current_mode,
                        }
                    )
            elif msg_type == "ToolMessage":
                tool_call_id = getattr(msg, "tool_call_id", "unknown_id")
                tool_name = "unknown_tool"

                for prev_msg in messages:
                    if hasattr(prev_msg, "tool_calls"):
                        for tc in prev_msg.tool_calls:
                            if tc.get("id") == tool_call_id:
                                tool_name = tc.get("name", "unknown_tool")
                                break

                st.session_state.chat_history.append(
                    {"role": "tool_result", "content": content, "tool_name": tool_name}
                )

                if tool_name == "view_cart":
                    cart_items = parse_cart_from_tool_message(content)
                    if cart_items:
                        st.session_state.cart_items = cart_items

        # Store the updated messages
        st.session_state.messages = messages

        # Direct cart update
        direct_cart_update()

    except Exception as e:
        st.session_state.chat_history.append(
            {"role": "error", "content": f"Error viewing cart: {str(e)}"}
        )




# Function to toggle cart visibility
def toggle_cart():
    st.session_state.show_cart = not st.session_state.show_cart

def process_supervisor_input():
    if not st.session_state.supervisor_input.strip():
        return

    supervisor_response = st.session_state.supervisor_input

    st.session_state.chat_history.append(
        {"role": "supervisor", "content": supervisor_response}
    )

    st.session_state.supervisor_input = ""
    st.session_state.pending_approval = None

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    try:
        # Async invoke
        result = asyncio.run(graph.ainvoke(Command(resume=supervisor_response), config))

        snapshot = graph.get_state(config)
        state = snapshot.values

        # Update messages and mode
        st.session_state.messages = state["messages"]
        dialog_state = state.get("dialog_state", [])
        st.session_state.current_mode = (
            dialog_state[-1] if dialog_state else "sales_rep"
        )

        if len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            content = getattr(last_message, "content", "")
            if last_message.__class__.__name__ == "AIMessage":
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    for tc in last_message.tool_calls:
                        formatted_call = format_tool_call(tc)
                        tool_name = tc.get("name", "unknown_tool")
                        st.session_state.chat_history.append(
                            {
                                "role": "tool_call",
                                "content": formatted_call,
                                "tool_name": tool_name,
                            }
                        )
                else:
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": content,
                            "mode": st.session_state.current_mode,
                        }
                    )

    except Exception as e:
        st.session_state.chat_history.append(
            {"role": "error", "content": f"Error processing approval: {str(e)}"}
        )


# Function to reset the conversation
def reset_conversation():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.pending_approval = None
    st.session_state.current_mode = "sales_rep"
    st.session_state.cart_items = {}


def toggle_debug():
    st.session_state.debug_mode = not st.session_state.debug_mode


# Calculate cart total items and price
def get_cart_totals():
    total_items = 0
    total_price = 0.0
    for item_id, item_data in st.session_state.cart_items.items():
        quantity = item_data.get("quantity", 0)
        price = item_data.get("price", 0.0)
        total_items += quantity
        total_price += quantity * price
    return total_items, total_price


# Sidebar for controls and info
with st.sidebar:
    # st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=50)
    st.title("Shopping Assistant")

    # Cart Section with toggle
    total_items, total_price = get_cart_totals()
    cart_label = (
        f"Your Cart ({total_items} items)" if total_items > 0 else "Your Cart (empty)"
    )

    st.markdown(
        f"""
    <div class="cart-header">
        <h3 style="margin:0">{cart_label}</h3>
        <div class="cart-toggle" onclick="document.getElementById('cart-content').style.display = document.getElementById('cart-content').style.display === 'none' ? 'block' : 'none';document.getElementById('cart-toggle-icon').innerText = document.getElementById('cart-toggle-icon').innerText === '▼' ? '▶' : '▼'">
            <span id="cart-toggle-icon">▼</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    cart_container = st.container()
    with cart_container:
        if st.session_state.cart_items:
            for item_id, item_data in st.session_state.cart_items.items():
                quantity = item_data.get("quantity", 0)
                price = item_data.get("price", 0.0)
                item_total = quantity * price
                st.markdown(
                    f"""
                <div class="cart-item">
                    <div class="product-name">{item_data['name']}</div>
                    <div style="display: flex; align-items: center;">
                        <div class="product-id">ID: {item_id}</div>
                        <div class="product-price">${price:.2f}</div>
                        <div class="product-quantity">×{quantity}</div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Display cart total
            st.markdown(
                f"""
            <div class="cart-total">
                <div>Total: {total_items} items</div>
                <div class="price-total">Total Price: ${total_price:.2f}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div class="cart-empty">
                Your cart is empty
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Divider
    st.divider()

    # Session Info
    st.markdown("""<div class="sidebar-section">""", unsafe_allow_html=True)
    st.markdown("### Session Info")
    st.markdown(f"**Thread ID:** {st.session_state.thread_id}")
    st.markdown(
        f"**Current Mode:** {st.session_state.current_mode.upper().replace('_', ' ')}"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.button("Reset Chat", on_click=reset_conversation)
    with col2:
        st.button("Toggle Debug", on_click=toggle_debug)

    # Enhanced cart interaction button
    if st.button("Show My Cart"):
        # First update the cart display directly - this is immediate and accurate
        direct_cart_update()

        # Then also trigger the conversational view to maintain the chat flow
        # This helps the user see the cart items in the conversation context
        view_current_cart()

    if st.session_state.debug_mode:
        st.divider()
        st.markdown("### Debug Information")
        snapshot, current_mode = get_current_state()

        st.markdown("**Dialog State:**")
        dialog_state = snapshot.get("dialog_state", [])
        st.json(dialog_state)

        st.markdown("**Need Human Approval:**")
        st.json(snapshot.get("need_human_approval"))

        st.markdown("### Message Count")
        message_types = {}
        for msg in st.session_state.messages:
            msg_type = msg.__class__.__name__
            message_types[msg_type] = message_types.get(msg_type, 0) + 1

        df = pd.DataFrame({"Count": message_types.values()}, index=message_types.keys())
        st.dataframe(df)

        st.markdown("### Current Cart Content")
        st.json(st.session_state.cart_items)

# Main chat interface
st.markdown(
    """
<div class="chat-header">
    <span style="font-size: 2em;">🛒</span>
    <span class="chat-title">Shopping Assistant</span>
</div>
""",
    unsafe_allow_html=True,
)

# Display chat messages
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            st.markdown(
                f'<div class="user-bubble"><strong>You:</strong><br>{content}</div>',
                unsafe_allow_html=True,
            )

        elif role == "assistant":
            mode = msg.get("mode", "assistant").upper().replace("_", " ")
            st.markdown(
                f'<div class="assistant-bubble"><strong>{mode}:</strong><br>{content}</div>',
                unsafe_allow_html=True,
            )

        elif role == "tool_call":
            tool_name = msg.get("tool_name", "TOOL")
            st.markdown(
                f'<div class="tool-bubble"><strong>🔧 CALLING {tool_name}:</strong><br>{content}</div>',
                unsafe_allow_html=True,
            )

        elif role == "tool_result":
            tool_name = msg.get("tool_name", "TOOL")
            st.markdown(
                f'<div class="tool-bubble"><strong>🔧 {tool_name} RESULT:</strong><br>{content}</div>',
                unsafe_allow_html=True,
            )

        elif role == "supervisor":
            st.markdown(
                f'<div class="supervisor-bubble"><strong>👨‍💼 SUPERVISOR:</strong><br>{content}</div>',
                unsafe_allow_html=True,
            )

        elif role == "error":
            st.error(content)

# Approval interface (shows up when approval is needed)
if st.session_state.pending_approval:
    approval_data = st.session_state.pending_approval

    st.warning(
        f"""
    ### Human Approval Required
    **Issue:** {approval_data.get('summary')}  
    **Severity:** {approval_data.get('severity')}  
    **Message:** {approval_data.get('message')}
    """
    )

    st.text_input(
        "Supervisor Response:",
        key="supervisor_input",
        on_change=process_supervisor_input,
        placeholder="Enter your approval decision or instructions...",
    )
else:
    # User input
    st.text_input(
        "Your message:",
        key="user_input",
        on_change=process_user_input,
        placeholder="Type a message...",
    )

# Footer
st.markdown(
    """
<div class="small-text">
Built with Streamlit + LangGraph •
</div>
""",
    unsafe_allow_html=True,
)

if __name__ == "__main__":
    # This won't run when imported as a module
    pass
