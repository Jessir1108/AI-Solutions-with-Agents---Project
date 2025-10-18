import pytest
from src.tools import RouteToCustomerSupport, cart_tool, set_thread_id, view_cart


def test_cart_tool_add_operation():
    set_thread_id("test-cart-thread")
    result = cart_tool.invoke(
        {"cart_operation": "add", "product_id": 123, "quantity": 2}
    )
    assert "added" in result.lower()


def test_cart_tool_view_operation():
    set_thread_id("test-view-thread")
    cart_tool.invoke({"cart_operation": "add", "product_id": 456, "quantity": 1})
    result = view_cart.invoke({})
    assert "cart contains" in result.lower()


def test_route_schema_basic_usage():
    obj = RouteToCustomerSupport(reason="missing package")
    assert obj.reason == "missing package"


def test_route_schema_validation():
    obj = RouteToCustomerSupport(reason="defective item")
    assert obj.reason == "defective item"
    with pytest.raises(Exception):
        RouteToCustomerSupport()


def test_route_to_customer_support_schema_usage():
    obj = RouteToCustomerSupport(reason="missing package")
    assert obj.reason == "missing package"
