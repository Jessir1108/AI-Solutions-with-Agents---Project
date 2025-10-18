from unittest.mock import MagicMock, patch

from src.tools import cart_tool, search_tool, set_thread_id, view_cart


@patch("src.tools.get_vector_store")
def test_search_to_cart_workflow(mock_get_vector_store):
    mock_vectorstore = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {
        "product_id": 24852,
        "product_name": "Organic Bananas",
        "aisle": "fresh fruits",
        "department": "produce",
    }
    mock_doc.page_content = "Organic Bananas, found in the fresh fruits aisle."
    mock_vectorstore.similarity_search.return_value = [mock_doc]
    mock_get_vector_store.return_value = mock_vectorstore

    set_thread_id("test-workflow")
    search_result = search_tool.invoke({"query": "bananas"})
    assert "Organic Bananas" in search_result
    assert "24852" in search_result

    cart_result = cart_tool.invoke(
        {"cart_operation": "add", "product_id": 24852, "quantity": 1}
    )
    assert "added" in cart_result.lower()

    view_result = view_cart.invoke({})
    assert "24852" in view_result
    assert "cart contains" in view_result.lower()
    assert "ID: 24852" in view_result
