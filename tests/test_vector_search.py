from unittest.mock import MagicMock, patch
import pytest

from src.tools import search_products, search_tool

# --- Fixtures for global patching ---
@pytest.fixture(autouse=True)
def patch_get_vector_store():
    with patch("src.tools.get_vector_store") as mock_get_vector_store:
        mock_vectorstore = MagicMock()
        mock_doc1 = MagicMock()
        mock_doc1.metadata = {
            "product_id": 123,
            "product_name": "Organic Milk",
            "aisle": "dairy",
            "department": "dairy eggs",
        }
        mock_doc1.page_content = "Organic Milk..."
        mock_doc2 = MagicMock()
        mock_doc2.metadata = {
            "product_id": 456,
            "product_name": "Almond Milk",
            "aisle": "dairy alternatives",
            "department": "dairy eggs",
        }
        mock_doc2.page_content = "Almond Milk..."
        mock_vectorstore.similarity_search.return_value = [mock_doc1, mock_doc2]
        mock_get_vector_store.return_value = mock_vectorstore
        yield


# --- Unit tests ---

def test_search_products_with_mocked_vectorstore():
    result = search_products("milk")
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["product_id"] == 123
    assert result[1]["product_id"] == 456


@patch("src.tools.search_products")
def test_search_tool_integration(mock_search_products):
    mock_search_products.return_value = [
        {
            "product_id": 789,
            "product_name": "Greek Yogurt",
            "aisle": "yogurt",
            "department": "dairy eggs",
            "text": "Greek Yogurt...",
        }
    ]
    result = search_tool.invoke({"query": "yogurt"})
    assert isinstance(result, str)
    assert "Greek Yogurt" in result
    assert "ID: 789" in result


@patch("src.tools.search_products")
def test_search_tool_no_results(mock_search_products):
    mock_search_products.return_value = []
    result = search_tool.invoke({"query": "nonexistentproduct12345"})
    assert "No products found" in result


@patch("src.tools.search_products")
def test_search_tool_executes(mock_search_products):
    mock_search_products.return_value = [
        {
            "product_id": 101,
            "product_name": "Soy Milk",
            "aisle": "dairy",
            "department": "dairy eggs",
            "text": "Soy Milk Description",
        }
    ]
    result = search_tool.invoke({"query": "milk"})
    assert isinstance(result, str)
    assert "Soy Milk" in result


def test_search_products_executes():
    result = search_products("milk")
    assert isinstance(result, list)
    assert len(result) == 2
