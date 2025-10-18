from src.tools import DEFAULT_USER_ID, set_user_id, structured_search_tool


def test_structured_search_basic_execution():
    result = structured_search_tool.invoke({"product_name": "milk"})
    assert isinstance(result, list)


def test_structured_search_with_department_filter():
    result = structured_search_tool.invoke({"department": "dairy eggs"})
    assert isinstance(result, list)
    if result and isinstance(result[0], dict) and "error" not in result[0]:
        assert all(
            item.get("department") == "dairy eggs"
            for item in result
            if "department" in item
        )


def test_structured_search_history_mode_with_user_set():
    set_user_id(DEFAULT_USER_ID)
    result = structured_search_tool.invoke({"history_only": True, "top_k": 5})
    assert isinstance(result, list)
    if result and isinstance(result[0], dict) and "error" not in result[0]:
        assert all("count" in item for item in result if isinstance(item, dict))
