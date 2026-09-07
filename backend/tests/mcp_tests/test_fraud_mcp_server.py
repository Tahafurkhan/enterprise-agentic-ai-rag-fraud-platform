import asyncio

import pytest

from backend.mcp.fraud_mcp_server import mcp


EXPECTED_TOOLS = {
    "get_high_value_transactions",
    "get_fraud_card_alerts",
    "get_transaction_velocity",
    "list_approved_datasets",
    "get_safe_dataset_schema",
    "get_dataset_rows",
}


def run_async(coro):
    return asyncio.run(coro)


def test_mcp_registers_expected_tools():
    tools = run_async(mcp.list_tools())

    tool_names = {tool.name for tool in tools}

    assert tool_names == EXPECTED_TOOLS


def test_mcp_transaction_velocity():
    result = run_async(
        mcp.call_tool(
            "get_transaction_velocity",
            {"limit": 5},
        )
    )

    assert result.is_error is False
    assert result.structured_content is not None

    records = result.structured_content["result"]

    assert isinstance(records, list)
    assert len(records) <= 5

    if records:
        assert "window_start" in records[0]
        assert "window_end" in records[0]
        assert "transaction_count" in records[0]


def test_mcp_dataset_rows_are_sanitized():
    result = run_async(
        mcp.call_tool(
            "get_dataset_rows",
            {
                "dataset_name": "high_value_transactions_alert",
                "limit": 5,
            },
        )
    )

    assert result.is_error is False
    assert result.structured_content is not None

    records = result.structured_content["result"]

    assert isinstance(records, list)

    sensitive_fields = {
        "card_number",
        "customer_email",
        "customer_name",
    }

    for record in records:
        assert sensitive_fields.isdisjoint(record.keys())


def test_mcp_rejects_unapproved_dataset():
    with pytest.raises(Exception, match="Dataset 'random_table' is not approved"):
        run_async(
            mcp.call_tool(
                "get_dataset_rows",
                {
                    "dataset_name": "random_table",
                    "limit": 5,
                },
            )
        )


def test_mcp_lists_only_logical_approved_datasets():
    result = run_async(
        mcp.call_tool(
            "list_approved_datasets",
            {},
        )
    )

    assert result.is_error is False
    assert result.structured_content is not None

    datasets = result.structured_content["result"]

    dataset_names = {
        item["dataset_name"]
        for item in datasets
    }

    assert dataset_names == {
        "fraud_card_alert",
        "high_value_transactions_alert",
        "transaction_velocity",
        "transaction_velocity_sliding_window",
    }

    for item in datasets:
        assert "table_name" not in item


def test_mcp_safe_schema_excludes_sensitive_fields():
    result = run_async(
        mcp.call_tool(
            "get_safe_dataset_schema",
            {
                "dataset_name": "high_value_transactions_alert",
            },
        )
    )

    assert result.is_error is False
    assert result.structured_content is not None

    schema = result.structured_content["result"]

    column_names = {
        column["col_name"].lower()
        for column in schema
    }

    assert "card_number" not in column_names
    assert "customer_email" not in column_names
    assert "customer_name" not in column_names