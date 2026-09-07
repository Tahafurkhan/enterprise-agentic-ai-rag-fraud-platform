from unittest.mock import Mock

import pytest

from backend.agents.tools.fraud_tools import build_fraud_tools


@pytest.fixture
def mock_data_access():
    data_access = Mock()

    data_access.registry = {
        "fraud_card_alert": {
            "table_name": "fraud_card_alert",
            "description": "Fraud card transaction alerts.",
        },
        "high_value_transactions_alert": {
            "table_name": "high_value_transactions_alert",
            "description": "High-value transaction alerts.",
        },
        "transaction_velocity": {
            "table_name": "transaciton_count_by_minute",
            "description": "Transaction counts aggregated by minute.",
        },
        "transaction_velocity_sliding_window": {
            "table_name": "transaciton_count_by_minute_sliding_window",
            "description": (
                "Transaction counts aggregated using a sliding window."
            ),
        },
    }

    return data_access


@pytest.fixture
def tools(mock_data_access):
    return build_fraud_tools(mock_data_access)


def get_tool(tools, name):
    return next(tool for tool in tools if tool.name == name)


def test_build_fraud_tools_returns_six_tools(tools):
    assert len(tools) == 6

    tool_names = [tool.name for tool in tools]

    assert tool_names == [
        "get_high_value_transactions",
        "get_fraud_card_alerts",
        "get_transaction_velocity",
        "list_approved_datasets",
        "get_safe_dataset_schema",
        "get_dataset_rows",
    ]


def test_get_high_value_transactions_delegates_to_data_access(
    mock_data_access,
    tools,
):
    mock_data_access.get_high_value_transactions.return_value = [
        {
            "transaction_id": "TX001",
            "transaction_amount": 150000.0,
        }
    ]

    tool = get_tool(tools, "get_high_value_transactions")

    result = tool.invoke(
        {
            "minimum_amount": 120000,
            "limit": 10,
        }
    )

    assert result == [
        {
            "transaction_id": "TX001",
            "transaction_amount": 150000.0,
        }
    ]

    mock_data_access.get_high_value_transactions.assert_called_once_with(
        minimum_amount=120000,
        limit=10,
    )


def test_get_fraud_card_alerts_delegates_to_data_access(
    mock_data_access,
    tools,
):
    mock_data_access.get_fraud_card_alerts.return_value = [
        {
            "alert_id": "ALERT001",
            "transaction_id": "TX001",
        }
    ]

    tool = get_tool(tools, "get_fraud_card_alerts")

    result = tool.invoke({"limit": 5})

    assert result == [
        {
            "alert_id": "ALERT001",
            "transaction_id": "TX001",
        }
    ]

    mock_data_access.get_fraud_card_alerts.assert_called_once_with(
        limit=5,
    )


def test_get_transaction_velocity_delegates_to_data_access(
    mock_data_access,
    tools,
):
    mock_data_access.get_transaction_velocity.return_value = [
        {
            "window_start": "15:52",
            "window_end": "15:53",
            "transaction_count": 177,
        }
    ]

    tool = get_tool(tools, "get_transaction_velocity")

    result = tool.invoke({"limit": 5})

    assert result == [
        {
            "window_start": "15:52",
            "window_end": "15:53",
            "transaction_count": 177,
        }
    ]

    mock_data_access.get_transaction_velocity.assert_called_once_with(
        limit=5,
    )


def test_list_approved_datasets_returns_logical_datasets(
    tools,
):
    tool = get_tool(tools, "list_approved_datasets")

    result = tool.invoke({})

    assert result == [
        {
            "dataset_name": "fraud_card_alert",
            "description": "Fraud card transaction alerts.",
        },
        {
            "dataset_name": "high_value_transactions_alert",
            "description": "High-value transaction alerts.",
        },
        {
            "dataset_name": "transaction_velocity",
            "description": "Transaction counts aggregated by minute.",
        },
        {
            "dataset_name": "transaction_velocity_sliding_window",
            "description": (
                "Transaction counts aggregated using a sliding window."
            ),
        },
    ]

    for dataset in result:
        assert "table_name" not in dataset


def test_get_safe_dataset_schema_delegates_to_data_access(
    mock_data_access,
    tools,
):
    mock_data_access.get_safe_dataset_schema.return_value = [
        {
            "col_name": "transaction_id",
            "data_type": "string",
            "comment": None,
        },
        {
            "col_name": "transaction_amount",
            "data_type": "double",
            "comment": None,
        },
    ]

    tool = get_tool(tools, "get_safe_dataset_schema")

    result = tool.invoke(
        {
            "dataset_name": "high_value_transactions_alert",
        }
    )

    assert result == [
        {
            "col_name": "transaction_id",
            "data_type": "string",
            "comment": None,
        },
        {
            "col_name": "transaction_amount",
            "data_type": "double",
            "comment": None,
        },
    ]

    mock_data_access.get_safe_dataset_schema.assert_called_once_with(
        dataset_name="high_value_transactions_alert",
    )


def test_get_dataset_rows_delegates_to_data_access(
    mock_data_access,
    tools,
):
    mock_data_access.get_dataset_rows.return_value = [
        {
            "transaction_id": "TX001",
            "transaction_amount": 150000.0,
        }
    ]

    tool = get_tool(tools, "get_dataset_rows")

    result = tool.invoke(
        {
            "dataset_name": "high_value_transactions_alert",
            "limit": 5,
        }
    )

    assert result == [
        {
            "transaction_id": "TX001",
            "transaction_amount": 150000.0,
        }
    ]

    mock_data_access.get_dataset_rows.assert_called_once_with(
        dataset_name="high_value_transactions_alert",
        limit=5,
    )


def test_get_dataset_rows_rejects_unapproved_dataset(
    mock_data_access,
    tools,
):
    mock_data_access.get_dataset_rows.side_effect = ValueError(
        "Dataset 'random_table' is not approved."
    )

    tool = get_tool(tools, "get_dataset_rows")

    with pytest.raises(
        ValueError,
        match="Dataset 'random_table' is not approved.",
    ):
        tool.invoke(
            {
                "dataset_name": "random_table",
                "limit": 5,
            }
        )


def test_get_dataset_rows_preserves_sensitive_data_boundary(
    mock_data_access,
    tools,
):
    mock_data_access.get_dataset_rows.return_value = [
        {
            "transaction_id": "TX001",
            "transaction_amount": 150000.0,
        }
    ]

    tool = get_tool(tools, "get_dataset_rows")

    result = tool.invoke(
        {
            "dataset_name": "high_value_transactions_alert",
            "limit": 5,
        }
    )

    assert "customer_email" not in result[0]
    assert "customer_name" not in result[0]
    assert "card_number" not in result[0]