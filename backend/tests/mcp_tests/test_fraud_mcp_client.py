
import pytest

from backend.mcp.fraud_mcp_client import FraudMCPClient


@pytest.mark.asyncio
async def test_mcp_client_calls_high_value_transactions():
    """
    Verify that the MCP client can call the governed
    high-value transaction tool.
    """

    client = FraudMCPClient()

    result = await client.call_tool(
        "get_high_value_transactions",
        {
            "minimum_amount": 200000,
            "limit": 10,
        },
    )

    assert result is not None
    assert isinstance(result, (list, dict))


@pytest.mark.asyncio
async def test_mcp_client_calls_fraud_card_alerts():
    """
    Verify that the MCP client can call the governed
    fraud-card-alert tool.
    """

    client = FraudMCPClient()

    result = await client.call_tool(
        "get_fraud_card_alerts",
        {
            "limit": 5,
        },
    )

    assert result is not None
    assert isinstance(result, (list, dict))


@pytest.mark.asyncio
async def test_mcp_client_calls_transaction_velocity():
    """
    Verify that the MCP client can call the governed
    transaction velocity tool.
    """

    client = FraudMCPClient()

    result = await client.call_tool(
        "get_transaction_velocity",
        {
            "limit": 5,
        },
    )

    assert result is not None
    assert isinstance(result, (list, dict))


@pytest.mark.asyncio
async def test_mcp_client_lists_approved_datasets():
    """
    Verify that dataset discovery is available through MCP.
    """

    client = FraudMCPClient()

    result = await client.call_tool(
        "list_approved_datasets",
        {},
    )

    assert isinstance(result, list)

    dataset_names = {
        dataset["dataset_name"]
        for dataset in result
    }

    assert "fraud_card_alert" in dataset_names
    assert "high_value_transactions_alert" in dataset_names


@pytest.mark.asyncio
async def test_mcp_client_gets_safe_schema():
    """
    Verify that the MCP client can retrieve a safe schema.
    """

    client = FraudMCPClient()

    result = await client.call_tool(
        "get_safe_dataset_schema",
        {
            "dataset_name": "high_value_transactions_alert",
        },
    )

    assert isinstance(result, list)

    for column in result:
        assert "card_number" not in column
        assert "customer_email" not in column
        assert "customer_name" not in column


@pytest.mark.asyncio
async def test_mcp_client_rejects_unknown_tool():
    """
    Verify that arbitrary tool names cannot be executed.
    """

    client = FraudMCPClient()

    with pytest.raises(Exception):
        await client.call_tool(
            "execute_arbitrary_sql",
            {
                "sql": "SELECT * FROM some_table",
            },
        )

