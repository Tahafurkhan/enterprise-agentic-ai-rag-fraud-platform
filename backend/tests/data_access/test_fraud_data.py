from pathlib import Path
import sys
from types import SimpleNamespace

from databricks.sdk.service.sql import StatementParameterListItem
from databricks.sdk.service.sql import StatementState

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data_access.fraud_data import FraudDataAccess


class FakeStatementExecution:
    def __init__(self, response):
        self.response = response
        self.last_call = None

    def execute_statement(self, **kwargs):
        self.last_call = kwargs
        return self.response


class FakeWorkspaceClient:
    def __init__(self, response):
        self.statement_execution = FakeStatementExecution(
            response
        )


def successful_response():
    return SimpleNamespace(
        status=SimpleNamespace(
            state=StatementState.SUCCEEDED,
            error=None,
        ),
        manifest=SimpleNamespace(
            schema=SimpleNamespace(
                columns=[
                    SimpleNamespace(name="transaction_id"),
                    SimpleNamespace(name="transaction_amount"),
                    SimpleNamespace(name="merchant_name"),
                    SimpleNamespace(name="is_international"),
                ]
            )
        ),
        result=SimpleNamespace(
            data_array=[
                [
                    "txn-001",
                    150000.0,
                    "Merchant A",
                    False,
                ],
            ]
        ),
    )


def test_high_value_transactions_returns_rows():
    fake_client = FakeWorkspaceClient(
        successful_response()
    )

    data_access = FraudDataAccess(
        client=fake_client
    )

    result = data_access.get_high_value_transactions()

    assert result == [
        {
            "transaction_id": "txn-001",
            "transaction_amount": 150000.0,
            "merchant_name": "Merchant A",
            "is_international": False,
        }
    ]


def test_high_value_query_uses_parameterized_threshold():
    fake_client = FakeWorkspaceClient(
        successful_response()
    )

    data_access = FraudDataAccess(
        client=fake_client
    )

    data_access.get_high_value_transactions(
        minimum_amount=200000,
        limit=10,
    )

    call = fake_client.statement_execution.last_call

    assert call["warehouse_id"] == data_access.warehouse_id
    assert call["catalog"] == data_access.catalog
    assert call["schema"] == data_access.schema

    assert {
    parameter.name: parameter.value
    for parameter in call["parameters"]
} == {
    "minimum_amount": "200000",
}


def test_fraud_card_alerts_exclude_sensitive_fields():
    fake_client = FakeWorkspaceClient(
        successful_response()
    )

    data_access = FraudDataAccess(
        client=fake_client
    )

    result = data_access.get_fraud_card_alerts(
        limit=10
    )

    call = fake_client.statement_execution.last_call

    statement = call["statement"].lower()

    assert "card_number" not in statement
    assert "customer_email" not in statement
    assert "customer_name" not in statement

    assert "transaction_id" in statement
    assert "amount" in statement
    assert result


def test_velocity_query_is_executed():
    fake_client = FakeWorkspaceClient(
        successful_response()
    )

    data_access = FraudDataAccess(
        client=fake_client
    )

    result = data_access.get_transaction_velocity(
        limit=5
    )

    call = fake_client.statement_execution.last_call

    assert "transaciton_count_by_minute" in call["statement"]

    assert "window_start" in call["statement"]
    assert "window_end" in call["statement"]
    assert "transaction_count" in call["statement"]

    assert result


def test_failed_databricks_execution_raises():
    response = SimpleNamespace(
        status=SimpleNamespace(
            state="FAILED",
            error="SQL execution failed",
        ),
        manifest=None,
        result=None,
    )

    fake_client = FakeWorkspaceClient(
        response
    )

    data_access = FraudDataAccess(
        client=fake_client
    )

    with pytest.raises(RuntimeError):
        data_access.get_fraud_card_alerts()

