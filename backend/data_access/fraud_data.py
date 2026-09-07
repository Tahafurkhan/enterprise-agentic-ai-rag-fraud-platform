
"""
Governed read-only access to approved Fraud Gold analytical datasets.

This module is intentionally separate from the Databricks Volume client.

The Volume client handles document/file operations.

This module handles read-only analytical access to approved Fraud Gold
tables through Databricks SQL Statement Execution.

Security principles:
- Only approved Gold tables may be queried.
- SQL is defined by application code rather than supplied by the LLM.
- Sensitive customer fields are never selected.
- Results are defensively sanitized before being returned to agents.
"""

from typing import Any, Dict, List

from .fraud_table_registry import FRAUD_TABLE_REGISTRY

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import (
    StatementParameterListItem,
    StatementState,
)

from ..config import (
    DATABRICKS_HOST,
    DATABRICKS_TOKEN,
    DATABRICKS_SQL_WAREHOUSE_ID,
    FRAUD_GOLD_CATALOG,
    FRAUD_GOLD_SCHEMA,
)


class FraudDataAccess:
    """
    Governed read-only access to approved Fraud Gold datasets.

    The agent does not provide arbitrary SQL.

    Each public method corresponds to an approved analytical operation.
    """

    APPROVED_TABLES = {
    "fraud_card_alert",
    "high_value_transactions_alert",
    "transaciton_count_by_minute",
    "transaciton_count_by_minute_sliding_window",
}

    SENSITIVE_FIELDS = {
        "card_number",
        "customer_email",
        "customer_name",
    }

    def __init__(
        self,
        client: WorkspaceClient | None = None,
    ):
        if not DATABRICKS_SQL_WAREHOUSE_ID:
            raise ValueError(
                "DATABRICKS_SQL_WAREHOUSE_ID is not configured."
            )

        if not FRAUD_GOLD_CATALOG:
            raise ValueError(
                "FRAUD_GOLD_CATALOG is not configured."
            )

        if not FRAUD_GOLD_SCHEMA:
            raise ValueError(
                "FRAUD_GOLD_SCHEMA is not configured."
            )

        self.client = client or WorkspaceClient(
            host=DATABRICKS_HOST,
            token=DATABRICKS_TOKEN,
        )
        
        self.registry = FRAUD_TABLE_REGISTRY

        self.warehouse_id = DATABRICKS_SQL_WAREHOUSE_ID
        self.catalog = FRAUD_GOLD_CATALOG
        self.schema = FRAUD_GOLD_SCHEMA

    def _sanitize_rows(
        self,
        rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
            """
            Defensively remove sensitive fields from returned records.

            Field matching is case-insensitive.
            """

            sensitive_fields = {
                field.lower()
                for field in self.SENSITIVE_FIELDS
            }

            sanitized_rows = []

            for row in rows:
                sanitized_row = {
                    key: value
                    for key, value in row.items()
                    if key.lower() not in sensitive_fields
                }

                sanitized_rows.append(sanitized_row)

            return sanitized_rows

    def _execute(
    self,
    statement: str,
    parameters: List[StatementParameterListItem] | None = None,
    row_limit: int = 100,
) -> List[Dict[str, Any]]:
        response = self.client.statement_execution.execute_statement(
            statement=statement,
            warehouse_id=self.warehouse_id,
            catalog=self.catalog,
            schema=self.schema,
            wait_timeout="30s",
            row_limit=row_limit,
            parameters=parameters or [],
        )

        if response.status is None:
            raise RuntimeError(
                "Databricks SQL execution returned no status."
            )

        if response.status.state != StatementState.SUCCEEDED:
            error = response.status.error

            if error is not None:
                raise RuntimeError(
                    f"Databricks SQL execution failed: {error}"
                )

            raise RuntimeError(
                f"Databricks SQL execution failed. "
                f"State={response.status.state}, "
                f"Status={response.status}"
            )

        if response.manifest is None or response.result is None:
            return []

        columns = [
            column.name
            for column in response.manifest.schema.columns
        ]

        rows = response.result.data_array or []

        records = [
            dict(zip(columns, row))
            for row in rows
        ]

        return self._sanitize_rows(records)
    
    def get_dataset_schema(
    self,
    dataset_name: str,
) -> List[Dict[str, Any]]:
        """
        Discover the schema of an approved Fraud Gold dataset.

        Only actual table columns are returned.
        Databricks DESCRIBE metadata rows are excluded.
        Duplicate column names are removed.
        """

        definition = self.registry.get(dataset_name)

        if definition is None:
            raise ValueError(
                f"Dataset '{dataset_name}' is not approved."
            )

        table_name = definition["table_name"]

        sql = f"""
        DESCRIBE {table_name}
        """

        schema = self._execute(
            statement=sql,
            parameters=[],
            row_limit=100,
        )

        cleaned_schema = []
        seen_columns = set()

        for column in schema:
            column_name = column.get("col_name")
            data_type = column.get("data_type")

            if not column_name:
                continue

            if column_name.startswith("#"):
                continue

            if not data_type:
                continue

            normalized_name = column_name.lower()

            if normalized_name in seen_columns:
                continue

            seen_columns.add(normalized_name)
            cleaned_schema.append(column)

        return cleaned_schema
    
    def _get_safe_columns(
    self,
    columns: List[str],
) -> List[str]:
        """
        Return columns that are safe to expose to the agent.
        """

        sensitive_fields = {
            field.lower()
            for field in self.SENSITIVE_FIELDS
        }

        return [
            column
            for column in columns
            if column.lower() not in sensitive_fields
        ]


    def get_safe_dataset_schema(
        self,
        dataset_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Return only non-sensitive columns for an approved dataset.
        """

        schema = self.get_dataset_schema(dataset_name)

        safe_column_names = self._get_safe_columns(
            [
                column.get("col_name")
                for column in schema
                if column.get("col_name")
            ]
        )

        return [
            column
            for column in schema
            if column.get("col_name") in safe_column_names
        ]
    def get_high_value_transactions(
    self,
    minimum_amount: float = 100000,
    limit: int = 50,
     ) -> List[Dict[str, Any]]:
        """
        Retrieve approved high-value transaction evidence.
        """

        if limit < 1 or limit > 100:
            raise ValueError(
                "limit must be between 1 and 100."
            )

        if minimum_amount < 0:
            raise ValueError(
                "minimum_amount cannot be negative."
            )

        sql = f"""
        SELECT
            alert_id,
            alert_type,
            alert_timestamp,
            transaction_id,
            customer_id,
            transaction_amount,
            transaction_limit,
            currency,
            merchant_name,
            merchant_category,
            transaction_type,
            payment_channel,
            city,
            country,
            is_international,
            transaction_timestamp,
            status
        FROM high_value_transactions_alert
        WHERE transaction_amount > :minimum_amount
        ORDER BY transaction_amount DESC
        LIMIT {limit}
        """

        parameters = [
            StatementParameterListItem(
                name="minimum_amount",
                value=str(minimum_amount),
                type="DOUBLE",
            ),
        ]

        return self._execute(
            statement=sql,
            parameters=parameters,
            row_limit=limit,
        )
        
    def get_dataset_rows(
    self,
    dataset_name: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
        """
        Retrieve rows from an approved Fraud Gold dataset.

        The caller provides only the logical dataset name.
        The physical table name comes exclusively from the
        trusted registry.

        Only non-sensitive columns are selected.
        """

        if limit < 1 or limit > 100:
            raise ValueError(
                "limit must be between 1 and 100."
            )

        definition = self.registry.get(dataset_name)

        if definition is None:
            raise ValueError(
                f"Dataset '{dataset_name}' is not approved."
            )

        table_name = definition["table_name"]

        schema = self.get_dataset_schema(dataset_name)

        column_names = [
            column["col_name"]
            for column in schema
            if column.get("col_name")
        ]

        safe_columns = self._get_safe_columns(
            column_names
        )

        if not safe_columns:
            raise ValueError(
                f"Dataset '{dataset_name}' has no safe columns."
            )

        select_columns = ", ".join(
            safe_columns
        )

        sql = f"""
        SELECT
            {select_columns}
        FROM {table_name}
        LIMIT {limit}
        """

        return self._execute(
            statement=sql,
            parameters=[],
            row_limit=limit,
        )

    def get_fraud_card_alerts(
    self,
    limit: int = 50,
) -> List[Dict[str, Any]]:
        """
        Retrieve approved fraud card alert evidence.

        Sensitive PAN and customer PII are intentionally excluded.
        """

        if limit < 1 or limit > 100:
            raise ValueError(
                "limit must be between 1 and 100."
            )

        sql = f"""
        SELECT
            alert_id,
            alert_type,
            alert_timestamp,
            transaction_id,
            customer_id,
            amount,
            currency,
            merchant_id,
            merchant_name,
            merchant_category,
            transaction_type,
            payment_channel,
            device_id,
            transaction_city,
            transaction_country,
            transaction_timestamp,
            is_international
        FROM fraud_card_alert
        ORDER BY alert_timestamp DESC
        LIMIT {limit}
        """

        return self._execute(
            statement=sql,
            parameters=[],
            row_limit=limit,
        )
    def get_transaction_velocity(
    self,
    limit: int = 50,
) -> List[Dict[str, Any]]:
        """
        Retrieve recent transaction velocity evidence.
        """

        if limit < 1 or limit > 100:
            raise ValueError(
                "limit must be between 1 and 100."
            )

        sql = f"""
        SELECT
            window_start,
            window_end,
            transaction_count
        FROM  transaciton_count_by_minute
        ORDER BY window_start DESC
        LIMIT {limit}
        """

        return self._execute(
            statement=sql,
            parameters=[],
            row_limit=limit,
        )

