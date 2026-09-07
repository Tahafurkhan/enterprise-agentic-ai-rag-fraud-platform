"""
Governed LangChain tools for Fraud Agent execution.

These tools expose approved analytical operations rather than arbitrary SQL.
"""

from typing import Any, Dict, List

from langchain_core.tools import StructuredTool

from ...data_access.fraud_data import FraudDataAccess


def build_fraud_tools(
    data_access: FraudDataAccess,
) -> List[StructuredTool]:
    """
    Build the bounded Fraud Agent toolset.
    """

    def get_high_value_transactions(
        minimum_amount: float = 100000,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        return data_access.get_high_value_transactions(
            minimum_amount=minimum_amount,
            limit=limit,
        )

    def get_fraud_card_alerts(
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        return data_access.get_fraud_card_alerts(
            limit=limit,
        )

    def get_transaction_velocity(
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        return data_access.get_transaction_velocity(
            limit=limit,
        )

    def list_approved_datasets() -> List[Dict[str, Any]]:
        """
        List datasets explicitly approved for agent access.
        """

        return [
            {
                "dataset_name": dataset_name,
                "description": definition["description"],
            }
            for dataset_name, definition in data_access.registry.items()
        ]

    def get_safe_dataset_schema(
        dataset_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the non-sensitive schema of an approved dataset.
        """

        return data_access.get_safe_dataset_schema(
            dataset_name=dataset_name,
        )

    def get_dataset_rows(
        dataset_name: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve governed rows from an approved dataset.

        The dataset must be explicitly approved in the registry.
        Sensitive fields are removed by the governed data layer.
        """

        return data_access.get_dataset_rows(
            dataset_name=dataset_name,
            limit=limit,
        )

    return [
        StructuredTool.from_function(
            func=get_high_value_transactions,
            name="get_high_value_transactions",
            description=(
                "Retrieve approved high-value fraud transaction evidence. "
                "The governed data layer applies the approved high-value "
                "threshold and excludes sensitive customer fields."
            ),
        ),

        StructuredTool.from_function(
            func=get_fraud_card_alerts,
            name="get_fraud_card_alerts",
            description=(
                "Retrieve approved fraud card alert evidence. "
                "Sensitive PAN and customer email fields are excluded."
            ),
        ),

        StructuredTool.from_function(
            func=get_transaction_velocity,
            name="get_transaction_velocity",
            description=(
                "Retrieve approved transaction velocity evidence for "
                "fraud analysis."
            ),
        ),

        StructuredTool.from_function(
            func=list_approved_datasets,
            name="list_approved_datasets",
            description=(
                "List the analytical datasets explicitly approved "
                "for Fraud Agent access."
            ),
        ),

        StructuredTool.from_function(
            func=get_safe_dataset_schema,
            name="get_safe_dataset_schema",
            description=(
                "Retrieve the safe, non-sensitive schema of an "
                "approved Fraud Gold dataset."
            ),
        ),

        StructuredTool.from_function(
            func=get_dataset_rows,
            name="get_dataset_rows",
            description=(
                "Retrieve governed analytical rows from an approved "
                "Fraud Gold dataset. The dataset must be approved in "
                "the registry and sensitive fields are excluded."
            ),
        ),
    ]