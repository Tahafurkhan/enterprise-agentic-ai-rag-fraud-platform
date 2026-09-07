"""
Governed registry of Fraud Gold analytical datasets.

Only datasets explicitly registered here are available to the
application and agent layer.
"""

FRAUD_TABLE_REGISTRY = {
    "fraud_card_alert": {
        "table_name": "fraud_card_alert",
        "description": (
            "Fraud card transaction alerts."
        ),
    },

    "high_value_transactions_alert": {
        "table_name": "high_value_transactions_alert",
        "description": (
            "High-value transaction alerts."
        ),
    },

    "transaction_velocity": {
        "table_name": "transaciton_count_by_minute",
        "description": (
            "Transaction counts aggregated by minute."
        ),
    },

    "transaction_velocity_sliding_window": {
        "table_name": "transaciton_count_by_minute_sliding_window",
        "description": (
            "Transaction counts aggregated using a sliding window."
        ),
    },
}