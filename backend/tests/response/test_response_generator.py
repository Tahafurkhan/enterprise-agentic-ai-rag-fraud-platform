from backend.response.response_generator import (
    ResponseGenerator,
    build_response_generator,
)


def test_response_generator_uses_approved_evidence():
    generator = ResponseGenerator()

    evidence = [
        {
            "source": "fraud_gold",
            "tool": "get_high_value_transactions",
            "data": [
                {
                    "transaction_id": "TX001",
                    "amount": 150000,
                },
                {
                    "transaction_id": "TX002",
                    "amount": 200000,
                },
            ],
        }
    ]

    result = generator.generate(
        query="Show me high value fraud transactions",
        evidence=evidence,
    )

    assert result["answer"]
    assert result["evidence_used"] == evidence
    assert result["claims"]

    assert (
        "2 high-value transaction records"
        in result["claims"][0]["claim"]
    )


def test_response_generator_preserves_evidence_source():
    generator = ResponseGenerator()

    evidence = [
        {
            "source": "fraud_gold",
            "tool": "get_transaction_velocity",
            "data": [
                {
                    "velocity_count": 10,
                }
            ],
        }
    ]

    result = generator.generate(
        query="Show transaction velocity",
        evidence=evidence,
    )

    assert result["claims"][0]["source"] == "fraud_gold"
    assert (
        result["claims"][0]["tool"]
        == "get_transaction_velocity"
    )


def test_response_generator_handles_no_evidence():
    generator = ResponseGenerator()

    result = generator.generate(
        query="Show me fraud transactions",
        evidence=[],
    )

    assert result["evidence_used"] == []
    assert result["claims"] == []

    assert (
        "could not find approved evidence"
        in result["answer"]
    )


def test_response_generator_factory():
    generator = build_response_generator()

    assert isinstance(generator, ResponseGenerator)