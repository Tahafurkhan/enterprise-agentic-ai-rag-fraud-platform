from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise RuntimeError(
        "Missing NEO4J_URI, NEO4J_USERNAME, or NEO4J_PASSWORD in .env"
    )


CONSTRAINTS = [
    """
    CREATE CONSTRAINT company_name_unique IF NOT EXISTS
    FOR (n:Company)
    REQUIRE n.name IS UNIQUE
    """,
    """
    CREATE CONSTRAINT department_name_unique IF NOT EXISTS
    FOR (n:Department)
    REQUIRE n.name IS UNIQUE
    """,
    """
    CREATE CONSTRAINT employee_id_unique IF NOT EXISTS
    FOR (n:Employee)
    REQUIRE n.employee_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT policy_id_unique IF NOT EXISTS
    FOR (n:Policy)
    REQUIRE n.policy_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT procedure_id_unique IF NOT EXISTS
    FOR (n:Procedure)
    REQUIRE n.procedure_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT process_id_unique IF NOT EXISTS
    FOR (n:Process)
    REQUIRE n.process_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT control_id_unique IF NOT EXISTS
    FOR (n:Control)
    REQUIRE n.control_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT document_id_unique IF NOT EXISTS
    FOR (n:Document)
    REQUIRE n.document_id IS UNIQUE
    """,
]


SEED_QUERIES = [
    # Company
    """
    MERGE (c:Company {name: "Acme Corporation"})
    SET c.description = "Enterprise organization used for company knowledge RAG testing"
    """,

    # Departments
    """
    MERGE (d:Department {name: "Finance"})
    SET d.description = "Responsible for financial operations and expense management"

    MERGE (f:Department {name: "Fraud"})
    SET f.description = "Responsible for fraud monitoring, investigation, and prevention"

    MERGE (c:Company {name: "Acme Corporation"})
    MERGE (c)-[:HAS_DEPARTMENT]->(d)
    MERGE (c)-[:HAS_DEPARTMENT]->(f)
    """,

    # Employees
    """
    MERGE (e1:Employee {employee_id: "EMP-001"})
    SET e1.name = "Finance Manager",
        e1.title = "Finance Manager"

    MERGE (e2:Employee {employee_id: "EMP-002"})
    SET e2.name = "Fraud Investigator",
        e2.title = "Fraud Investigator"

    MERGE (finance:Department {name: "Finance"})
    MERGE (fraud:Department {name: "Fraud"})

    MERGE (finance)-[:HAS_EMPLOYEE]->(e1)
    MERGE (fraud)-[:HAS_EMPLOYEE]->(e2)
    """,

    # Policies
    """
    MERGE (p:Policy {policy_id: "POL-EXP-001"})
    SET p.name = "Expense Policy",
        p.description = "Defines requirements for employee expense submission and approval"

    MERGE (c:Company {name: "Acme Corporation"})
    MERGE (c)-[:HAS_POLICY]->(p)
    """,

    """
    MERGE (p:Policy {policy_id: "POL-FRD-001"})
    SET p.name = "Fraud Investigation Policy",
        p.description = "Defines procedures and controls for investigating suspected fraud"

    MERGE (c:Company {name: "Acme Corporation"})
    MERGE (c)-[:HAS_POLICY]->(p)
    """,

    # Procedures
    """
    MERGE (p:Procedure {procedure_id: "PROC-EXP-001"})
    SET p.name = "Expense Approval Procedure",
        p.description = "Procedure for reviewing and approving employee expenses"

    MERGE (policy:Policy {policy_id: "POL-EXP-001"})
    MERGE (policy)-[:HAS_PROCEDURE]->(p)
    """,

    """
    MERGE (p:Procedure {procedure_id: "PROC-FRD-001"})
    SET p.name = "Fraud Investigation Procedure",
        p.description = "Procedure for investigating suspicious transactions and fraud alerts"

    MERGE (policy:Policy {policy_id: "POL-FRD-001"})
    MERGE (policy)-[:HAS_PROCEDURE]->(p)
    """,

    # Processes
    """
    MERGE (p:Process {process_id: "PROC-BIZ-001"})
    SET p.name = "Expense Reimbursement",
        p.description = "Business process for reimbursing approved employee expenses"

    MERGE (finance:Department {name: "Finance"})
    MERGE (finance)-[:OWNS_PROCESS]->(p)
    """,

    """
    MERGE (p:Process {process_id: "PROC-BIZ-002"})
    SET p.name = "Fraud Investigation",
        p.description = "Business process for investigating suspected fraudulent activity"

    MERGE (fraud:Department {name: "Fraud"})
    MERGE (fraud)-[:OWNS_PROCESS]->(p)
    """,

    # Controls
    """
    MERGE (c:Control {control_id: "CTRL-EXP-001"})
    SET c.name = "Manager Approval",
        c.description = "Expenses above the configured threshold require manager approval"

    MERGE (p:Process {process_id: "PROC-BIZ-001"})
    MERGE (p)-[:HAS_CONTROL]->(c)

    MERGE (policy:Policy {policy_id: "POL-EXP-001"})
    MERGE (policy)-[:HAS_CONTROL]->(c)
    """,

    """
    MERGE (c:Control {control_id: "CTRL-FRD-001"})
    SET c.name = "Fraud Alert Review",
        c.description = "Fraud alerts must be reviewed by an authorized fraud investigator"

    MERGE (p:Process {process_id: "PROC-BIZ-002"})
    MERGE (p)-[:HAS_CONTROL]->(c)

    MERGE (policy:Policy {policy_id: "POL-FRD-001"})
    MERGE (policy)-[:HAS_CONTROL]->(c)
    """,

    # Documents
    """
    MERGE (d:Document {document_id: "DOC-EXP-001"})
    SET d.file_name = "Expense Policy.pdf",
        d.document_type = "policy",
        d.description = "Enterprise expense policy source document"

    MERGE (p:Policy {policy_id: "POL-EXP-001"})
    MERGE (d)-[:DOCUMENTS]->(p)
    """,

    """
    MERGE (d:Document {document_id: "DOC-FRD-001"})
    SET d.file_name = "Fraud Investigation Policy.pdf",
        d.document_type = "policy",
        d.description = "Enterprise fraud investigation policy source document"

    MERGE (p:Policy {policy_id: "POL-FRD-001"})
    MERGE (d)-[:DOCUMENTS]->(p)
    """,

    # Procedure/process relationships
    """
    MERGE (procedure:Procedure {procedure_id: "PROC-EXP-001"})
    MERGE (process:Process {process_id: "PROC-BIZ-001"})
    MERGE (procedure)-[:SUPPORTS_PROCESS]->(process)
    """,

    """
    MERGE (procedure:Procedure {procedure_id: "PROC-FRD-001"})
    MERGE (process:Process {process_id: "PROC-BIZ-002"})
    MERGE (procedure)-[:SUPPORTS_PROCESS]->(process)
    """,
]


def main() -> None:
    print("Connecting to Neo4j AuraDB...")
    print(f"URI: {NEO4J_URI}")
    print(f"Database: {NEO4J_DATABASE}")

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )

    try:
        driver.verify_connectivity()
        print("Neo4j connectivity: OK")

        with driver.session(database=NEO4J_DATABASE) as session:
            print("Creating constraints...")
            for query in CONSTRAINTS:
                session.run(query).consume()

            print("Clearing previous demo graph...")
            session.run(
                """
                MATCH (n)
                DETACH DELETE n
                """
            ).consume()

            print("Seeding company knowledge graph...")
            for query in SEED_QUERIES:
                session.run(query).consume()

            result = session.run(
                """
                MATCH (n)
                RETURN count(n) AS node_count
                """
            ).single()

            relationship_result = session.run(
                """
                MATCH ()-[r]->()
                RETURN count(r) AS relationship_count
                """
            ).single()

            print(f"Nodes created: {result['node_count']}")
            print(
                f"Relationships created: "
                f"{relationship_result['relationship_count']}"
            )

    finally:
        driver.close()

    print("Company knowledge graph seeded successfully.")


if __name__ == "__main__":
    main()