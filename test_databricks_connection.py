from databricks.sdk import WorkspaceClient

from backend.config import (
    DATABRICKS_HOST,
    DATABRICKS_TOKEN,
    DATABRICKS_SQL_WAREHOUSE_ID,
)

print("Host:", DATABRICKS_HOST)
print("Warehouse ID:", DATABRICKS_SQL_WAREHOUSE_ID)

client = WorkspaceClient(
    host=DATABRICKS_HOST,
    token=DATABRICKS_TOKEN,
)

print("\nTesting workspace connection...")

try:
    me = client.current_user.me()
    print("Authenticated as:", me.user_name)
except Exception as e:
    print("AUTH ERROR:")
    print(type(e).__name__, e)

print("\nTesting SQL warehouse...")

try:
    warehouse = client.warehouses.get(
        DATABRICKS_SQL_WAREHOUSE_ID
    )

    print("Warehouse name:", warehouse.name)
    print("Warehouse state:", warehouse.state)
    print("Warehouse ID:", warehouse.id)

except Exception as e:
    print("WAREHOUSE ERROR:")
    print(type(e).__name__, e)

print("\nTesting SELECT 1...")

try:
    response = client.statement_execution.execute_statement(
        statement="SELECT 1",
        warehouse_id=DATABRICKS_SQL_WAREHOUSE_ID,
        wait_timeout="30s",
    )

    print("SQL SUCCESS:")
    print(response)

except Exception as e:
    print("SQL ERROR:")
    print(type(e).__name__, e)