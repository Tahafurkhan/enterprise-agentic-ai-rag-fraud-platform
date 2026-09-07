import os

from dotenv import load_dotenv

load_dotenv()


DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")

DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")


VOLUME_PATH = "/Volumes/enterprise_rag/source/rag_docs/incoming"


MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".xml",
    ".txt",
    ".md",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}


DATABRICKS_SQL_WAREHOUSE_ID = os.getenv(
    "DATABRICKS_SQL_WAREHOUSE_ID",
    "",
)

FRAUD_GOLD_CATALOG = os.getenv(
    "FRAUD_GOLD_CATALOG",
    "",
)

FRAUD_GOLD_SCHEMA = os.getenv(
    "FRAUD_GOLD_SCHEMA",
    "",
)