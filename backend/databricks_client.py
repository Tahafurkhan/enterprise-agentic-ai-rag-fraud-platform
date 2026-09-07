from io import BytesIO
from pathlib import PurePosixPath

from config import DATABRICKS_HOST, DATABRICKS_TOKEN, VOLUME_PATH
from databricks.sdk import WorkspaceClient


class DatabricksVolumeClient:
    def __init__(self):

        self.client = WorkspaceClient(host=DATABRICKS_HOST, token=DATABRICKS_TOKEN)

    def upload_file(self, file_name: str, file_content: bytes):

        target_path = str(PurePosixPath(VOLUME_PATH) / file_name)

        # Convert bytes into a file-like object.
        # Databricks SDK requires an object that supports
        # seekable(), read(), etc.
        file_stream = BytesIO(file_content)

        self.client.files.upload(target_path, file_stream, overwrite=True)

        return target_path

    def list_files(self):

        files = []

        for item in self.client.files.list_directory_contents(VOLUME_PATH):
            if not item.is_directory:
                files.append(
                    {
                        "name": PurePosixPath(item.path).name,
                        "path": item.path,
                        "size": item.file_size or 0,
                        "modified_time": item.modification_time,
                    }
                )

        return files
