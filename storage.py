"""

Blob storage access via managed identity

"""

import io
import os
import tempfile
from functools import lru_cache

# IDs 
_DEFAULT_ACCOUNT_URL = "" # Default account rg-kyudai-deeptech
ACCOUNT_URL = os.environ.get("BLOB_ACCOUNT_URL", _DEFAULT_ACCOUNT_URL).strip() # Blob account URL env var. If empty string -> local-only

# local-only = 1 forces Blob off entirely (pure local dev with no Azure)
if os.environ.get("LOCAL_ONLY", "").strip() in ("1", "true", "True"):
    ACCOUNT_URL = ""
BLOB_ENABLED = bool(ACCOUNT_URL)
_PREFIX = os.environ.get("BLOB_CONTAINER_PREFIX", "").strip()
UPLOADS =  f"{_PREFIX}uploads"
ASSETS = f"{_PREFIX}assets"
RESULTS = f"{_PREFIX}results"

@lru_cache(maxsize=1)
def _svc():
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient
    return BlobServiceClient(ACCOUNT_URL, credential=DefaultAzureCredential())

def upload_bytes(container: str, blob_name: str, data: bytes) -> str:
    if not BLOB_ENABLED:
        return f"(local-only) {container}/{blob_name}"
    _svc().get_blob_client(container, blob_name).upload_blob(data, overwrite=True)
    return  f"{container}/{blob_name}"

def download_to_temp(container: str, blob_name: str, suffix=".mat") -> str:
    """
    
    Blob -> local temp path
    
    """
    data = _svc().get_blob_client(container, blob_name).download_blob().readall()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return tmp.name

def list_blobs(container: str, prefix: str=""):
    if not BLOB_ENABLED:
        return []
    return [b.name for b in _svc().get_container_client(container)
            .list_blobs(name_starts_with=prefix)]

def upload_dataframe_parquet(df, container: str, blob_name: str) -> str:
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=False,
                  coerce_timestamps="us", allow_truncated_timestamps=True)
    return upload_bytes(container, blob_name, buf.getvalue())
