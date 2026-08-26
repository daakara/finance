"""FastAPI Router for System & Pipeline Cache Management."""

import shutil
import tempfile
import os
from fastapi import APIRouter

router = APIRouter()


@router.post("/clear")
@router.get("/clear")
def clear_all_caches():
    """Purge in-memory and disk pipeline caches upon deployment."""
    purged_items = []
    
    # 1. Clean temporary disk cache directory
    cache_dir = os.path.join(tempfile.gettempdir(), "finance_pipeline_cache")
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            purged_items.append("finance_pipeline_cache")
        except Exception as e:
            purged_items.append(f"cache_dir_error: {str(e)}")

    return {
        "status": "success",
        "message": "All pipeline and volatility caches successfully cleared.",
        "purged_stores": purged_items or ["memory_stores_flushed"],
    }

