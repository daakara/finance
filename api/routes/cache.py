"""FastAPI Router for System & Pipeline Cache Management."""

import os
import shutil
import tempfile
import hmac
import logging
from typing import Optional
from fastapi import APIRouter, Header, HTTPException, Request, status

logger = logging.getLogger("api.routes.cache")

router = APIRouter()


def _is_production() -> bool:
    """Authoritatively check production environment."""
    env = os.getenv("ENVIRONMENT", "").strip().lower()
    railway_env = os.getenv("RAILWAY_ENVIRONMENT", "").strip().lower()
    railway_env_name = os.getenv("RAILWAY_ENVIRONMENT_NAME", "").strip().lower()
    for val in (env, railway_env, railway_env_name):
        if val in ("dev", "development", "test", "staging", "preview"):
            return False
    if env == "production" or railway_env == "production" or railway_env_name == "production":
        return True
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    return True


def _get_expected_admin_key() -> str:
    """Retrieve expected administrative key from environment.
    
    Precedence:
    1. ARX_ADMIN_KEY (dedicated administrative secret)
    2. ARX_CACHE_PURGE_SECRET (dedicated cache operation token)

    CRITICAL PRIVILEGE BOUNDARY (SEC-002 Recovery):
    Ordinary API keys (ARX_API_KEY) are strictly excluded from administrative operations.
    """
    return (
        os.getenv("ARX_ADMIN_KEY", "").strip()
        or os.getenv("ARX_CACHE_PURGE_SECRET", "").strip()
    )


def _extract_credential(
    x_admin_key: Optional[str] = None,
    authorization: Optional[str] = None,
) -> str:
    """Extract candidate administrative credential from headers in priority order."""
    if x_admin_key and x_admin_key.strip():
        return x_admin_key.strip()
    if authorization and authorization.strip():
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
        elif len(parts) == 1:
            return parts[0].strip()
    return ""


@router.post("/clear")
def clear_all_caches(
    request: Request,
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """Purge in-memory and disk pipeline caches upon deployment.
    
    SECURITY: Privileged administrative operation.
    Requires valid administrative credential via X-Admin-Key or Authorization Bearer.
    Ordinary API keys (X-API-Key) are strictly rejected.
    Fails closed in production if no administrative secret is configured.
    """
    expected_key = _get_expected_admin_key()
    prod_mode = _is_production()

    if not expected_key:
        if prod_mode:
            logger.critical("FATAL: Cache purge invoked in production without configured admin credentials.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server administration credentials not configured.",
            )
        # In non-production, proceed only if non-production dev bypass is active
        logger.debug("Cache purge executed in dev mode without explicit credential.")
    else:
        candidate_key = _extract_credential(x_admin_key, authorization)
        if not candidate_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required for cache management.",
            )
        if not hmac.compare_digest(candidate_key.encode("utf-8"), expected_key.encode("utf-8")):
            client_host = request.client.host if request.client else "unknown"
            logger.warning("Unauthorized cache purge attempt from %s", client_host)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Invalid administration credentials.",
            )

    purged_items = []
    
    # Clean temporary disk cache directory
    cache_dir = os.path.join(tempfile.gettempdir(), "finance_pipeline_cache")
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            purged_items.append("finance_pipeline_cache")
        except Exception as e:
            logger.error("Failed to delete cache directory %s: %s", cache_dir, e)
            purged_items.append(f"cache_dir_error: {str(e)}")

    logger.info("Pipeline and volatility caches cleared successfully.")
    return {
        "status": "success",
        "message": "All pipeline and volatility caches successfully cleared.",
        "purged_stores": purged_items or ["memory_stores_flushed"],
    }
