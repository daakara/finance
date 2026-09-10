"""FastAPI Router for Unified CQRS Cockpit Read Model (Horizon 14.1).

Guarantees:
1. Zero Authentication: ARX has no user logins, accounts, sessions, or auth tokens.
   Profile identifiers (X-Profile-Id, X-User-Id, profile_id, subject_id) are local
   record selectors, not proofs of identity. Default selector is 'default'.
   Never returns 401 Unauthorized.
2. Explicit UNAVAILABLE state when a profile selector has no persisted records.
3. Zero fabricated default scores (no fake 70/75/60/65 triad scores, no fake 88/70
   confidences, no fake 0.85/0.5 conviction ratios, no fake 3-year burn projections).
4. Strict private non-shared cache policy (Cache-Control: private, no-cache, no-store).
5. Authentic runway semantics: distinguishes unrecorded expenditure (None -> EXPENDITURE_UNRECORDED)
   from recorded zero expenditure (0.0 -> ZERO_EXPENDITURE, unencumbered reserves).
6. Separates response generation timestamp (generatedAt) from telemetry observation/update timestamp.
"""

import re
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Response, Header, Query, HTTPException, status
from pydantic import BaseModel, Field
from analyst_dashboard.data.db_engine import HistoryDatabaseEngine

logger = logging.getLogger("api.cockpit")
router = APIRouter()
history_db = HistoryDatabaseEngine()


class UserProfilePayload(BaseModel):
    name: Optional[str] = Field(None, description="Profile full or preferred name")
    role: Optional[str] = Field(None, description="Declared identity / professional role")
    lhi: Optional[float] = Field(None, ge=0, le=100, description="Life Health Index (0-100)")
    hhi: Optional[float] = Field(None, ge=0, le=100, description="Household Health Index (0-100)")
    iai: Optional[float] = Field(None, ge=0, le=100, description="Identity Alignment Index (0-100)")
    liquidReserves: Optional[float] = Field(None, ge=0, description="Liquid cash and money-market reserves in USD")
    monthlyBurn: Optional[float] = Field(None, ge=0, description="Committed monthly household burn in USD")


class ActionItemPayload(BaseModel):
    id: str = Field(..., description="Action identifier e.g. NBA-01")
    title: str = Field(..., description="Action title")
    domain: str = Field(..., description="Domain e.g. CAREER, HEALTH, CAPITAL, HOUSEHOLD")
    priorityScore: float = Field(..., ge=0, le=100, description="Priority score (0-100)")
    identityContribution: Optional[float] = Field(0.0, ge=0, description="Contribution to IAI")
    isPrimary: Optional[bool] = Field(False, description="Whether this is the primary Next Best Action")
    durationMinutes: Optional[int] = Field(30, ge=5, le=480, description="Estimated duration in minutes")
    energyRequired: Optional[str] = Field("MODERATE", description="Cognitive energy category")
    rationale: Optional[str] = Field(None, description="Strategic rationale for action")
    scheduledTimeWindow: Optional[str] = Field(None, description="Target execution window")


def _resolve_profile_selector(
    x_profile_id: Optional[str] = None,
    x_user_id: Optional[str] = None,
    profile_id: Optional[str] = None,
    subject_id: Optional[str] = None,
) -> str:
    """Resolve and sanitize record selector from headers or query parameters.
    
    ARX has no authentication. Identifiers select persistent records in the local store.
    Defaults to 'default' when no selector is supplied.
    """
    raw_id = (x_profile_id or x_user_id or profile_id or subject_id or "default").strip()
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "", raw_id)
    return cleaned[:64] if cleaned else "default"


@router.get("/state", tags=["Unified Cockpit"])
def get_unified_cockpit_state(
    response: Response,
    x_profile_id: Optional[str] = Header(None, alias="X-Profile-Id"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    profile_id: Optional[str] = Query(None, description="Profile Record Selector"),
    subject_id: Optional[str] = Query(None, description="Subject Selector Alias"),
):
    """Fetch authoritative CQRS Unified Cockpit Read Model for the specified record selector.
    
    Guarantees:
    - Zero authentication: never returns 401 Unauthorized; defaults to 'default' selector.
    - 200 OK with authentic persisted metrics if records exist.
    - 200 OK with explicit UNAVAILABLE state if profile selector has no persisted records.
    - Zero fabricated default numbers (no fake 70/75/65 triad, no fake 88/70 confidence).
    """
    # Enforce private non-shared cache policy on personal state
    response.headers["Cache-Control"] = "private, no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Vary"] = "X-Profile-Id, X-User-Id"

    sel_id = _resolve_profile_selector(x_profile_id, x_user_id, profile_id, subject_id)
    now_iso = datetime.now(timezone.utc).isoformat()

    # Query persistent SQLite database
    try:
        profile = history_db.get_user_profile(sel_id)
        holdings = history_db.get_user_portfolio(sel_id)
        actions = history_db.get_user_actions(sel_id)
    except Exception as e:
        logger.error(f"Database error reading state for selector {sel_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Persistent database error reading cockpit state.",
        )

    # If selector has no records, return explicit UNAVAILABLE state (DO NOT invent a fake person)
    if not profile and not holdings and not actions:
        return {
            "version": "14.1.0-CQRS-API",
            "status": "UNAVAILABLE",
            "available": False,
            "subjectId": sel_id,
            "dataSource": "UNAVAILABLE",
            "detail": f"No persisted records or telemetry found for profile selector '{sel_id}'. Profile uninitialized.",
            "triad": None,
            "portfolio": None,
            "signalQuality": {
                "freshness": "UNAVAILABLE",
                "confidence": None,
                "activeSignalsCount": 0,
                "highConvictionRatio": None,
                "lastTelemetrySync": None,
            },
            "nextBestAction": None,
            "secondaryActions": [],
            "primaryForecast": None,
            "outcomeForecasts": [],
            "activeConstraints": [],
            "generatedAt": now_iso,
        }

    # 1. Health Triad Calculation (Strictly from recorded values; NEVER fabricated defaults)
    triad = None
    if profile and profile.get("lhi") is not None and profile.get("hhi") is not None and profile.get("iai") is not None:
        lhi = float(profile["lhi"])
        hhi = float(profile["hhi"])
        iai = float(profile["iai"])
        composite_resilience = round((lhi * 0.35 + hhi * 0.35 + iai * 0.30), 1)

        if composite_resilience >= 75.0:
            triad_status = "STABLE_COMPOUNDING"
            triad_interp = f"Life is stable (LHI {lhi:.0f}), household is resilient (HHI {hhi:.0f}), and identity alignment is compounding (IAI {iai:.0f})."
        elif composite_resilience >= 50.0:
            triad_status = "AT_RISK"
            triad_interp = f"Moderate pressure detected across domains (Composite resilience: {composite_resilience}). Rebalancing recommended."
        else:
            triad_status = "DEGRADED"
            triad_interp = f"Degraded resilience across core life domains ({composite_resilience}). Risk mitigation required."

        triad = {
            "lhi": lhi,
            "hhi": hhi,
            "iai": iai,
            "compositeResilience": composite_resilience,
            "status": triad_status,
            "interpretation": triad_interp,
            "provenance": "USER_REPORTED",
            "isSystemCalculated": False,
        }

    # 2. Portfolio Summary from authentic holdings
    portfolio_summary = None
    if holdings:
        total_cost = sum(float(h.get("shares", 0)) * float(h.get("entryPrice", 0)) for h in holdings)
        has_unpriced = any(h.get("currentPrice") is None for h in holdings)
        if has_unpriced:
            # Preserve missing current prices honestly: do not substitute entry price
            portfolio_summary = {
                "holdingsCount": len(holdings),
                "totalMarketValue": None,
                "totalCostBasis": round(total_cost, 2),
                "unrealizedPnL": None,
                "isComplete": False,
                "status": "PARTIAL_UNPRICED",
            }
        else:
            total_val = sum(float(h.get("shares", 0)) * float(h["currentPrice"]) for h in holdings)
            portfolio_summary = {
                "holdingsCount": len(holdings),
                "totalMarketValue": round(total_val, 2),
                "totalCostBasis": round(total_cost, 2),
                "unrealizedPnL": round(total_val - total_cost, 2),
                "isComplete": True,
                "status": "AVAILABLE",
            }

    # Resolve profile identity attributes
    if profile:
        user_name = profile.get("name") or sel_id
        target_role = profile.get("role") or "Self-Directed Investor"
        liquid_reserves = profile.get("liquidReserves")
        monthly_burn = profile.get("monthlyBurn")
    else:
        user_name = sel_id
        target_role = "Active Portfolio Operator" if holdings else "Uninitialized Profile"
        liquid_reserves = None
        monthly_burn = None

    # 3. Actions
    primary_action = None
    secondary_actions = []
    for act in actions:
        if act.get("isPrimary") and primary_action is None:
            primary_action = act
        elif len(secondary_actions) < 2:
            secondary_actions.append(act)

    if primary_action is None and actions:
        primary_action = actions[0]
        secondary_actions = actions[1:3]

    # 4. Runway & Solvency Semantics
    # Distinguish:
    # - monthly_burn is None -> EXPENDITURE_UNRECORDED
    # - monthly_burn == 0.0 -> ZERO_EXPENDITURE (liquid reserves unencumbered, never display 0.0 mo)
    # - monthly_burn > 0.0 -> CALCULATED
    forecasts = []
    active_constraints = []
    runway_months = None
    runway_status = "UNAVAILABLE"
    runway_explanation = None

    if monthly_burn is None:
        runway_status = "EXPENDITURE_UNRECORDED"
        runway_explanation = "Monthly expenditure has not been recorded in profile; cash runway cannot be calculated."
    elif monthly_burn == 0.0:
        runway_status = "ZERO_EXPENDITURE"
        runway_explanation = "Zero recurring expenditure recorded; liquid reserves are unencumbered by monthly burn."
    elif monthly_burn > 0.0:
        if liquid_reserves is not None:
            runway_months = round(liquid_reserves / monthly_burn, 1)
            runway_status = "CALCULATED"
            runway_explanation = f"Based on ${liquid_reserves:,.2f} liquid reserves and ${monthly_burn:,.2f} monthly burn."
            if runway_months < 6.0:
                active_constraints.append({
                    "id": "C-RUNWAY-01",
                    "type": "CAPITAL",
                    "severity": "CRITICAL",
                    "message": f"Cash runway ({runway_months:.1f} mo) is below the 6.0-month safety threshold.",
                    "enforcementRule": "Capital floor preservation active. Restrict non-essential capital allocations.",
                    "currentUtilization": f"{runway_months:.1f}/6.0 mo",
                })
        else:
            runway_status = "RESERVES_UNRECORDED"
            runway_explanation = "Liquid reserves have not been recorded in profile; cash runway cannot be calculated."

    if runway_status == "CALCULATED" and runway_months is not None:
        forecasts.append({
            "id": f"FC-{sel_id[:8]}",
            "title": "Liquid Capital Runway & Solvency",
            "metric": "Cash Runway Months",
            "currentValue": f"{runway_months:.1f} Months",
            "projectedValue3Yr": None,  # Speculative 3-year projection removed
            "confidencePct": None,      # No fabricated confidence percentage
            "runwayStatus": runway_status,
            "explanation": runway_explanation,
            "primaryDriver": "Liquid Reserve Defense Ratio",
            "riskFactors": ["Unanticipated emergency capital outflow", "Sustained monthly expenditure creep"],
        })
    elif runway_status == "ZERO_EXPENDITURE":
        forecasts.append({
            "id": f"FC-{sel_id[:8]}",
            "title": "Liquid Capital Runway & Solvency",
            "metric": "Cash Runway Months",
            "currentValue": "Unencumbered",
            "projectedValue3Yr": None,
            "confidencePct": None,
            "runwayStatus": runway_status,
            "explanation": runway_explanation,
            "primaryDriver": "Zero Recorded Recurring Outflow",
            "riskFactors": ["Unanticipated emergency capital outflow"],
        })
    elif runway_status == "EXPENDITURE_UNRECORDED":
        forecasts.append({
            "id": f"FC-{sel_id[:8]}",
            "title": "Liquid Capital Runway & Solvency",
            "metric": "Cash Runway Months",
            "currentValue": "Unrecorded Burn",
            "projectedValue3Yr": None,
            "confidencePct": None,
            "runwayStatus": runway_status,
            "explanation": runway_explanation,
            "primaryDriver": "Expenditure Unrecorded in Profile",
            "riskFactors": ["Unquantified recurring living costs"],
        })
    elif runway_status == "RESERVES_UNRECORDED":
        forecasts.append({
            "id": f"FC-{sel_id[:8]}",
            "title": "Liquid Capital Runway & Solvency",
            "metric": "Cash Runway Months",
            "currentValue": "Unrecorded Reserves",
            "projectedValue3Yr": None,
            "confidencePct": None,
            "runwayStatus": runway_status,
            "explanation": runway_explanation,
            "primaryDriver": "Reserves Unrecorded in Profile",
            "riskFactors": ["Unquantified liquid reserve buffer"],
        })

    # Explicit runway read model contract for frontend and store
    runway_obj = None
    if liquid_reserves is not None or monthly_burn is not None:
        runway_shield = (
            "PROTECTED" if (runway_months and runway_months >= 6.0) or runway_status == "ZERO_EXPENDITURE"
            else "CAUTION" if (runway_months and runway_months > 0)
            else "CRITICAL" if runway_status == "RESERVES_UNRECORDED" or (runway_months and runway_months < 3.0)
            else "UNCONFIGURED"
        )
        runway_obj = {
            "monthsUnencumbered": runway_months,
            "liquidReserves": liquid_reserves,
            "burnRateMonthly": monthly_burn,
            "runwayStatus": runway_status,
            "runwayShieldStatus": runway_shield,
            "capitalFloorRule": runway_explanation or "Liquid capital preservation rules active.",
        }

    # 5. Signal Quality from authentic telemetry
    total_telemetry_points = len(holdings) + len(actions)
    # Never substitute request-generation time (now_iso) for telemetry update time
    last_sync = profile.get("updatedAt") if profile else None

    return {
        "version": "14.1.0-CQRS-API",
        "status": "AVAILABLE",
        "available": True,
        "subjectId": sel_id,
        "subjectName": user_name,
        "targetIdentityRole": target_role,
        "dataSource": "PERSISTED_STORE",
        "triad": triad,
        "portfolio": portfolio_summary,
        "runway": runway_obj,
        "signalQuality": {
            "freshness": "PERSISTED_STORE",
            "confidence": None,            # No fabricated 88/70 confidence
            "activeSignalsCount": total_telemetry_points,
            "highConvictionRatio": None,   # No fabricated 0.85/0.5 ratio
            "lastTelemetrySync": last_sync,
        },
        "nextBestAction": primary_action,
        "secondaryActions": secondary_actions,
        "primaryForecast": forecasts[0] if forecasts else None,
        "outcomeForecasts": forecasts,
        "activeConstraints": active_constraints,
        "generatedAt": now_iso,
    }


@router.post("/profile", status_code=status.HTTP_201_CREATED, tags=["Unified Cockpit"])
def update_user_profile(
    payload: UserProfilePayload,
    response: Response,
    x_profile_id: Optional[str] = Header(None, alias="X-Profile-Id"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    profile_id: Optional[str] = Query(None),
    subject_id: Optional[str] = Query(None),
):
    """Create or update profile records in persistent store. Zero auth required."""
    response.headers["Cache-Control"] = "private, no-cache, no-store, must-revalidate"
    sel_id = _resolve_profile_selector(x_profile_id, x_user_id, profile_id, subject_id)

    profile_dict = {
        "name": payload.name or sel_id,
        "role": payload.role or "Investor",
        "lhi": payload.lhi,
        "hhi": payload.hhi,
        "iai": payload.iai,
        "liquidReserves": payload.liquidReserves,
        "monthlyBurn": payload.monthlyBurn,
    }

    try:
        success = history_db.save_user_profile(sel_id, profile_dict)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to persist profile to SQLite store.",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database error persisting profile for {sel_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist profile to SQLite store.",
        )

    return {
        "status": "SUCCESS",
        "message": f"Profile persisted for record selector '{sel_id}'.",
        "profileId": sel_id,
        "userId": sel_id,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/actions", status_code=status.HTTP_201_CREATED, tags=["Unified Cockpit"])
def add_user_action(
    payload: ActionItemPayload,
    response: Response,
    x_profile_id: Optional[str] = Header(None, alias="X-Profile-Id"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    profile_id: Optional[str] = Query(None),
    subject_id: Optional[str] = Query(None),
):
    """Add or update an action item for the specified profile selector."""
    response.headers["Cache-Control"] = "private, no-cache, no-store, must-revalidate"
    sel_id = _resolve_profile_selector(x_profile_id, x_user_id, profile_id, subject_id)

    try:
        success = history_db.save_user_action(sel_id, payload.dict())
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to persist action item to SQLite store.",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database error persisting action item for {sel_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist action item to SQLite store.",
        )

    return {
        "status": "SUCCESS",
        "message": f"Action item '{payload.id}' saved for selector '{sel_id}'.",
        "actionId": payload.id,
    }


@router.post("/action", status_code=status.HTTP_201_CREATED, tags=["Unified Cockpit"], include_in_schema=False)
def add_user_action_singular(
    payload: ActionItemPayload,
    response: Response,
    x_profile_id: Optional[str] = Header(None, alias="X-Profile-Id"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    profile_id: Optional[str] = Query(None),
    subject_id: Optional[str] = Query(None),
):
    """Singular alias for add_user_action."""
    return add_user_action(
        payload=payload,
        response=response,
        x_profile_id=x_profile_id,
        x_user_id=x_user_id,
        profile_id=profile_id,
        subject_id=subject_id,
    )
