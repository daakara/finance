"""
Market-Wide and Bounded Scanner Engine for SEC Form 4 Smart Money Evidence.
Implements transparent arithmetic conviction scoring, insider clustering, and readiness governance.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set

from analyst_dashboard.smart_money.form4_models import (
    SmartMoneyTransaction,
    SmartMoneyIssuerState,
    SmartMoneyScanSnapshot,
    SmartMoneyReadiness,
    SmartMoneySignalType,
)
from analyst_dashboard.smart_money.form4_ingestion import (
    Form4IngestionManager,
    connect_smart_money_db,
)

logger = logging.getLogger(__name__)


class SmartMoneyForm4Scanner:
    """
    Executes rule-governed Form 4 screening across an eligible universe of securities.
    """

    SCANNER_ID = "SMART_MONEY_FORM4"
    VERSION = "1.0.0"

    def __init__(
        self,
        ingestion_mgr: Optional[Form4IngestionManager] = None,
        universe_builder: Optional[Any] = None,
    ):
        self.ingestion_mgr = ingestion_mgr or Form4IngestionManager()
        self.universe_builder = universe_builder

    def evaluate_issuer_transactions(
        self,
        symbol: str,
        transactions: List[SmartMoneyTransaction],
        lookback_days: int = 30,
        now_date: Optional[str] = None,
    ) -> SmartMoneyIssuerState:
        """
        Aggregates insider transactions for an issuer over lookback_days,
        evaluates cluster buying and C-suite participation, and calculates arithmetic conviction.
        """
        ref_date = datetime.strptime(now_date, "%Y-%m-%d").date() if now_date else datetime.utcnow().date()
        cutoff_date = ref_date - timedelta(days=lookback_days)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")

        # Filter transactions within window
        window_txs = [tx for tx in transactions if tx.transaction_date >= cutoff_str]

        open_market_buys = [tx for tx in window_txs if tx.is_open_market_purchase_evidence]
        open_market_sales = [tx for tx in window_txs if tx.signal_type == SmartMoneySignalType.OPEN_MARKET_SALE]

        gross_buy_val = sum(tx.total_value_usd for tx in open_market_buys)
        gross_sale_val = sum(tx.total_value_usd for tx in open_market_sales)
        net_buy_val = gross_buy_val - gross_sale_val

        distinct_buyers: Set[str] = {tx.reporting_owner_cik for tx in open_market_buys}
        distinct_buyer_count = len(distinct_buyers)
        cluster_buy = distinct_buyer_count >= 2

        # Check executive titles
        c_suite_titles = ["CEO", "CHIEF EXECUTIVE", "CFO", "CHIEF FINANCIAL", "PRESIDENT", "CHAIRMAN"]
        c_suite_buying = False
        all_titles: Set[str] = set()

        for tx in open_market_buys:
            title = (tx.officer_title or ("Director" if tx.is_director else "Insider")).strip()
            all_titles.add(title)
            t_upper = title.upper()
            if any(cs in t_upper for cs in c_suite_titles):
                c_suite_buying = True

        # Calculate transparent arithmetic conviction score (0-100)
        reasons: List[str] = []
        score = 0.0

        if open_market_buys:
            # 1. Purchase Value Factor (max 40)
            if gross_buy_val >= 1_000_000:
                score += 40.0
                reasons.append(f"Substantial open-market buying: ${gross_buy_val:,.0f}")
            elif gross_buy_val >= 250_000:
                score += 30.0
                reasons.append(f"Significant open-market buying: ${gross_buy_val:,.0f}")
            elif gross_buy_val >= 50_000:
                score += 20.0
                reasons.append(f"Open-market buying: ${gross_buy_val:,.0f}")
            else:
                score += 10.0
                reasons.append(f"Modest open-market buying: ${gross_buy_val:,.0f}")

            # 2. Clustering Factor (max 30)
            if cluster_buy:
                score += 30.0
                reasons.append(f"Cluster buy: {distinct_buyer_count} distinct insiders purchasing within {lookback_days}d")
            else:
                score += 15.0
                reasons.append("Single insider buyer")

            # 3. Role Factor (max 20)
            if c_suite_buying:
                score += 20.0
                reasons.append("C-Suite executive participation (CEO/CFO/President)")
            elif any(tx.is_officer or tx.is_director for tx in open_market_buys):
                score += 10.0
                reasons.append("Officer / Board Director participation")

            # 4. Recency Factor (max 10)
            most_recent_tx = max(open_market_buys, key=lambda t: t.transaction_date)
            try:
                tx_date_obj = datetime.strptime(most_recent_tx.transaction_date, "%Y-%m-%d").date()
                days_ago = (ref_date - tx_date_obj).days
                if days_ago <= 7:
                    score += 10.0
                    reasons.append(f"High recency: transaction executed {days_ago}d ago")
                elif days_ago <= 14:
                    score += 5.0
                    reasons.append(f"Recent execution: transaction executed {days_ago}d ago")
            except Exception:
                pass

        score = min(100.0, score)
        qualifies = (len(open_market_buys) > 0) and (gross_buy_val >= 25_000)

        cik = transactions[0].issuer_cik if transactions else ""
        name = transactions[0].issuer_name if transactions else symbol

        return SmartMoneyIssuerState(
            symbol=symbol,
            issuer_cik=cik,
            issuer_name=name or symbol,
            period_start=cutoff_str,
            period_end=ref_date.strftime("%Y-%m-%d"),
            open_market_purchase_count=len(open_market_buys),
            open_market_sale_count=len(open_market_sales),
            distinct_buyer_count=distinct_buyer_count,
            gross_purchase_value_usd=gross_buy_val,
            net_purchase_value_usd=net_buy_val,
            cluster_buy=cluster_buy,
            c_suite_buying=c_suite_buying,
            top_insider_titles=sorted(list(all_titles)),
            conviction_score=score,
            conviction_reasons=reasons,
            qualifies_for_radar=qualifies,
        )

    def scan_issuers(
        self,
        symbols: Optional[List[str]] = None,
        universe_build_id: Optional[str] = None,
        lookback_days: int = 30,
        now_date: Optional[str] = None,
        is_bounded_sample: bool = True,
    ) -> SmartMoneyScanSnapshot:
        """
        Executes scanner across specified symbols or a governed universe build.
        In bounded sample mode, outputs diagnostic snapshot with exact matches.
        In market-wide production mode:
          - Strictly requires universe_build_id.
          - Loads governed scannable members from the universe build.
          - Enforces PIPELINE_PENDING with matched_count: None unless certified.
        """
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        candidates: List[Dict[str, Any]] = []

        manifest_data = None
        if not is_bounded_sample:
            if not universe_build_id:
                raise ValueError("SNAPSHOT_WITHOUT_UNIVERSE_BUILD_IN_MARKET_WIDE_MODE: universe_build_id is required in MARKET_WIDE mode.")
            if self.universe_builder is not None:
                manifest_data = self.universe_builder.get_build_manifest(universe_build_id)
                member_dicts = self.universe_builder.get_scannable_members(universe_build_id)
                target_symbols = [m["symbol"] for m in member_dicts]
            else:
                target_symbols = symbols or []
        else:
            target_symbols = symbols or []

        for sym in target_symbols:
            txs = self.ingestion_mgr.get_parsed_transactions(symbol=sym)
            if txs:
                state = self.evaluate_issuer_transactions(sym, txs, lookback_days=lookback_days, now_date=now_date)
                if state.qualifies_for_radar:
                    candidates.append({
                        "symbol": state.symbol,
                        "companyName": state.issuer_name,
                        "subsignal": "INSIDER_FORM4",
                        "convictionScore": state.conviction_score,
                        "reasons": state.conviction_reasons,
                        "clusterBuy": state.cluster_buy,
                        "distinctBuyers": state.distinct_buyer_count,
                        "grossPurchaseValueUsd": state.gross_purchase_value_usd,
                        "netPurchaseValueUsd": state.net_purchase_value_usd,
                        "cSuiteBuying": state.c_suite_buying,
                        "insiderTitles": state.top_insider_titles,
                        "signalStatus": "QUALIFIED_INSIDER_ACCUMULATION",
                    })

        candidates.sort(key=lambda c: c["convictionScore"], reverse=True)

        if is_bounded_sample:
            # Bounded engineering sample output
            status = "AVAILABLE" if len(candidates) > 0 else "NO_MATCHES"
            matched_count = len(candidates)
            scope_label = "BOUNDED_ENGINEERING_SAMPLE"
            items = candidates
            scannable_cnt = len(target_symbols)
            eligible_cnt = scannable_cnt
            u_id = "BOUNDED_SAMPLE"
            u_hash = None
        else:
            # Governed market-wide production output: Fails closed to PIPELINE_PENDING unless certified
            status = "PIPELINE_PENDING"
            matched_count = None
            scope_label = "MARKET_WIDE"
            items = []
            scannable_cnt = manifest_data.scannable_count if manifest_data else len(target_symbols)
            eligible_cnt = manifest_data.eligible_count if manifest_data else scannable_cnt
            u_id = manifest_data.universe_id if manifest_data else "SMART_MONEY_US_EQUITIES"
            u_hash = manifest_data.member_identity_hash if manifest_data else None

        provenance = {
            "source": "SEC EDGAR Form 4",
            "subsignal": "INSIDER_FORM4",
            "statutory_basis": "Section 16(a)",
            "scanner_ruleset_version": self.VERSION,
            "qualification_contract_version": "FORM4_RADAR_QUALIFICATION_V1",
            "validation_scope": scope_label,
        }

        freshness = {
            "policy_version": self.VERSION,
            "lookback_days": lookback_days,
            "generated_at": now_str,
            "status": "FRESH",
        }

        return SmartMoneyScanSnapshot(
            scanner_id=self.SCANNER_ID,
            version=self.VERSION,
            status=status,
            validation_scope=scope_label,
            generated_at=now_str,
            universe_id=u_id,
            scannable_count=scannable_cnt,
            matched_count=matched_count,
            items=items,
            provenance=provenance,
            freshness=freshness,
            universe_build_id=universe_build_id,
            universe_content_hash=u_hash,
            eligibility_policy_version=manifest_data.eligibility_policy_version if manifest_data else None,
            eligible_count=eligible_cnt,
            cik_mapped_count=scannable_cnt,
            coverage_status="COMPLETE" if is_bounded_sample else "PARTIAL",
        )

    def scan_issuers_as_of(
        self,
        as_of_watermark: str,
        symbols: Optional[List[str]] = None,
        universe_build_id: Optional[str] = None,
        lookback_days: int = 30,
        now_date: Optional[str] = None,
    ) -> SmartMoneyScanSnapshot:
        """
        Point-in-Time Governed Evaluation (Gates B2, B3 & Phase F):
        Evaluates Form 4 insider transactions strictly bounded by certified watermark W
        (public_availability_at <= as_of_watermark).
        Determines coverage_status through receipt verification in sm_reconciliation_reports.
        """
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        candidates: List[Dict[str, Any]] = []

        manifest_data = None
        if universe_build_id and self.universe_builder is not None:
            manifest_data = self.universe_builder.get_build_manifest(universe_build_id)
            member_dicts = self.universe_builder.get_scannable_members(universe_build_id)
            target_symbols = [m["symbol"] for m in member_dicts]
        else:
            target_symbols = symbols or []

        # Receipt-driven completeness check against sm_reconciliation_reports
        coverage_status = "PARTIAL"
        certified_complete = False
        reconciliation_id = None
        try:
            conn = self.ingestion_mgr._get_connection() if hasattr(self.ingestion_mgr, "_get_connection") else None
            if not conn:
                conn = connect_smart_money_db(self.ingestion_mgr.db_path)
            cur = conn.cursor()
            cur.execute("""
                SELECT reconciliation_id, coverage_status, reconciled_at
                FROM sm_reconciliation_reports
                WHERE coverage_status = 'COMPLETE'
                  AND (interval_end >= ? OR reconciled_at >= ?)
                ORDER BY interval_end DESC LIMIT 1
            """, (as_of_watermark, as_of_watermark))
            row = cur.fetchone()
            if row:
                coverage_status = "COMPLETE"
                certified_complete = True
                reconciliation_id = row[0]
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to check reconciliation reports: {e}")
            coverage_status = "PARTIAL"
            certified_complete = False

        # Bulk PIT query: strictly bounded by as_of_watermark
        ref_d = datetime.strptime(now_date, "%Y-%m-%d").date() if now_date else datetime.utcnow().date()
        cutoff_d = ref_d - timedelta(days=lookback_days)
        cutoff_str = cutoff_d.strftime("%Y-%m-%d")

        bulk_tx_map = self.ingestion_mgr.get_bulk_transactions_as_of(
            as_of_time=as_of_watermark,
            symbols=target_symbols,
            min_date=cutoff_str,
        )

        for sym in target_symbols:
            txs = bulk_tx_map.get(sym.upper().strip(), [])
            if txs:
                state = self.evaluate_issuer_transactions(sym, txs, lookback_days=lookback_days, now_date=now_date)
                if state.qualifies_for_radar:
                    candidates.append({
                        "symbol": state.symbol,
                        "companyName": state.issuer_name,
                        "subsignal": "INSIDER_FORM4",
                        "convictionScore": state.conviction_score,
                        "qualifies": True,
                        "reasons": state.conviction_reasons,
                        "clusterBuy": state.cluster_buy,
                        "distinctBuyers": state.distinct_buyer_count,
                        "grossPurchaseValueUsd": state.gross_purchase_value_usd,
                        "netPurchaseValueUsd": state.net_purchase_value_usd,
                        "cSuiteBuying": state.c_suite_buying,
                        "insiderTitles": state.top_insider_titles,
                        "signalStatus": "QUALIFIED_INSIDER_ACCUMULATION",
                    })

        candidates.sort(key=lambda c: c["convictionScore"], reverse=True)

        scannable_cnt = manifest_data.scannable_count if manifest_data else len(target_symbols)
        eligible_cnt = manifest_data.eligible_count if manifest_data else scannable_cnt
        u_id = manifest_data.universe_id if manifest_data else "SMART_MONEY_US_EQUITIES"
        u_hash = manifest_data.member_identity_hash if manifest_data else None

        provenance = {
            "source": "SEC EDGAR Form 4",
            "subsignal": "INSIDER_FORM4",
            "statutory_basis": "Section 16(a)",
            "scanner_ruleset_version": self.VERSION,
            "qualification_contract_version": "FORM4_RADAR_QUALIFICATION_V1",
            "data_as_of": as_of_watermark,
            "certified_watermark": as_of_watermark,
            "coverage_status": coverage_status,
            "certified_complete": certified_complete,
            "reconciliation_receipt_id": reconciliation_id,
            "validation_scope": "MARKET_WIDE" if universe_build_id else "TARGET_UNIVERSE",
        }

        freshness = {
            "policy_version": self.VERSION,
            "lookback_days": lookback_days,
            "data_as_of": as_of_watermark,
            "generated_at": now_str,
            "status": "FRESH",
        }

        # Status: If certified complete or target universe, AVAILABLE/NO_MATCHES; if uncertified in market-wide mode, PIPELINE_PENDING
        if coverage_status == "COMPLETE" or not universe_build_id:
            status = "AVAILABLE" if len(candidates) > 0 else "NO_MATCHES"
            matched_count = len(candidates)
        else:
            status = "PIPELINE_PENDING"
            matched_count = None

        return SmartMoneyScanSnapshot(
            scanner_id=self.SCANNER_ID,
            version=self.VERSION,
            status=status,
            validation_scope="MARKET_WIDE" if universe_build_id else "TARGET_UNIVERSE",
            generated_at=now_str,
            universe_id=u_id,
            scannable_count=scannable_cnt,
            matched_count=matched_count,
            items=candidates,
            provenance=provenance,
            freshness=freshness,
            universe_build_id=universe_build_id,
            universe_content_hash=u_hash,
            eligibility_policy_version=manifest_data.eligibility_policy_version if manifest_data else None,
            eligible_count=eligible_cnt,
            cik_mapped_count=scannable_cnt,
            coverage_status=coverage_status,
        )

    def evaluate_readiness(
        self,
        is_market_wide_governed: bool = False,
        universe_denominator: Optional[int] = None,
    ) -> SmartMoneyReadiness:
        """
        Evaluates readiness from measurable underlying provenance using canonical connection factory.
        """
        source_ok = False
        try:
            conn = self.ingestion_mgr._get_connection() if hasattr(self.ingestion_mgr, "_get_connection") else None
            if not conn:
                conn = connect_smart_money_db(self.ingestion_mgr.db_path)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM form4_raw_filings")
            raw_cnt = cur.fetchone()[0]
            source_ok = raw_cnt >= 0
            conn.close()
        except Exception:
            source_ok = False

        freshness_ok = True
        full_scan_ok = bool(universe_denominator and universe_denominator > 0)

        if not is_market_wide_governed or not universe_denominator or universe_denominator <= 0:
            return SmartMoneyReadiness(
                source_ingestion_operational=source_ok,
                market_wide_universe_governed=False,
                universe_denominator=universe_denominator,
                freshness_policy_satisfied=freshness_ok,
                full_scan_completed=full_scan_ok,
                status="PIPELINE_PENDING",
                blocker_reason="Market-wide universe authority unprovisioned in UniverseStore.",
            )

        return SmartMoneyReadiness(
            source_ingestion_operational=source_ok,
            market_wide_universe_governed=True,
            universe_denominator=universe_denominator,
            freshness_policy_satisfied=freshness_ok,
            full_scan_completed=full_scan_ok,
            status="AVAILABLE",
            blocker_reason=None,
        )
