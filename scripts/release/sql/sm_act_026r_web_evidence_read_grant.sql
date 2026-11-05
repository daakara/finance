-- SM-ACT-026R / Production Recovery 002.
-- Execute ONLY after owner-approved production GRANT authorization.
-- Target: PostgreSQL production with the authorized schema-administration principal.
-- The web principal must remain unable to mutate authoritative filing evidence.
BEGIN;
GRANT SELECT ON TABLE public.form4_raw_filings, public.sm_amendments TO smart_money_web;
DO $$
BEGIN
    IF NOT has_table_privilege('smart_money_web', 'public.form4_raw_filings', 'SELECT')
       OR NOT has_table_privilege('smart_money_web', 'public.sm_amendments', 'SELECT') THEN
        RAISE EXCEPTION 'Smart Money read role SELECT qualification failed';
    END IF;
    IF has_table_privilege('smart_money_web', 'public.form4_raw_filings', 'INSERT')
       OR has_table_privilege('smart_money_web', 'public.form4_raw_filings', 'UPDATE')
       OR has_table_privilege('smart_money_web', 'public.form4_raw_filings', 'DELETE')
       OR has_table_privilege('smart_money_web', 'public.sm_amendments', 'INSERT')
       OR has_table_privilege('smart_money_web', 'public.sm_amendments', 'UPDATE')
       OR has_table_privilege('smart_money_web', 'public.sm_amendments', 'DELETE') THEN
        RAISE EXCEPTION 'Unexpected Smart Money web evidence write authority';
    END IF;
END $$;
COMMIT;
-- Follow with a separate, actual web-identity API readback of the NVDA SEC accession.
-- Do not run a new SEC ingestion or toggle any HOLD.

-- ==============================================================================
-- EXPLICIT AUDITABLE ROLLBACK PROCEDURE (Recovery 002 / Requirement 5)
-- Target: PostgreSQL production with the authorized schema-administration principal.
-- Execute if read grant must be revoked or if postflight verification fails.
-- ==============================================================================
-- BEGIN;
-- REVOKE SELECT ON TABLE public.form4_raw_filings, public.sm_amendments FROM smart_money_web;
-- DO $$
-- BEGIN
--     IF has_table_privilege('smart_money_web', 'public.form4_raw_filings', 'SELECT')
--        OR has_table_privilege('smart_money_web', 'public.sm_amendments', 'SELECT') THEN
--         RAISE EXCEPTION 'Smart Money read role REVOKE qualification failed: SELECT privilege remains active';
--     END IF;
-- END $$;
-- COMMIT;
