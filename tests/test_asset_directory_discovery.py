"""Offline directory tests; no database, orders or external transport."""
import concurrent.futures
import json
import unittest
from unittest.mock import Mock, patch

from analyst_dashboard.data.asset_discovery import AssetDirectoryDiscovery, DiscoveryUnavailable, _fetch_catalog, _NoRedirect


def asset(symbol='ANT', name='Anthropic, PBC', **extra):
    return dict(symbol=symbol, name=name, exchange='NASDAQ', status='active', **{'class': 'us_equity'}, **extra)


class AssetDiscoveryTests(unittest.TestCase):
    def test_unknown_new_listing_needs_no_price_or_static_catalog_entry(self):
        service = AssetDirectoryDiscovery(loader=lambda: [asset()])
        result = service.search('Anthropic')
        self.assertEqual(result['results'][0]['symbol'], 'ANT')
        self.assertFalse(result['results'][0]['identityVerified'])
        self.assertEqual(result['source'], 'ALPACA_ASSET_DIRECTORY')
        self.assertNotIn('score', result['results'][0])

    def test_catalog_refresh_discovers_next_listing_without_code_edit(self):
        clock = [0]
        load = Mock(side_effect=[[asset()], [asset(), asset('NEWCO', 'New Company')]])
        service = AssetDirectoryDiscovery(loader=load, clock=lambda: clock[0])
        self.assertEqual(service.search('New Company')['results'], [])
        clock[0] = 301
        self.assertEqual(service.search('New Company')['results'][0]['symbol'], 'NEWCO')
        self.assertEqual(load.call_count, 2)

    def test_queries_reuse_one_directory_fetch_and_bound_matches(self):
        load = Mock(return_value=[asset(f'NEW{i}', f'New Company {i}') for i in range(20)])
        service = AssetDirectoryDiscovery(loader=load)
        self.assertEqual(len(service.search('New')['results']), 8)
        self.assertEqual(service.search('NEW12')['results'][0]['symbol'], 'NEW12')
        self.assertEqual(load.call_count, 1)

    def test_name_and_symbol_search_are_case_insensitive(self):
        service = AssetDirectoryDiscovery(loader=lambda: [asset()])
        self.assertEqual(service.search('aNt')['results'][0]['symbol'], 'ANT')
        self.assertEqual(service.search('anthropic')['results'][0]['symbol'], 'ANT')

    def test_bad_queries_never_call_provider(self):
        load = Mock()
        for query in ['', 'A', 'x' * 81, 'bad\x00query']:
            with self.assertRaises(ValueError):
                AssetDirectoryDiscovery(loader=load).search(query)
        load.assert_not_called()

    def test_active_nontradable_assets_are_discoverable_without_execution_permission(self):
        service = AssetDirectoryDiscovery(loader=lambda: [asset(tradable=False)])
        self.assertEqual(service.search('ANT')['results'][0]['symbol'], 'ANT')
        self.assertNotIn('executionEligibility', service.search('ANT')['results'][0])

    def test_inactive_unknown_and_duplicate_assets_are_not_promoted(self):
        inactive = asset('OLD', 'Old Company'); inactive['status'] = 'inactive'
        service = AssetDirectoryDiscovery(loader=lambda: [inactive, {}, None, asset(), asset()])
        self.assertEqual(len(service.search('ANT')['results']), 1)
        self.assertEqual(service.search('Old')['results'], [])

    def test_malformed_directory_and_expired_provider_failure_are_unavailable(self):
        for payload in [{}, ['bad'], None, []]:
            with self.assertRaises(DiscoveryUnavailable):
                AssetDirectoryDiscovery(loader=lambda: payload).search('ANT')
        clock = [0]
        load = Mock(side_effect=[[asset()], RuntimeError('private upstream failure')])
        service = AssetDirectoryDiscovery(loader=load, clock=lambda: clock[0])
        service.search('ANT'); clock[0] = 301
        with self.assertRaisesRegex(DiscoveryUnavailable, '^Asset directory temporarily unavailable$'):
            service.search('ANT')

    def test_parallel_queries_do_not_duplicate_refresh(self):
        load = Mock(return_value=[asset()])
        service = AssetDirectoryDiscovery(loader=load)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(service.search, ['ANT'] * 6))
        self.assertEqual(load.call_count, 1)
        self.assertTrue(all(r['results'][0]['symbol'] == 'ANT' for r in results))

    def test_provider_failure_backoff_prevents_request_storms(self):
        load = Mock(side_effect=RuntimeError('upstream failure'))
        service = AssetDirectoryDiscovery(loader=load, clock=lambda: 0)
        for _ in range(6):
            with self.assertRaises(DiscoveryUnavailable):
                service.search('ANT')
        self.assertEqual(load.call_count, 1)

    def test_queries_during_backoff_do_not_postpone_recovery(self):
        clock = [0]
        load = Mock(side_effect=[RuntimeError('upstream failure'), [asset()]])
        service = AssetDirectoryDiscovery(loader=load, clock=lambda: clock[0])
        for current in [0, 20, 29]:
            clock[0] = current
            with self.assertRaises(DiscoveryUnavailable):
                service.search('ANT')
        clock[0] = 31
        self.assertEqual(service.search('ANT')['results'][0]['symbol'], 'ANT')
        self.assertEqual(load.call_count, 2)

    def test_transport_targets_only_reference_directory_and_disables_redirects(self):
        response = Mock(); response.read.return_value = json.dumps([asset()]).encode()
        opener = Mock(); opener.open.return_value.__enter__ = Mock(return_value=response)
        opener.open.return_value.__exit__ = Mock(return_value=False)
        with patch.dict('os.environ', {'ALPACA_API_KEY_ID': 'test-key', 'ALPACA_API_SECRET_KEY': 'test-secret'}), patch('analyst_dashboard.data.asset_discovery.build_opener', return_value=opener):
            self.assertEqual(_fetch_catalog()[0]['symbol'], 'ANT')
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, 'https://paper-api.alpaca.markets/v2/assets?status=active&asset_class=us_equity')
        self.assertEqual(request.get_method(), 'GET')
        self.assertEqual(opener.open.call_count, 1)
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, '', {}, 'https://elsewhere.example'))

    def test_missing_credentials_make_no_transport_call(self):
        with patch.dict('os.environ', {}, clear=True), patch('analyst_dashboard.data.asset_discovery.build_opener') as opener:
            with self.assertRaises(DiscoveryUnavailable):
                _fetch_catalog()
            opener.assert_not_called()


if __name__ == '__main__':
    unittest.main()
