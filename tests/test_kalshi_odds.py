import datetime as dt
import importlib

import pytest


@pytest.fixture
def kalshi():
    return importlib.import_module('kalshi_odds')


def market(name, price, **extra):
    return dict(event_ticker='KXUFCFIGHT-26SEP12AB', ticker='KXUFCFIGHT-26SEP12AB-' + name,
                yes_sub_title=name, yes_ask_dollars=price, yes_ask_size_fp='10.00',
                market_type='binary', notional_value_dollars='1.0000', status='active', result='', **extra)


def test_matches_both_fighters_and_keeps_corner_orientation(kalshi):
    quotes = kalshi.match_quotes([market('Rong Zhu', '.60'), market('Rafa García', '.42')],
                                 [('Rafa Garcia', 'Rongzhu')], dt.date(2026, 9, 12))
    assert quotes[('rafa garcia', 'rongzhu')][0]['price'] == .42
    assert quotes[('rafa garcia', 'rongzhu')][1]['price'] == .60
    assert quotes[('rafa garcia', 'rongzhu')][0]['odds'] == pytest.approx(138.095238)


@pytest.mark.parametrize('change', [dict(status='settled'), dict(yes_ask_dollars='0'),
    dict(yes_ask_dollars='1'), dict(yes_ask_dollars='NaN'), dict(yes_ask_size_fp='0'),
    dict(event_ticker='KXUFCFIGHT-26SEP19AB'), dict(market_type='scalar'), dict(result='yes')])
def test_untradable_or_wrong_date_quotes_are_unavailable(kalshi, change):
    a = market('A', '.60'); a.update(change)
    assert kalshi.match_quotes([a, market('B', '.41')], [('A','B')], dt.date(2026,9,12)) == {}


def test_does_not_pair_fighters_from_different_markets(kalshi):
    b = market('B', '.41'); b['event_ticker'] = 'KXUFCFIGHT-26SEP12BC'
    assert kalshi.match_quotes([market('A','.60'), b], [('A','B')], dt.date(2026,9,12)) == {}


def test_ambiguous_matching_groups_are_unavailable(kalshi):
    originals = [market('A','.60'), market('B','.41')]
    duplicates = [dict(m, event_ticker='KXUFCFIGHT-26SEP12AB2') for m in originals]
    assert kalshi.match_quotes(originals + duplicates, [('A','B')], dt.date(2026,9,12)) == {}


def test_fetch_follows_pagination_and_uses_asks_not_last_trade(kalshi, monkeypatch):
    from types import SimpleNamespace
    calls=[]
    def get(url, params, timeout):
        calls.append(params.copy())
        payload = {'markets':[market('A','.60')], 'cursor':'next'} if len(calls)==1 else {
            'markets':[market('B','.41')], 'cursor':''}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    monkeypatch.setattr(kalshi.requests, 'get', get)
    quotes=kalshi.fetch_quotes([('A','B')], dt.date(2026,9,12))
    assert quotes[('a','b')][0]['price'] == .60
    assert calls[1]['cursor']=='next' and calls[0]['series_ticker']=='KXUFCFIGHT'
