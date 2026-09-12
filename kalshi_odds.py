"""Read-only Kalshi UFC winner quotes. Prices are executable YES asks, before fees."""
from collections import defaultdict
import math
import unicodedata

import requests

API = 'https://external-api.kalshi.com/trade-api/v2/markets'


def name_key(name):
    text = unicodedata.normalize('NFKD', name).casefold()
    return ''.join(c for c in text if c.isalnum() and not unicodedata.combining(c))


def american_odds(price):
    return 100 * (1 - price) / price if price <= .5 else -100 * price / (1 - price)


def match_quotes(markets, bouts, date):
    groups = defaultdict(list)
    prefix = 'KXUFCFIGHT-' + date.strftime('%y%b%d').upper()
    for market in markets:
        if market.get('event_ticker', '').startswith(prefix):
            groups[market['event_ticker']].append(market)
    candidates = defaultdict(list)
    for group in groups.values():
        if len(group) != 2:
            continue
        quotes = {}
        for market in group:
            try:
                price = float(market['yes_ask_dollars'])
                size = float(market['yes_ask_size_fp'])
                if (market.get('status') != 'active' or market.get('result')
                        or market.get('market_type') != 'binary'
                        or float(market['notional_value_dollars']) != 1
                        or not 0 < price < 1 or not math.isfinite(size) or size <= 0):
                    continue
                key = name_key(market['yes_sub_title'])
                if key in quotes:
                    break
                quotes[key] = dict(price=price, odds=american_odds(price), ticker=market['ticker'])
            except (KeyError, ValueError, TypeError):
                continue
        if len(quotes) == 2:
            candidates[frozenset(quotes)].append(quotes)
    out = {}
    for a, b in bouts:
        matches = candidates.get(frozenset((name_key(a), name_key(b))), [])
        if len(matches) == 1:
            out[(a.lower(), b.lower())] = [matches[0][name_key(a)], matches[0][name_key(b)]]
    return out


def fetch_quotes(bouts, date):
    markets, seen = [], set()
    params = dict(series_ticker='KXUFCFIGHT', status='open', limit=1000)
    for _ in range(20):
        response = requests.get(API, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload.get('markets'), list):
            raise ValueError('Kalshi returned no market list')
        markets.extend(payload['markets'])
        cursor = payload.get('cursor')
        if not cursor:
            return match_quotes(markets, bouts, date)
        if cursor in seen:
            raise ValueError('Kalshi pagination repeated a cursor')
        seen.add(cursor)
        params = dict(params, cursor=cursor)
    raise ValueError('Kalshi pagination exceeded the page limit')
