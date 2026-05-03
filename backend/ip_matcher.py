"""
IPRuleMatcher
=============

Engine-side matcher that decides whether a source IP belongs to the
whitelist or blacklist published by the API into Redis.

Reads two keys (JSON list of {id, cidr}):
    ip_rules:whitelist
    ip_rules:blacklist

Refreshes from Redis on construction, then at most every REFRESH_INTERVAL
seconds (called via maybe_refresh() from the hot path). The API also
re-publishes the keys on every CRUD, so changes propagate within seconds.

On a conflict (IP in both lists) **blacklist wins** — safer default.
"""

import ipaddress
import json
import logging
import time

logger = logging.getLogger(__name__)

WHITELIST_KEY = "ip_rules:whitelist"
BLACKLIST_KEY = "ip_rules:blacklist"


class IPRuleMatcher:
    REFRESH_INTERVAL = 30.0  # seconds

    def __init__(self, redis_client):
        self.redis = redis_client
        self._whitelist: list[tuple[int, ipaddress._BaseNetwork]] = []
        self._blacklist: list[tuple[int, ipaddress._BaseNetwork]] = []
        self._last_refresh: float = 0.0

    # --- Cache loading -------------------------------------------------------

    def _load_key(self, key: str) -> list[tuple[int, ipaddress._BaseNetwork]]:
        try:
            raw = self.redis.get(key)
        except Exception as e:
            logger.warning(f"[IPRuleMatcher] Redis GET {key} failed: {e}")
            return []

        if not raw:
            return []

        try:
            entries = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"[IPRuleMatcher] Bad JSON in {key}: {e}")
            return []

        out: list[tuple[int, ipaddress._BaseNetwork]] = []
        for entry in entries:
            try:
                rid = int(entry["id"])
                net = ipaddress.ip_network(entry["cidr"], strict=False)
                out.append((rid, net))
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"[IPRuleMatcher] Skipping bad entry {entry!r}: {e}")
        return out

    def refresh(self) -> None:
        self._whitelist = self._load_key(WHITELIST_KEY)
        self._blacklist = self._load_key(BLACKLIST_KEY)
        self._last_refresh = time.time()

    def maybe_refresh(self) -> None:
        if time.time() - self._last_refresh > self.REFRESH_INTERVAL:
            self.refresh()

    # --- Matching ------------------------------------------------------------

    def match(self, src_ip: str) -> tuple[str | None, int | None]:
        """Return ('blacklist'|'whitelist'|None, rule_id|None) for src_ip.

        Blacklist takes precedence if the same IP matches both lists.
        """
        try:
            ip = ipaddress.ip_address(src_ip)
        except (ValueError, TypeError):
            return (None, None)

        for rid, net in self._blacklist:
            if ip in net:
                return ("blacklist", rid)
        for rid, net in self._whitelist:
            if ip in net:
                return ("whitelist", rid)
        return (None, None)
