"""nDPI L7_PROTO side-channel using nfstream.

Runs an nfstream NFStreamer in a daemon thread on the same interface that
Scapy is sniffing on. Every flow nfstream completes is converted to a
5-tuple key plus the numeric nDPI app_protocol id; NFv3FlowExporter can
then look up the L7 id when it closes its own copy of the flow.

We intentionally don't reuse nfstream as the full exporter — it lacks the
TTL, retransmission, packet-size histogram, ICMP/DNS/FTP and TCP_WIN_MAX
fields the training stats are tuned to. This module only fills the
L7_PROTO slot that the port-table fallback was approximating.

Notes:
  * nfstream's `application_name` is a string ("TLS.Google",
    "HTTP.Apple", "Unknown"). The master protocol — the part before the
    first dot — maps to nDPI's numeric id. We carry a curated dictionary
    of ~80 common protocols. Anything missing falls back to 0 (UNKNOWN),
    and the caller can decide whether to use the port-table heuristic
    instead.
  * nfstream emits a flow once it expires. Active/idle timeouts are
    configured to match NFv3FlowExporter so the two pipelines close
    flows at roughly the same time.
  * The resolver discards entries older than 10 minutes so memory does
    not grow unbounded on long-running captures.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

# nDPI master-protocol name → numeric id (ndpi_protocol_ids.h).
# Sarhan NF-v3 datasets carry these numeric codes; matching means our
# L7_PROTO values land in the same z-score bucket the training stats
# expect (L7_PROTO mean=23.5, std=21.2 — broad distribution).
_NDPI_NAME_TO_ID = {
    "Unknown": 0,
    "FTP_CONTROL": 1, "FTP_DATA": 86, "FTP": 1,
    "MAIL_POP": 2, "POP3": 2, "POPS": 4,  # POPS shares mapping (no separate code)
    "MAIL_SMTP": 3, "SMTP": 3, "SMTPS": 3,
    "MAIL_IMAP": 4, "IMAP": 4, "IMAPS": 4,
    "DNS": 5, "MDNS": 8, "LLMNR": 30,
    "IPP": 6,
    "HTTP": 7, "HTTP_Connect": 7, "HTTP_Proxy": 7,
    "NTP": 9,
    "NetBIOS": 10,
    "NFS": 11,
    "RTSP": 13,
    "SSDP": 138,
    "BGP": 65,
    "SNMP": 10,
    "TFTP": 73,
    "DHCP": 6,
    "Syslog": 113,
    "DHCPV6": 103,
    "OpenVPN": 110,
    "WireGuard": 248,
    "GRE": 187,
    "ICMP": 81,
    "IGMP": 82,
    "ICMPV6": 102,
    "PPTP": 92,
    "Telnet": 23,
    "STUN": 41,
    "TLS": 91, "SSL": 91,
    "SSH": 92,
    "IRC": 12,
    "RDP": 88,
    "VNC": 87,
    "SIP": 100,
    "Skype": 125, "SkypeCall": 125,
    "SOCKS": 90,
    "Tor": 163,
    "BitTorrent": 37,
    "QUIC": 188,
    "Kerberos": 132,
    "LDAP": 84,
    "MsSQL-TDS": 18,
    "MySQL": 19,
    "Postgres": 17,
    "Oracle": 117,
    "MongoDB": 226,
    "Redis": 224,
    "MQTT": 222,
    "AMQP": 225,
    "AJP": 142,
    "RX": 184,
    "Modbus": 73,
    "SOMEIP": 235,
    "DCE_RPC": 121,
    "RPC": 121,
    "MS_RPCH": 122,
    "Apple": 140,
    "AppleiCloud": 89,
    "AppleiTunes": 140,
    "ApplePush": 89,
    "AppleStore": 140,
    "Google": 126,
    "GoogleServices": 126,
    "GoogleHangoutDuo": 226,
    "GMail": 121,
    "YouTube": 124,
    "GoogleMaps": 26,
    "GoogleDocs": 217,
    "Facebook": 119,
    "FacebookMessenger": 132,
    "Instagram": 211,
    "WhatsApp": 142,
    "WhatsAppCall": 142,
    "WhatsAppFiles": 142,
    "Telegram": 185,
    "Snapchat": 200,
    "Twitter": 120,
    "TwitchTV": 195,
    "Twitch": 195,
    "Spotify": 156,
    "Microsoft": 212,
    "Microsoft365": 217,
    "MicrosoftAzure": 212,
    "MS_OneDrive": 213,
    "AmazonAWS": 178,
    "AmazonVideo": 209,
    "Netflix": 133,
    "GitHub": 220,
    "Crashlytics": 134,
    "Cloudflare": 244,
    "Dropbox": 121,
}


def _parse_nfstream_app(name: str) -> int:
    """Turn nfstream's "Master.Sub" string into an nDPI numeric id.

    nfstream returns "Unknown" when nDPI couldn't classify the flow yet
    (typically because the flow ended in fewer than 3-10 packets). We
    return 0 in that case so the caller can decide whether to use the
    port-table fallback.
    """
    if not name:
        return 0
    master = name.split(".", 1)[0]
    return _NDPI_NAME_TO_ID.get(master, 0)


class NDPIResolver:
    """5-tuple → nDPI numeric L7 protocol id cache, populated by nfstream.

    Thread-safe. Lookup is direction-agnostic: a flow registered with
    src=A,dst=B can be looked up either way around.
    """

    def __init__(
        self,
        interface: str = "any",
        max_age_sec: float = 600.0,
        active_timeout_sec: int = 120,
        idle_timeout_sec: int = 15,
    ):
        self.interface = interface
        self.max_age_sec = max_age_sec
        self.active_timeout_sec = active_timeout_sec
        self.idle_timeout_sec = idle_timeout_sec
        self._cache: dict = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Spawn the nfstream sniffer thread. Returns True on success."""
        if self._started:
            return True
        try:
            from nfstream import NFStreamer  # noqa: F401
        except ImportError:
            print("[ndpi] nfstream not installed — L7_PROTO falls back to port table.")
            return False

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="ndpi-resolver", daemon=True
        )
        self._thread.start()
        self._started = True
        print(f"[ndpi] resolver started on interface={self.interface}")
        return True

    def stop(self, join_timeout: float = 3.0) -> None:
        if not self._started:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
        self._started = False

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(
        self,
        src_ip: str,
        src_port: int,
        dst_ip: str,
        dst_port: int,
        protocol: int,
    ) -> Optional[int]:
        """Return cached nDPI id for the flow, or None if not yet seen.

        Tries both directions of the 5-tuple — caller doesn't have to
        know which side nfstream registered.
        """
        keys = (
            (src_ip, src_port, dst_ip, dst_port, protocol),
            (dst_ip, dst_port, src_ip, src_port, protocol),
        )
        with self._lock:
            for k in keys:
                entry = self._cache.get(k)
                if entry is not None:
                    return entry[0]
        return None

    # ------------------------------------------------------------------
    # Background sniffer
    # ------------------------------------------------------------------

    def _run(self) -> None:
        from nfstream import NFStreamer
        try:
            streamer = NFStreamer(
                source=self.interface,
                active_timeout=self.active_timeout_sec,
                idle_timeout=self.idle_timeout_sec,
                # We only need transport-layer + L7 metadata.
                statistical_analysis=False,
                splt_analysis=0,
                # Stay lightweight — we are the side-channel, not the exporter.
                accounting_mode=1,  # Layer 3 (IP) packet size — matches our exporter
                n_dissections=20,
                decode_tunnels=False,
            )
        except Exception as exc:
            print(f"[ndpi] failed to start NFStreamer: {exc}")
            return

        try:
            for flow in streamer:
                if self._stop.is_set():
                    break
                self._ingest(flow)
        except Exception as exc:
            print(f"[ndpi] sniffer terminated: {exc}")

    def _ingest(self, flow) -> None:
        app_id = _parse_nfstream_app(flow.application_name)
        if app_id == 0 and flow.application_is_guessed:
            # nfstream guessed by port — don't pollute the cache with that;
            # the caller's own port-table fallback will produce the same thing.
            return
        key = (
            flow.src_ip,
            flow.src_port,
            flow.dst_ip,
            flow.dst_port,
            flow.protocol,
        )
        now = time.time()
        with self._lock:
            self._cache[key] = (app_id, now)
            if len(self._cache) > 2048:
                self._evict(now)

    def _evict(self, now: float) -> None:
        cutoff = now - self.max_age_sec
        stale = [k for k, (_, ts) in self._cache.items() if ts < cutoff]
        for k in stale:
            self._cache.pop(k, None)
