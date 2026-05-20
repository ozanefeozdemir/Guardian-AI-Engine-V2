"""NF-v3 flow exporter for live packet capture.

Emits a raw NF-v3 row (dict) per closed flow, matching the column schema in
src.data.features.NF_V3_RAW_FEATURES. FlowGuardProvider consumes the dict
and runs canavar-model's training preprocess on it; no preprocessing happens
here.

Decisions encoded in this implementation:
  * Identity/timestamp columns (IPV4_*, L4_*, FLOW_START/END_MILLISECONDS) are
    populated even though preprocess drops them — keeps the live record
    byte-comparable to a training CSV row.
  * ICMP_TYPE follows the Sarhan NF-v3 spec: (type << 8) | code. Training
    stats.mean=22777 confirms this is the dataset encoding. ICMP_IPV4_TYPE is
    the bare type.
  * Flow close: RST closes immediately. FIN waits for the peer FIN OR a
    5 s FIN-grace, whichever comes first. nProbe-style — single-FIN closes in
    the old tracker produced clipped CLIENT/SERVER_TCP_FLAGS distributions.
  * Active timeout 120 s, inactive timeout 15 s — nProbe defaults.
  * Retransmission tracks (seq, payload_len) tuples to avoid flagging MTU-
    fragmented packets as retransmits.
  * L7_PROTO is currently a port/protocol fallback. nDPI integration will
    replace _guess_l7_proto when available.
"""

from typing import Callable, Optional
import numpy as np
from scapy.all import IP, TCP, UDP, ICMP, DNS

ACTIVE_TIMEOUT_DEFAULT = 120.0
INACTIVE_TIMEOUT_DEFAULT = 15.0
FIN_GRACE_DEFAULT = 5.0


# Port → nDPI-numeric protocol mapping. Best-effort fallback; superseded by
# real nDPI when the binding is enabled.
_PORT_L7 = {
    20: 1, 21: 1,          # FTP
    22: 92,                # SSH
    23: 23,                # Telnet
    25: 3,                 # SMTP
    53: 5,                 # DNS
    67: 6, 68: 6,          # DHCP
    69: 73,                # TFTP
    80: 7, 8080: 7,        # HTTP
    110: 4,                # POP3
    119: 137,              # NNTP
    123: 9,                # NTP
    137: 11, 138: 11, 139: 11,  # NetBIOS
    143: 9,                # IMAP (re-using #9 like nDPI uses NTP=NTP — see fallback below)
    161: 10, 162: 10,      # SNMP
    179: 65,               # BGP
    194: 12,               # IRC
    389: 84,               # LDAP
    443: 91, 8443: 91,     # TLS/HTTPS
    465: 3,                # SMTPS
    514: 113,              # Syslog
    520: 14,               # RIP
    554: 13,               # RTSP
    587: 3,                # Submission
    631: 86,               # IPP
    636: 84,               # LDAPS
    993: 9,                # IMAPS
    995: 4,                # POP3S
    1080: 90,              # SOCKS
    1194: 110,             # OpenVPN
    1433: 18,              # MSSQL
    1521: 117,             # Oracle SQL
    1812: 36, 1813: 36,    # RADIUS
    1883: 222,             # MQTT
    1900: 138,             # SSDP
    2049: 70,              # NFS
    3306: 19,              # MySQL
    3389: 88,              # RDP
    3478: 41,              # STUN
    4789: 224,             # VXLAN
    5060: 100, 5061: 100,  # SIP
    5222: 16, 5223: 16,    # XMPP
    5353: 8,               # mDNS
    5355: 30,              # LLMNR
    5432: 17,              # PostgreSQL
    5900: 87,              # VNC
    6379: 224,             # Redis (placeholder — no canonical nDPI id)
    27017: 226,            # MongoDB
}


def _guess_l7_proto(dst_port: int, src_port: int, protocol: int) -> int:
    """Port + protocol heuristic for nDPI app_protocol code.

    Real nDPI does payload-aware DPI; we approximate with port lookups plus
    transport fallback. Distribution will be narrower than training, which is
    why nDPI integration is the recommended follow-up.
    """
    if dst_port in _PORT_L7:
        return _PORT_L7[dst_port]
    if src_port in _PORT_L7:
        return _PORT_L7[src_port]
    if protocol == 1:    # ICMP
        return 81
    if protocol == 2:    # IGMP
        return 82
    if protocol == 17:   # generic UDP
        return 67
    return 0             # Unknown


class NFv3FlowExporter:
    """Group live packets into 5-tuple flows and emit raw NF-v3 dicts.

    Args:
        on_flow_ready: callback(features_dict, src_ip, dst_ip)
        active_timeout: max flow age in seconds (nProbe default 120).
        inactive_timeout: idle gap that closes a flow (nProbe default 15).
        fin_grace: seconds to wait after the first FIN before force-closing.
        require_syn: drop TCP flows that don't begin with a SYN (live capture
            into the middle of an existing flow). Default True — without this,
            stray ACKs look like 0-duration port scans.
    """

    def __init__(
        self,
        on_flow_ready: Optional[Callable] = None,
        active_timeout: float = ACTIVE_TIMEOUT_DEFAULT,
        inactive_timeout: float = INACTIVE_TIMEOUT_DEFAULT,
        fin_grace: float = FIN_GRACE_DEFAULT,
        require_syn: bool = True,
        l7_resolver=None,
    ):
        self.on_flow_ready = on_flow_ready
        self.active_timeout = active_timeout
        self.inactive_timeout = inactive_timeout
        self.fin_grace = fin_grace
        self.require_syn = require_syn
        # Optional side-channel for L7_PROTO. When set, _build_record asks the
        # resolver for the nDPI numeric id; falls back to _guess_l7_proto.
        self.l7_resolver = l7_resolver
        self.active_flows: dict = {}
        self._last_timeout_check = 0.0

    # ---------------------------------------------------------------
    # Flow identification
    # ---------------------------------------------------------------

    def _flow_keys(self, pkt):
        if IP not in pkt or (TCP not in pkt and UDP not in pkt):
            return None
        src_ip, dst_ip, proto = pkt[IP].src, pkt[IP].dst, pkt[IP].proto
        if TCP in pkt:
            src_port, dst_port = pkt[TCP].sport, pkt[TCP].dport
        else:
            src_port, dst_port = pkt[UDP].sport, pkt[UDP].dport
        fwd = (src_ip, src_port, dst_ip, dst_port, proto)
        bwd = (dst_ip, dst_port, src_ip, src_port, proto)
        return fwd, bwd, src_ip, dst_ip, src_port, dst_port, proto

    # ---------------------------------------------------------------
    # Packet handler
    # ---------------------------------------------------------------

    def process_packet(self, pkt):
        keys = self._flow_keys(pkt)
        if keys is None:
            return
        fwd, bwd, src_ip, dst_ip, src_port, dst_port, proto = keys

        current_time = float(pkt.time)
        pkt_len = pkt[IP].len if IP in pkt else len(pkt)
        ttl = pkt[IP].ttl if IP in pkt else 0

        if fwd in self.active_flows:
            flow = self.active_flows[fwd]
            direction = "fwd"
        elif bwd in self.active_flows:
            flow = self.active_flows[bwd]
            direction = "bwd"
        else:
            if proto == 6:  # TCP
                if self.require_syn and TCP in pkt and "S" not in str(pkt[TCP].flags):
                    return
            flow = self._create_flow(
                pkt, current_time, pkt_len, ttl,
                src_ip, dst_ip, src_port, dst_port, proto,
            )
            self.active_flows[fwd] = flow
            direction = "fwd"

        self._update_flow(flow, pkt, direction, current_time, pkt_len, ttl)

        # Timeouts — checked once per second of capture time.
        if current_time - self._last_timeout_check > 1.0:
            self._last_timeout_check = current_time
            self._sweep_timeouts(current_time)

        # TCP close conditions.
        if TCP in pkt:
            flags = pkt[TCP].flags
            if "R" in flags:
                self._close(fwd if direction == "fwd" else bwd)
                return
            if "F" in flags:
                if direction == "fwd":
                    flow["fwd_fin"] = True
                else:
                    flow["bwd_fin"] = True
                if flow["fin_deadline"] is None:
                    flow["fin_deadline"] = current_time + self.fin_grace
                if flow["fwd_fin"] and flow["bwd_fin"]:
                    self._close(fwd if direction == "fwd" else bwd)
                    return

    # ---------------------------------------------------------------
    # Flow lifecycle
    # ---------------------------------------------------------------

    def _create_flow(self, pkt, t, pkt_len, ttl,
                     src_ip, dst_ip, src_port, dst_port, proto):
        flow = {
            "src_ip": src_ip, "dst_ip": dst_ip,
            "src_port": src_port, "dst_port": dst_port,
            "protocol": proto,
            "l7_proto": _guess_l7_proto(dst_port, src_port, proto),
            "start_time": t,
            "last_time": t,

            "in_bytes": 0, "in_pkts": 0,
            "out_bytes": 0, "out_pkts": 0,
            "fwd_pkt_lengths": [],
            "bwd_pkt_lengths": [],
            "fwd_timestamps": [],
            "bwd_timestamps": [],

            "ttl_values": [],
            "longest_pkt": 0,
            "shortest_pkt": None,

            "tcp_flags": 0,
            "client_tcp_flags": 0,
            "server_tcp_flags": 0,
            "tcp_win_max_in": 0,
            "tcp_win_max_out": 0,

            "retransmitted_in_bytes": 0, "retransmitted_in_pkts": 0,
            "retransmitted_out_bytes": 0, "retransmitted_out_pkts": 0,
            "_fwd_seq_seen": set(),  # (seq, payload_len)
            "_bwd_seq_seen": set(),

            "pkts_up_to_128": 0,
            "pkts_128_to_256": 0,
            "pkts_256_to_512": 0,
            "pkts_512_to_1024": 0,
            "pkts_1024_to_1514": 0,

            "icmp_type_combined": 0,  # (type << 8) | code
            "icmp_ipv4_type": 0,

            "dns_query_id": 0,
            "dns_query_type": 0,
            "dns_ttl_answer": 0,

            "ftp_command_ret_code": 0,

            "fwd_fin": False,
            "bwd_fin": False,
            "fin_deadline": None,
        }
        return flow

    def _update_flow(self, flow, pkt, direction, t, pkt_len, ttl):
        flow["last_time"] = t
        flow["ttl_values"].append(ttl)

        if flow["shortest_pkt"] is None or pkt_len < flow["shortest_pkt"]:
            flow["shortest_pkt"] = pkt_len
        if pkt_len > flow["longest_pkt"]:
            flow["longest_pkt"] = pkt_len

        if pkt_len <= 128:
            flow["pkts_up_to_128"] += 1
        elif pkt_len <= 256:
            flow["pkts_128_to_256"] += 1
        elif pkt_len <= 512:
            flow["pkts_256_to_512"] += 1
        elif pkt_len <= 1024:
            flow["pkts_512_to_1024"] += 1
        elif pkt_len <= 1514:
            flow["pkts_1024_to_1514"] += 1
        # >1514 (jumbo) is intentionally not counted — no training bucket.

        if direction == "fwd":
            flow["in_bytes"] += pkt_len
            flow["in_pkts"] += 1
            flow["fwd_pkt_lengths"].append(pkt_len)
            flow["fwd_timestamps"].append(t)
        else:
            flow["out_bytes"] += pkt_len
            flow["out_pkts"] += 1
            flow["bwd_pkt_lengths"].append(pkt_len)
            flow["bwd_timestamps"].append(t)

        if TCP in pkt:
            flags_int = int(pkt[TCP].flags)
            flow["tcp_flags"] |= flags_int
            if direction == "fwd":
                flow["client_tcp_flags"] |= flags_int
                if pkt[TCP].window > flow["tcp_win_max_in"]:
                    flow["tcp_win_max_in"] = pkt[TCP].window
                key = (pkt[TCP].seq, len(pkt[TCP].payload))
                if key in flow["_fwd_seq_seen"]:
                    flow["retransmitted_in_bytes"] += pkt_len
                    flow["retransmitted_in_pkts"] += 1
                else:
                    flow["_fwd_seq_seen"].add(key)
            else:
                flow["server_tcp_flags"] |= flags_int
                if pkt[TCP].window > flow["tcp_win_max_out"]:
                    flow["tcp_win_max_out"] = pkt[TCP].window
                key = (pkt[TCP].seq, len(pkt[TCP].payload))
                if key in flow["_bwd_seq_seen"]:
                    flow["retransmitted_out_bytes"] += pkt_len
                    flow["retransmitted_out_pkts"] += 1
                else:
                    flow["_bwd_seq_seen"].add(key)

        if ICMP in pkt:
            icmp_type = int(pkt[ICMP].type)
            icmp_code = int(pkt[ICMP].code)
            flow["icmp_type_combined"] = (icmp_type << 8) | icmp_code
            flow["icmp_ipv4_type"] = icmp_type

        if pkt.haslayer(DNS):
            dns = pkt[DNS]
            if dns.qr == 0:
                flow["dns_query_id"] = int(dns.id)
                if dns.qdcount > 0 and dns.qd is not None:
                    try:
                        flow["dns_query_type"] = int(dns.qd.qtype)
                    except AttributeError:
                        pass
            else:
                if dns.ancount > 0 and dns.an is not None:
                    try:
                        flow["dns_ttl_answer"] = int(dns.an.ttl)
                    except AttributeError:
                        pass

    # ---------------------------------------------------------------
    # Timeouts
    # ---------------------------------------------------------------

    def _sweep_timeouts(self, now: float):
        to_close = []
        for key, flow in self.active_flows.items():
            if now - flow["start_time"] > self.active_timeout:
                to_close.append(key)
                continue
            if now - flow["last_time"] > self.inactive_timeout:
                to_close.append(key)
                continue
            if flow["fin_deadline"] is not None and now > flow["fin_deadline"]:
                to_close.append(key)
        for k in to_close:
            self._close(k)

    def _close(self, key):
        flow = self.active_flows.pop(key, None)
        if flow is None:
            return
        features = self._build_record(flow)
        if self.on_flow_ready is not None:
            self.on_flow_ready(features, flow["src_ip"], flow["dst_ip"])

    def flush_all(self):
        for key in list(self.active_flows.keys()):
            self._close(key)

    # ---------------------------------------------------------------
    # Record construction (raw NF-v3 dict)
    # ---------------------------------------------------------------

    @staticmethod
    def _iat_stats_ms(timestamps):
        if len(timestamps) < 2:
            return 0.0, 0.0, 0.0, 0.0
        iats = np.diff(timestamps) * 1000.0
        return (
            float(np.min(iats)),
            float(np.max(iats)),
            float(np.mean(iats)),
            float(np.std(iats)),
        )

    def _build_record(self, flow) -> dict:
        duration_ms = (flow["last_time"] - flow["start_time"]) * 1000.0
        duration_sec = duration_ms / 1000.0

        if len(flow["fwd_timestamps"]) > 1:
            duration_in_ms = (flow["fwd_timestamps"][-1] - flow["fwd_timestamps"][0]) * 1000.0
        else:
            duration_in_ms = 0.0
        if len(flow["bwd_timestamps"]) > 1:
            duration_out_ms = (flow["bwd_timestamps"][-1] - flow["bwd_timestamps"][0]) * 1000.0
        else:
            duration_out_ms = 0.0

        if duration_sec > 0:
            s2d_bps = flow["in_bytes"] / duration_sec
            d2s_bps = flow["out_bytes"] / duration_sec
            s2d_thr = (flow["in_bytes"] * 8) / duration_sec
            d2s_thr = (flow["out_bytes"] * 8) / duration_sec
        else:
            s2d_bps = d2s_bps = s2d_thr = d2s_thr = 0.0

        s2d_iat = self._iat_stats_ms(flow["fwd_timestamps"])
        d2s_iat = self._iat_stats_ms(flow["bwd_timestamps"])

        l7_proto = flow["l7_proto"]
        if self.l7_resolver is not None:
            resolved = self.l7_resolver.lookup(
                flow["src_ip"], flow["src_port"],
                flow["dst_ip"], flow["dst_port"],
                flow["protocol"],
            )
            if resolved is not None and resolved != 0:
                l7_proto = resolved

        return {
            "IPV4_SRC_ADDR": flow["src_ip"],
            "L4_SRC_PORT": flow["src_port"],
            "IPV4_DST_ADDR": flow["dst_ip"],
            "L4_DST_PORT": flow["dst_port"],
            "FLOW_START_MILLISECONDS": int(flow["start_time"] * 1000),
            "FLOW_END_MILLISECONDS": int(flow["last_time"] * 1000),
            "PROTOCOL": flow["protocol"],
            "L7_PROTO": l7_proto,
            "IN_BYTES": flow["in_bytes"],
            "IN_PKTS": flow["in_pkts"],
            "OUT_BYTES": flow["out_bytes"],
            "OUT_PKTS": flow["out_pkts"],
            "TCP_FLAGS": flow["tcp_flags"],
            "CLIENT_TCP_FLAGS": flow["client_tcp_flags"],
            "SERVER_TCP_FLAGS": flow["server_tcp_flags"],
            "FLOW_DURATION_MILLISECONDS": duration_ms,
            "DURATION_IN": duration_in_ms,
            "DURATION_OUT": duration_out_ms,
            "MIN_TTL": min(flow["ttl_values"]) if flow["ttl_values"] else 0,
            "MAX_TTL": max(flow["ttl_values"]) if flow["ttl_values"] else 0,
            "LONGEST_FLOW_PKT": flow["longest_pkt"],
            "SHORTEST_FLOW_PKT": flow["shortest_pkt"] or 0,
            "MIN_IP_PKT_LEN": flow["shortest_pkt"] or 0,
            "MAX_IP_PKT_LEN": flow["longest_pkt"],
            "SRC_TO_DST_SECOND_BYTES": s2d_bps,
            "DST_TO_SRC_SECOND_BYTES": d2s_bps,
            "RETRANSMITTED_IN_BYTES": flow["retransmitted_in_bytes"],
            "RETRANSMITTED_IN_PKTS": flow["retransmitted_in_pkts"],
            "RETRANSMITTED_OUT_BYTES": flow["retransmitted_out_bytes"],
            "RETRANSMITTED_OUT_PKTS": flow["retransmitted_out_pkts"],
            "SRC_TO_DST_AVG_THROUGHPUT": s2d_thr,
            "DST_TO_SRC_AVG_THROUGHPUT": d2s_thr,
            "NUM_PKTS_UP_TO_128_BYTES": flow["pkts_up_to_128"],
            "NUM_PKTS_128_TO_256_BYTES": flow["pkts_128_to_256"],
            "NUM_PKTS_256_TO_512_BYTES": flow["pkts_256_to_512"],
            "NUM_PKTS_512_TO_1024_BYTES": flow["pkts_512_to_1024"],
            "NUM_PKTS_1024_TO_1514_BYTES": flow["pkts_1024_to_1514"],
            "TCP_WIN_MAX_IN": flow["tcp_win_max_in"],
            "TCP_WIN_MAX_OUT": flow["tcp_win_max_out"],
            "ICMP_TYPE": flow["icmp_type_combined"],
            "ICMP_IPV4_TYPE": flow["icmp_ipv4_type"],
            "DNS_QUERY_ID": flow["dns_query_id"],
            "DNS_QUERY_TYPE": flow["dns_query_type"],
            "DNS_TTL_ANSWER": flow["dns_ttl_answer"],
            "FTP_COMMAND_RET_CODE": flow["ftp_command_ret_code"],
            "SRC_TO_DST_IAT_MIN": s2d_iat[0],
            "SRC_TO_DST_IAT_MAX": s2d_iat[1],
            "SRC_TO_DST_IAT_AVG": s2d_iat[2],
            "SRC_TO_DST_IAT_STDDEV": s2d_iat[3],
            "DST_TO_SRC_IAT_MIN": d2s_iat[0],
            "DST_TO_SRC_IAT_MAX": d2s_iat[1],
            "DST_TO_SRC_IAT_AVG": d2s_iat[2],
            "DST_TO_SRC_IAT_STDDEV": d2s_iat[3],
        }


# Backwards-compatible alias for callers still importing the old name.
NFv3FlowTracker = NFv3FlowExporter
