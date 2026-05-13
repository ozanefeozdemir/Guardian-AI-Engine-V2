"""Attack-detection sanity check.

Runs in two phases:
  1. Baseline — collect normal background flows for BASELINE_SEC seconds.
  2. Attack   — launch a real probe (nmap port scan / curl burst) against
                a target and continue capturing for ATTACK_SEC seconds.

Reports flow counts and attack-rate for each phase, plus a focused view of
the flows whose destination matches the attack target.

Requires sudo (raw socket capture). Example:

    sudo $(which python) backend/test_attack_detection.py \
        --attack scan --target scanme.nmap.org

scanme.nmap.org is Nmap's official scan-permitted test host. For purely
local testing point --target at your own host (127.0.0.1 has no real
flows; pick a LAN host or a server you control).
"""

import os
import sys
import time
import socket
import argparse
import subprocess
import threading
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scapy.all import AsyncSniffer, IP  # noqa: E402

from nfv3_flow_tracker import NFv3FlowExporter  # noqa: E402
from model_provider import get_model_provider  # noqa: E402


class AttackProbe:
    """Wraps a subprocess attack so we can run it on a timeline."""

    def __init__(self, attack: str, target: str):
        self.attack = attack
        self.target = target
        self._proc = None

    def start(self):
        if self.attack == "scan":
            # -sT (TCP connect scan) avoids needing sudo for nmap itself;
            # sniff still requires sudo on macOS. -T4 = aggressive timing.
            # -p 1-500 keeps the run bounded.
            cmd = ["nmap", "-sT", "-T4", "-p", "1-500", "-Pn", self.target]
        elif self.attack == "http_burst":
            # 200 parallel HTTPS connections to the target's port 443.
            cmd = [
                "bash", "-c",
                f"for i in $(seq 1 200); do "
                f"curl -s -o /dev/null --max-time 3 https://{self.target}/ & "
                f"done; wait",
            ]
        else:
            raise ValueError(f"Unknown attack: {self.attack}")
        print(f"[attack] launching: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def wait(self, timeout: float):
        if self._proc is None:
            return
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def resolve_targets(target: str) -> set:
    """Return all IPv4 addresses a hostname resolves to, plus the literal."""
    out = {target}
    try:
        infos = socket.getaddrinfo(target, None, socket.AF_INET)
        for info in infos:
            out.add(info[4][0])
    except socket.gaierror:
        pass
    return out


class Tester:
    def __init__(self, provider_name, target, attack, baseline_sec, attack_sec):
        self.provider_name = provider_name
        self.target = target
        self.attack = attack
        self.baseline_sec = baseline_sec
        self.attack_sec = attack_sec

        self.provider = None
        self.target_ips = resolve_targets(target)
        self.attack_start = None  # set when attack subprocess launches
        self._lock = threading.Lock()
        self.flows = []  # list of (close_time, src_ip, dst_ip, is_attack, conf, features)

    def _on_flow_ready(self, features, src_ip, dst_ip):
        try:
            res = self.provider.predict(features)
        except Exception as exc:
            print(f"[predict-error] {exc}")
            return
        with self._lock:
            self.flows.append(
                (
                    time.time(),
                    src_ip,
                    dst_ip,
                    bool(res["is_attack"]),
                    float(res["confidence"]),
                    features,
                )
            )

    def run(self):
        print(f"[setup] target={self.target} → IPs={sorted(self.target_ips)}")
        print(f"[setup] provider={self.provider_name} | baseline={self.baseline_sec}s | "
              f"attack={self.attack_sec}s | type={self.attack}")
        self.provider = get_model_provider(self.provider_name)
        self.provider.load()

        exporter = NFv3FlowExporter(on_flow_ready=self._on_flow_ready)

        # Background sweep so idle / half-open flows time out even when no
        # further packets arrive on them. NFv3FlowExporter's internal sweep
        # is packet-driven, which fails for unanswered SYNs.
        sweep_stop = threading.Event()

        def sweep_loop():
            while not sweep_stop.is_set():
                try:
                    exporter._sweep_timeouts(time.time())
                except Exception as exc:
                    print(f"[sweep-error] {exc}")
                sweep_stop.wait(1.0)

        sweep_thread = threading.Thread(target=sweep_loop, daemon=True)
        sweep_thread.start()

        def packet_cb(pkt):
            if IP in pkt:
                exporter.process_packet(pkt)

        # BPF: skip our own infra ports so they don't pollute attack-window.
        bpf = "ip and not (port 6379 or port 5432 or port 8000)"
        sniffer = AsyncSniffer(prn=packet_cb, store=False, filter=bpf)
        sniffer.start()
        print(f"[sniff] async capture started.")

        # ── Baseline window ───────────────────────────────────────────
        print(f"[phase] baseline ({self.baseline_sec}s) — leave the machine idle.")
        time.sleep(self.baseline_sec)
        baseline_end = time.time()
        baseline_count = len(self.flows)
        print(f"[phase] baseline ended: {baseline_count} flows closed so far.")

        # ── Attack window ─────────────────────────────────────────────
        probe = AttackProbe(self.attack, self.target)
        self.attack_start = time.time()
        probe.start()
        probe.wait(timeout=self.attack_sec)
        # Continue sniffing briefly so attack-period flows have time to close.
        post_wait = max(0.0, self.attack_sec - (time.time() - self.attack_start))
        if post_wait > 0:
            time.sleep(post_wait)
        time.sleep(5)  # drain — give exporter time to flush short-lived flows

        # Stop sniffer with join so queued packets finish processing before
        # we partition windows.
        sniffer.stop(join=True)
        sweep_stop.set()
        sweep_thread.join(timeout=2)
        exporter.flush_all()

        # ── Partition flows by window ─────────────────────────────────
        baseline_flows = [f for f in self.flows if f[0] <= baseline_end]
        attack_flows = [f for f in self.flows if f[0] > baseline_end]

        self._report("baseline", baseline_flows)
        self._report("attack-window (all)", attack_flows)

        # Flows directly involving the attack target.
        target_attack_flows = [
            f for f in attack_flows if f[2] in self.target_ips or f[1] in self.target_ips
        ]
        self._report("attack-window (target only)", target_attack_flows, target_focus=True)

    def _report(self, label, flows, target_focus=False):
        n = len(flows)
        print()
        print("=" * 72)
        print(f"  {label}: {n} flows")
        if n == 0:
            print("  (no flows in this slice)")
            print("=" * 72)
            return
        attacks = [f for f in flows if f[3]]
        benigns = [f for f in flows if not f[3]]
        print(f"  attack={len(attacks)} ({100 * len(attacks) / n:.1f}%) | "
              f"benign={len(benigns)} ({100 * len(benigns) / n:.1f}%)")
        if attacks:
            confs = [f[4] for f in attacks]
            print(f"  attack confidence: min={min(confs):.3f} max={max(confs):.3f} "
                  f"mean={sum(confs)/len(confs):.3f}")
        if benigns:
            confs = [f[4] for f in benigns]
            print(f"  benign confidence: min={min(confs):.3f} max={max(confs):.3f} "
                  f"mean={sum(confs)/len(confs):.3f}")

        dport_attack = defaultdict(int)
        dport_total = defaultdict(int)
        dip_total = defaultdict(int)
        for _ct, _sip, dip, isatk, _conf, feats in flows:
            dp = feats.get("L4_DST_PORT", 0)
            dport_total[dp] += 1
            if isatk:
                dport_attack[dp] += 1
            dip_total[dip] += 1
        top_ips = sorted(dip_total.items(), key=lambda x: -x[1])[:6]
        print("  top dst IPs:")
        for dip, cnt in top_ips:
            mark = "  ← target" if dip in self.target_ips else ""
            print(f"    {dip:<32}  flows={cnt}{mark}")
        if target_focus:
            top_ports = sorted(dport_total.items(), key=lambda x: -x[1])[:10]
            print("  top dst ports (target-bound):")
            for port, total in top_ports:
                print(f"    {port:>6}  total={total}  attack={dport_attack.get(port, 0)}")
        print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="flowguard")
    parser.add_argument("--target", default="scanme.nmap.org",
                        help="Hostname or IP to probe (default: scanme.nmap.org, "
                             "Nmap's official scan-permitted test host).")
    parser.add_argument("--attack", choices=["scan", "http_burst"], default="scan")
    parser.add_argument("--baseline-sec", type=int, default=15)
    parser.add_argument("--attack-sec", type=int, default=30)
    args = parser.parse_args()

    Tester(
        provider_name=args.provider,
        target=args.target,
        attack=args.attack,
        baseline_sec=args.baseline_sec,
        attack_sec=args.attack_sec,
    ).run()
