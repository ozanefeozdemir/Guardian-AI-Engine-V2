import { useEffect, useRef, useState } from 'react';
import { WS_BASE, API_BASE } from '../config';

// Backend (analyze_engine.py → api.py /ws/alerts) gerçek canlı akış sözleşmesi.
const WS_URL = `${WS_BASE}/ws/alerts`;
const STATUS_URL = `${API_BASE}/status`;
const MAX_TRAFFIC_POINTS = 20;

const fmtTime = (d = new Date()) => d.toTimeString().slice(0, 8);

// attack_type → whitelist/blacklist rozet türü (diğerleri AI tespiti)
const listTypeOf = (attackType) => {
  if (attackType === 'Whitelisted') return 'whitelist';
  if (attackType === 'Blacklisted') return 'blacklist';
  return undefined;
};

// source = "src_ip" (simülasyon) | "src_ip->dst_ip" (canlı). Hedef port'u
// önce original_features'tan, yoksa source'un hedef kısmından türet.
const deriveSrcDest = (data) => {
  const srcRaw = (data.source ?? '').toString();
  const [srcPart, dstPart] = srcRaw.split('->').map((s) => s.trim());
  const feat = data.original_features || {};
  const rawPort = feat['Destination Port'] ?? feat['Dst Port'] ?? feat['dst_port'];
  let dest = dstPart || '—';
  if (rawPort != null && rawPort !== '' && !Number.isNaN(Number(rawPort))) {
    dest = String(Math.round(Number(rawPort)));
  }
  return { src: srcPart || 'Gizli IP', dest };
};

/**
 * Tek WebSocket aboneliği + /status polling + saniyelik grafik tamponu.
 * Sonuç tüm ekranlarda (Modern/Classic/rail/statusbar) paylaşılır.
 */
export function useGuardianLive(enabled = true) {
  const [isConnected, setIsConnected] = useState(false);
  const [systemStatus, setSystemStatus] = useState('Checking...');
  const [alerts, setAlerts] = useState([]);
  const [stats, setStats] = useState({ totalTraffic: 0, totalAttacks: 0, lastConfidence: 0 });
  const [traffic, setTraffic] = useState([]);
  const [distribution, setDistribution] = useState([]);

  const bufferRef = useRef({ packets: 0, attacks: 0, types: {} });
  const seqRef = useRef(0);

  // 1) Sistem sağlığı — /status her 10 sn
  useEffect(() => {
    if (!enabled) return undefined;
    let alive = true;
    const check = async () => {
      try {
        const res = await fetch(STATUS_URL);
        if (alive) setSystemStatus(res.ok ? 'ONLINE' : 'API ERROR');
      } catch {
        if (alive) setSystemStatus('OFFLINE');
      }
    };
    check();
    const id = setInterval(check, 10000);
    return () => { alive = false; clearInterval(id); };
  }, [enabled]);

  // 2) WebSocket — canlı uyarılar (reconnect mantıklı)
  useEffect(() => {
    if (!enabled) return undefined;
    let ws;
    let reconnect;
    const connect = () => {
      ws = new WebSocket(WS_URL);

      ws.onopen = () => setIsConnected(true);

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          bufferRef.current.packets += 1;

          if (data.is_attack) {
            const { src, dest } = deriveSrcDest(data);
            const type = data.attack_type || 'Bilinmeyen';
            bufferRef.current.attacks += 1;
            bufferRef.current.types[type] = (bufferRef.current.types[type] || 0) + 1;
            seqRef.current += 1;

            const alert = {
              id: `${data.timestamp}-${seqRef.current}`,
              tsMs: Date.now(),
              time: fmtTime(),
              type,
              confidence: data.confidence ?? 0,
              src,
              dest,
              rule_id: data.rule_id,
              list_type: listTypeOf(data.attack_type),
            };

            setAlerts((prev) => [alert, ...prev].slice(0, 50));
            setStats((prev) => ({
              totalTraffic: prev.totalTraffic + 1,
              totalAttacks: prev.totalAttacks + 1,
              lastConfidence: data.confidence ?? prev.lastConfidence,
            }));
          } else {
            setStats((prev) => ({ ...prev, totalTraffic: prev.totalTraffic + 1 }));
          }
        } catch {
          /* bozuk mesaj — yok say */
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        reconnect = setTimeout(connect, 3000);
      };
      ws.onerror = () => { if (ws) ws.close(); };
    };

    connect();
    return () => { clearTimeout(reconnect); if (ws) ws.close(); };
  }, [enabled]);

  // 3) Saniyelik grafik güncelleme döngüsü
  useEffect(() => {
    if (!enabled) return undefined;
    const id = setInterval(() => {
      const point = {
        name: fmtTime(),
        traffic: bufferRef.current.packets,
        attack: bufferRef.current.attacks,
      };
      setTraffic((prev) => [...prev, point].slice(-MAX_TRAFFIC_POINTS));
      setDistribution(
        Object.entries(bufferRef.current.types).map(([name, value]) => ({ name, value }))
      );
      bufferRef.current.packets = 0;
      bufferRef.current.attacks = 0;
    }, 1000);
    return () => clearInterval(id);
  }, [enabled]);

  return { isConnected, systemStatus, alerts, stats, traffic, distribution };
}
