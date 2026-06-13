import { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE } from '../config';
import Icon from './ui/Icon';

// Görsel: Operations Console denetim log tablosu. Veri KORUNDU: GET /api/auth/logs.
export default function AuthLogs() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const res = await axios.get(`${API_BASE}/api/auth/logs`);
      setLogs(Array.isArray(res.data) ? res.data : []);
    } catch (err) {
      console.error('Loglar çekilemedi', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchLogs(); }, []);

  const iconFor = (action = '') => {
    if (action.includes('BAŞARISIZ') || action.includes('Yanlış')) return { name: 'user-x', color: 'var(--critical)' };
    if (action.includes('Başarılı')) return { name: 'user-check', color: 'var(--success)' };
    return { name: 'shield-alert', color: 'var(--warning)' };
  };

  const fmt = (iso) => {
    if (!iso) return '-';
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString('tr-TR', {
      day: '2-digit', month: '2-digit', year: 'numeric',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
  };

  return (
    <div className="dcard" style={{ padding: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden', width: '100%' }}>
      <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--ink-4)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div className="dcard__title">
          <Icon name="shield-alert" size={16} color="var(--signal-400)" />
          Sistem Denetim Logları (Audit Logs)
        </div>
        <button onClick={fetchLogs} className="btn btn--secondary" style={{ padding: '8px 12px', fontSize: 11, borderRadius: 8 }} aria-label="Yenile">
          <Icon name="refresh-cw" size={14} style={{ animation: loading ? 'spin 0.8s linear infinite' : 'none' }} />
          Yenile
        </button>
      </div>
      <div style={{ overflow: 'auto', maxHeight: 360 }}>
        <table className="table">
          <thead>
            <tr>
              <th style={{ width: '40%' }}>Eylem / Durum</th>
              <th>Kullanıcı Adı</th>
              <th>IP Adresi</th>
              <th>Tarih / Saat</th>
            </tr>
          </thead>
          <tbody>
            {logs.map((log) => {
              const ic = iconFor(log.action);
              const failed = (log.action || '').includes('BAŞARISIZ');
              return (
                <tr key={log.id}>
                  <td style={{ padding: '14px 12px' }}>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 10 }}>
                      <Icon name={ic.name} size={16} color={ic.color} />
                      <span style={{ color: failed ? 'var(--critical)' : 'var(--fg-1)', fontWeight: failed ? 600 : 500, fontSize: 12 }}>{log.action}</span>
                    </div>
                  </td>
                  <td style={{ color: 'var(--signal-300)', fontFamily: 'var(--font-mono)', fontWeight: 500, fontSize: 12 }}>{log.username}</td>
                  <td className="t-ip">{log.ip_address}</td>
                  <td className="t-time">{fmt(log.timestamp)}</td>
                </tr>
              );
            })}
            {logs.length === 0 && (
              <tr><td colSpan={4} style={{ padding: 40, textAlign: 'center', color: 'var(--fg-4)', fontStyle: 'italic' }}>
                {loading ? 'Loglar yükleniyor...' : 'Henüz kayıtlı bir denetim logu bulunmuyor.'}
              </td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
