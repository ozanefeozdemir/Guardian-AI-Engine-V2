import { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE } from '../config';
import { ConsoleRail, ConsoleTopBar, ConsoleStatusBar, Field } from './ui/primitives';
import Icon from './ui/Icon';

// Görsel: Operations Console IP politikaları. Veri KORUNDU: GET/POST/DELETE /api/ip-rules (Bearer).
export default function IPRules({ view, onView, onRoute, onLogout, isConnected, alerts = [], stats = {}, username = 'admin', role = 'Yönetici' }) {
  const [tab, setTab] = useState('whitelist');
  const [rules, setRules] = useState([]);
  const [draft, setDraft] = useState({ cidr: '', reason: '' });

  const authHeader = () => {
    const token = localStorage.getItem('guardian_token');
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  const fetchRules = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/ip-rules?list_type=${tab}`, { headers: authHeader() });
      setRules(Array.isArray(res.data) ? res.data : []);
    } catch (err) {
      console.error('Kurallar çekilemedi', err);
      setRules([]);
    }
  };

  useEffect(() => { fetchRules(); /* eslint-disable-next-line */ }, [tab]);

  const addRule = async (e) => {
    e.preventDefault();
    if (!draft.cidr.trim()) return;
    const payload = { cidr: draft.cidr.trim(), list_type: tab, reason: draft.reason || null, expires_at: null };
    try {
      await axios.post(`${API_BASE}/api/ip-rules`, payload, { headers: authHeader() });
      setDraft({ cidr: '', reason: '' });
      fetchRules();
    } catch (err) {
      const detail = err.response?.data?.detail;
      alert(typeof detail === 'object' ? 'Format Hatası: Verileri kontrol edin (Örn: 192.168.1.0/24)' : (detail || 'Kural eklenemedi'));
    }
  };

  const remove = async (id) => {
    if (!window.confirm('Bu kuralı silmek istediğinize emin misiniz?')) return;
    try {
      await axios.delete(`${API_BASE}/api/ip-rules/${id}`, { headers: authHeader() });
      fetchRules();
    } catch (err) {
      console.error('Silme hatası', err);
    }
  };

  const segBtn = (key, label, activeColor) => (
    <button onClick={() => setTab(key)} style={{
      padding: '5px 14px', borderRadius: 3, border: 0,
      background: tab === key ? activeColor : 'transparent',
      color: tab === key ? 'var(--ink-0)' : 'var(--fg-3)',
      font: '600 10px/1 var(--font-mono)', letterSpacing: '0.12em', textTransform: 'uppercase',
    }}>{label}</button>
  );

  return (
    <div className="dash-shell">
      <ConsoleRail route="iprules" view={view} onRoute={onRoute} onView={onView} alertCount={alerts.length} username={username} role={role} />

      <div className="console-main">
        <ConsoleTopBar here="IP Kuralları" crumbs={['Politika']} onLogout={onLogout}>
          <div style={{ display: 'flex', padding: 2, background: 'var(--ink-2)', borderRadius: 5, border: '1px solid var(--ink-5)' }}>
            {segBtn('whitelist', 'Güvenli', 'var(--success)')}
            {segBtn('blacklist', 'Yasaklı', 'var(--critical)')}
          </div>
        </ConsoleTopBar>

        <div className="console-canvas">
          <div style={{ marginBottom: 4 }}>
            <div className="eyebrow" style={{ marginBottom: 8 }}>Politika · CIDR formatında</div>
            <h1 style={{ font: '400 32px/1.1 var(--font-sans)', color: 'var(--fg-0)', margin: 0, letterSpacing: '-0.022em' }}>
              {tab === 'whitelist' ? 'Güvenli IP aralıkları' : 'Yasaklı IP aralıkları'}
            </h1>
            <p style={{ font: '400 13px/1.55 var(--font-sans)', color: 'var(--fg-3)', margin: '8px 0 0', maxWidth: '60ch' }}>
              {tab === 'whitelist'
                ? "Bu listedeki IP'ler AI motorunu atlatır — her zaman güvenli sayılır."
                : "Bu listedeki IP'lerden gelen tüm trafik otomatik olarak engellenir."}
            </p>
          </div>

          <form onSubmit={addRule} className="dcard" style={{ display: 'grid', gridTemplateColumns: '1fr 2fr auto', gap: 16, alignItems: 'end' }}>
            <Field label="IP / CIDR" value={draft.cidr} onChange={(e) => setDraft({ ...draft, cidr: e.target.value })} placeholder="192.168.1.0/24" />
            <Field label="Açıklama" value={draft.reason} onChange={(e) => setDraft({ ...draft, reason: e.target.value })} placeholder="Güvenilir iç ağ trafiği..." />
            <button type="submit" className="btn btn--primary"><Icon name="plus" size={13} /> Kural Ekle</button>
          </form>

          <div className="dcard" style={{ padding: 0, overflow: 'hidden' }}>
            <table className="table">
              <thead>
                <tr>
                  <th>Kapsam (CIDR)</th>
                  <th>Gerekçe</th>
                  <th>Ekleyen</th>
                  <th style={{ width: 80 }}>İşlem</th>
                </tr>
              </thead>
              <tbody>
                {rules.map((r) => (
                  <tr key={r.id}>
                    <td className="t-port" style={{ fontWeight: 600 }}>{r.cidr}</td>
                    <td style={{ color: 'var(--fg-1)' }}>{r.reason || '-'}</td>
                    <td style={{ color: 'var(--fg-3)', font: '400 11px/1 var(--font-mono)' }}>
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                        <Icon name="terminal" size={11} /> {r.created_by || '-'}
                      </span>
                    </td>
                    <td>
                      <button onClick={() => remove(r.id)} style={{ background: 'transparent', border: 0, color: 'var(--fg-4)', cursor: 'pointer', padding: 4 }}>
                        <Icon name="trash-2" size={15} />
                      </button>
                    </td>
                  </tr>
                ))}
                {rules.length === 0 && (
                  <tr><td colSpan={4} style={{ padding: 40, textAlign: 'center', color: 'var(--fg-4)', fontStyle: 'italic' }}>Henüz tanımlanmış bir kural bulunmuyor.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <ConsoleStatusBar isConnected={isConnected} totalTraffic={stats.totalTraffic || 0} totalAttacks={stats.totalAttacks || 0} />
    </div>
  );
}
