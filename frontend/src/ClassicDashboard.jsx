import { RiskBadge } from './components/ui/primitives';
import Icon from './components/ui/Icon';

const BAR_COLORS = ['#FF5A6B', '#F08A4B', '#F5B549', '#43B5C0', '#6DA6FF', '#A78BFA'];

// v1.0 — açık temalı (paper) varyant. Konsol kabuğunu atlar; kendi koyu sidebar'ı vardır.
// Aynı canlı veriyi (alerts, stats, distribution) tüketir.
export default function ClassicDashboard({ alerts = [], stats = {}, distribution = [], isConnected = false, onView, onLogout }) {
  const totalAttacks = stats.totalAttacks || 0;
  const critical = alerts.filter((a) => (a.confidence || 0) > 0.8).length;
  const watchedSources = new Set(alerts.map((a) => a.src)).size;

  const chart = (distribution.length ? distribution : []).map((d, i) => ({
    type: d.name, count: d.value, color: BAR_COLORS[i % BAR_COLORS.length],
  }));
  const maxCount = chart.length ? Math.max(...chart.map((d) => d.count)) : 1;

  const navItems = [
    { label: 'Ana Panel', icon: 'layout-dashboard', active: true },
    { label: 'Uyarı Geçmişi', icon: 'history' },
    { label: 'Veri Kaynakları', icon: 'database' },
    { label: 'Ayarlar', icon: 'settings' },
  ];

  const kpis = [
    { title: 'Toplam Saldırı', val: totalAttacks, icon: 'alert-triangle', color: '#FF5A6B' },
    { title: 'Kritik Uyarılar', val: critical, icon: 'shield-check', color: '#F08A4B' },
    { title: 'İzlenen Kaynak IP', val: watchedSources, icon: 'database', color: '#0891B2' },
  ];

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--paper-0)', color: 'var(--paper-ink)', fontFamily: 'var(--font-sans)' }}>
      {/* Floating controls (konsol kabuğu yok) */}
      <div style={{ position: 'fixed', top: 16, right: 16, zIndex: 9999, display: 'flex', gap: 8 }}>
        <button onClick={() => onView('modern')} style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '8px 14px', borderRadius: 8, border: '1px solid var(--signal-600)', background: 'var(--signal-500)', color: '#06251f', font: '600 12px/1 var(--font-sans)', cursor: 'pointer' }}>
          v2.0 Modern
        </button>
        <button onClick={onLogout} style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '8px 14px', borderRadius: 8, border: '1px solid rgba(178,59,71,0.5)', background: 'rgba(178,59,71,0.9)', color: '#FFE6E9', font: '600 12px/1 var(--font-sans)', cursor: 'pointer' }}>
          <Icon name="log-out" size={14} /> Çıkış
        </button>
      </div>

      {/* Sidebar (koyu) */}
      <aside style={{ width: 240, background: 'var(--ink-0)', color: 'var(--fg-1)', padding: '24px 16px', position: 'sticky', top: 0, height: '100vh', display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '0 8px 18px', borderBottom: '1px solid var(--ink-4)', marginBottom: 16 }}>
          <div className="brand-tile" style={{ width: 30, height: 30 }}><Icon name="shield-check" size={16} color="white" /></div>
          <span className="brand-word" style={{ fontSize: 18 }}>Guardian</span>
        </div>
        {navItems.map((item) => (
          <div key={item.label} style={{
            display: 'flex', alignItems: 'center', gap: 12, padding: '11px 14px', borderRadius: 8,
            background: item.active ? 'var(--signal-600)' : 'transparent',
            color: item.active ? 'white' : 'var(--fg-2)', cursor: 'pointer', marginBottom: 4, font: '500 13px/1 var(--font-sans)',
          }}>
            <Icon name={item.icon} size={16} /> {item.label}
          </div>
        ))}
        <div style={{ marginTop: 'auto', padding: '10px 14px', borderRadius: 8, background: isConnected ? 'rgba(79,209,139,0.10)' : 'rgba(255,90,107,0.10)', border: `1px solid ${isConnected ? 'rgba(79,209,139,0.30)' : 'rgba(255,90,107,0.30)'}`, display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className={`dot ${isConnected ? 'dot--ok' : 'dot--crit'}`} style={{ width: 6, height: 6 }} />
          <span style={{ font: '600 11px/1 var(--font-mono)', color: isConnected ? 'var(--success)' : 'var(--critical)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
            {isConnected ? 'Bağlantı Aktif' : 'Bağlantı Kesildi'}
          </span>
        </div>
      </aside>

      {/* Main */}
      <main style={{ flex: 1, padding: 32, overflowY: 'auto' }}>
        <h1 style={{ font: '600 30px/1.15 var(--font-sans)', letterSpacing: '-0.014em', color: 'var(--paper-ink)', margin: '0 0 24px' }}>Ana Panel</h1>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 20, marginBottom: 24 }}>
          {kpis.map((k) => (
            <div key={k.title} style={{ background: 'white', padding: '22px 24px', borderRadius: 12, boxShadow: '0 1px 2px rgba(14,20,34,0.06), 0 4px 12px -4px rgba(14,20,34,0.08)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ font: '500 11px/1 var(--font-mono)', color: 'var(--paper-ink-2)', textTransform: 'uppercase', letterSpacing: '0.12em', marginBottom: 8 }}>{k.title}</div>
                <div style={{ font: '600 32px/1 var(--font-mono)', color: 'var(--paper-ink)', fontVariantNumeric: 'tabular-nums' }}>{k.val}</div>
              </div>
              <div style={{ padding: 12, borderRadius: 12, background: `${k.color}15`, color: k.color }}>
                <Icon name={k.icon} size={22} />
              </div>
            </div>
          ))}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 20 }}>
          {/* Canlı uyarılar */}
          <div style={{ background: 'white', padding: 24, borderRadius: 12, boxShadow: '0 1px 2px rgba(14,20,34,0.06), 0 4px 12px -4px rgba(14,20,34,0.08)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16, font: '600 16px/1 var(--font-sans)', color: 'var(--paper-ink)' }}>
              <Icon name="bell" size={18} color="#0891B2" /> Anlık Uyarı Akışı
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 340, overflow: 'auto' }}>
              {alerts.slice(0, 8).map((a) => (
                <div key={a.id} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '10px 12px', background: 'var(--paper-1)', borderRadius: 8 }}>
                  <RiskBadge confidence={a.confidence} listType={a.list_type} />
                  <div style={{ flex: 1, font: '500 13px/1.3 var(--font-sans)', color: 'var(--paper-ink)' }}>
                    Tespit: {a.type}
                    <div style={{ font: '400 11px/1 var(--font-mono)', color: 'var(--paper-ink-2)', marginTop: 4 }}>
                      {a.src} → :{a.dest}
                    </div>
                  </div>
                  <div style={{ font: '400 11px/1 var(--font-mono)', color: 'var(--paper-ink-2)' }}>{a.time}</div>
                </div>
              ))}
              {alerts.length === 0 && (
                <div style={{ padding: 32, textAlign: 'center', color: 'var(--paper-ink-2)', fontStyle: 'italic' }}>Yeni uyarı bekleniyor...</div>
              )}
            </div>
          </div>

          {/* Dağılım */}
          <div style={{ background: 'white', padding: 24, borderRadius: 12, boxShadow: '0 1px 2px rgba(14,20,34,0.06), 0 4px 12px -4px rgba(14,20,34,0.08)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 18, font: '600 16px/1 var(--font-sans)', color: 'var(--paper-ink)' }}>
              <Icon name="bar-chart-horizontal" size={18} color="#0891B2" /> Saldırı Tipi Dağılımı
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {chart.length === 0 && <div style={{ color: 'var(--paper-ink-2)', fontSize: 13 }}>Henüz veri yok.</div>}
              {chart.map((d) => (
                <div key={d.type}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, font: '500 12px/1 var(--font-sans)', color: 'var(--paper-ink)' }}>
                    <span>{d.type}</span><span style={{ fontFamily: 'var(--font-mono)' }}>{d.count}</span>
                  </div>
                  <div style={{ height: 6, background: 'var(--paper-2)', borderRadius: 999, overflow: 'hidden' }}>
                    <div style={{ width: `${(d.count / maxCount) * 100}%`, height: '100%', background: d.color, borderRadius: 999 }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
