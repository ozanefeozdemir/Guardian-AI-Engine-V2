import React from 'react';
import Icon from './Icon';

// =========================================================================
// Marka kilidi
// =========================================================================
export const BrandMark = ({ size = 'standard' }) => {
  const t = {
    compact: { tile: 28, icon: 14, word: 14 },
    standard: { tile: 40, icon: 22, word: 20 },
    splash: { tile: 64, icon: 34, word: 28 },
  }[size];
  const stacked = size === 'splash';
  return (
    <div style={{
      display: 'flex', alignItems: 'center',
      flexDirection: stacked ? 'column' : 'row',
      gap: stacked ? 16 : 12,
      animation: stacked ? 'pulse 2.4s ease-in-out infinite' : 'none',
    }}>
      <div className="brand-tile" style={{ width: t.tile, height: t.tile }}>
        <Icon name="shield" size={t.icon} strokeWidth={2.2} />
      </div>
      <div style={{ lineHeight: 1, textAlign: stacked ? 'center' : 'left' }}>
        <div className="brand-word" style={{ fontSize: t.word, letterSpacing: stacked ? '0.12em' : '0.08em' }}>GUARDIAN</div>
        {size !== 'compact' && <div className="brand-sub" style={{ marginTop: 6 }}>AI Engine V2</div>}
      </div>
    </div>
  );
};

// KPI iç içeriği — kart sarmalayıcı (.dcard) layout tarafından sağlanır.
export const KPICard = ({ eyebrow, value, meta, metaColor = 'var(--info)' }) => (
  <div className="kpi">
    <div className="kpi__eyebrow">{eyebrow}</div>
    <div className="kpi__val">{value}</div>
    {meta != null && <div className="kpi__meta" style={{ color: metaColor }}>{meta}</div>}
  </div>
);

// AI güven yarım daire gauge — iç içerik (.dcard layout'tan gelir).
export const AiScoreCard = ({ confidence = 0 }) => {
  const pct = Math.round((confidence || 0) * 100);
  const angle = (pct / 100) * 180;
  const isHigh = pct > 80;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: 96 }}>
      <div className="kpi__eyebrow">AI Güven · Son Saldırı</div>
      <svg viewBox="0 0 100 60" style={{ width: 110, height: 66, marginTop: 8 }}>
        <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="var(--ink-5)" strokeWidth="8" strokeLinecap="round" />
        <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none"
              stroke={isHigh ? 'var(--critical)' : 'var(--signal-400)'}
              strokeWidth="8" strokeLinecap="round"
              strokeDasharray={`${(angle / 180) * 126} 126`} />
      </svg>
      <div style={{ marginTop: -8, font: '600 22px/1 var(--font-mono)', color: 'white', fontVariantNumeric: 'tabular-nums' }}>%{pct}</div>
      <div style={{ font: '400 10px/1 var(--font-mono)', color: 'var(--fg-4)', marginTop: 6, letterSpacing: '0.1em', textTransform: 'uppercase' }}>Tespit Kesinliği</div>
    </div>
  );
};

// Risk rozeti — handoff eşikleri (Bölüm 4.2)
export const RiskBadge = ({ confidence, listType }) => {
  if (listType === 'whitelist') return <span className="risk risk--safe">GÜVENLİ LİSTE</span>;
  if (listType === 'blacklist') return <span className="risk risk--block">YASAKLI IP <Icon name="shield-x" size={10} /></span>;
  if (confidence > 0.8) return <span className="risk risk--critical">KRİTİK</span>;
  if (confidence > 0.5) return <span className="risk risk--high">YÜKSEK</span>;
  if (confidence > 0.3) return <span className="risk risk--medium">ORTA</span>;
  return <span className="risk risk--low">DÜŞÜK</span>;
};

export const ConnectionPill = ({ isConnected }) => (
  <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
    <span className={`dot ${isConnected ? 'dot--ok' : 'dot--crit'}`} />
    <span style={{ font: '500 11px/1 var(--font-mono)', color: 'var(--fg-3)', letterSpacing: '0.08em' }}>
      {isConnected ? 'LIVE FEED' : 'OFFLINE'}
    </span>
  </div>
);

export const Field = ({ label, icon, ...inputProps }) => (
  <div className="field">
    <label className="field__label">{label}</label>
    <div className="input-wrap">
      {icon && <Icon name={icon} size={16} className="input-wrap__icon" />}
      <input className={`input ${!icon ? 'no-icon' : ''}`} {...inputProps} />
    </div>
  </div>
);

// =========================================================================
// Console kabuğu — sol rail + üst breadcrumb bar + alt durum çubuğu
// =========================================================================
export const ConsoleRail = ({ route, view, onRoute, onView, alertCount = 0, username = 'admin', role = 'Yönetici' }) => {
  const initials = (username || 'AD').slice(0, 2).toUpperCase();
  return (
    <aside className="console-rail">
      <div className="console-rail__brand">
        <div className="console-rail__brand-tile"><Icon name="shield" size={14} strokeWidth={2.4} /></div>
        <div>
          <div className="console-rail__brand-word">Guardian</div>
          <div className="console-rail__brand-sub">AI Engine V2</div>
        </div>
      </div>

      <div className="console-rail__section">Operasyon</div>
      <button className={`console-rail__item ${route === 'dashboard' ? 'active' : ''}`} onClick={() => onRoute('dashboard')}>
        <Icon name="layout-dashboard" size={15} /> Tehdit Brifingi
      </button>
      <button className={`console-rail__item ${alertCount > 0 ? 'armed' : ''}`} onClick={() => onRoute('dashboard')}>
        <Icon name="alert-triangle" size={15} /> Canlı Uyarılar
        {alertCount > 0 && <span className="badge-count">{alertCount}</span>}
      </button>
      <button className="console-rail__item"><Icon name="globe" size={15} /> Tehdit Haritası</button>

      <div className="console-rail__section">Politika</div>
      <button className={`console-rail__item ${route === 'iprules' ? 'active' : ''}`} onClick={() => onRoute('iprules')}>
        <Icon name="shield-check" size={15} /> IP Kuralları
      </button>
      <button className="console-rail__item"><Icon name="users" size={15} /> Roller & Erişim</button>
      <button className="console-rail__item"><Icon name="history" size={15} /> Denetim Logları</button>

      <div className="console-rail__section">Sistem</div>
      <button className="console-rail__item"><Icon name="server" size={15} /> Motor Durumu</button>
      <button className="console-rail__item"><Icon name="settings" size={15} /> Ayarlar</button>

      <div style={{ marginTop: 'auto', padding: '12px 12px 0', borderTop: '1px solid var(--ink-4)' }}>
        <div className="console-rail__section" style={{ padding: '0 8px 8px' }}>Görünüm</div>
        <div style={{ display: 'flex', background: 'var(--ink-2)', border: '1px solid var(--ink-5)', borderRadius: 5, padding: 2, margin: '0 8px 12px' }}>
          <button onClick={() => onView('modern')} style={{
            flex: 1, padding: '6px 8px', border: 0, borderRadius: 3,
            background: view === 'modern' ? 'var(--signal-700)' : 'transparent',
            color: view === 'modern' ? 'var(--signal-200)' : 'var(--fg-3)',
            font: '500 10px/1 var(--font-mono)', letterSpacing: '0.10em', textTransform: 'uppercase',
          }}>v2.0</button>
          <button onClick={() => onView('classic')} style={{
            flex: 1, padding: '6px 8px', border: 0, borderRadius: 3,
            background: view === 'classic' ? 'var(--ink-4)' : 'transparent',
            color: view === 'classic' ? 'var(--fg-0)' : 'var(--fg-3)',
            font: '500 10px/1 var(--font-mono)', letterSpacing: '0.10em', textTransform: 'uppercase',
          }}>v1.0</button>
        </div>
      </div>

      <div className="console-rail__foot">
        <div className="console-rail__user">
          <div className="console-rail__user-avatar">{initials}</div>
          <div className="console-rail__user-meta">
            <div className="console-rail__user-name">{username}</div>
            <div className="console-rail__user-role">{role}</div>
          </div>
        </div>
      </div>
    </aside>
  );
};

export const ConsoleTopBar = ({ here, crumbs = [], onLogout, children }) => (
  <div className="console-topbar">
    <div className="console-topbar__crumb">
      {crumbs.map((c, i) => (
        <React.Fragment key={i}><span>{c}</span><span className="sep">/</span></React.Fragment>
      ))}
      <span className="here">{here}</span>
    </div>
    <div className="console-topbar__spacer" />
    {children}
    <button className="console-topbar__util" onClick={onLogout}>
      <Icon name="log-out" size={13} /> Çıkış
    </button>
  </div>
);

export const ConsoleStatusBar = ({ isConnected, totalTraffic = 0, totalAttacks = 0 }) => (
  <footer className="console-statusbar">
    <span className="console-statusbar__seg">
      <span className={`dot ${isConnected ? 'dot--ok' : 'dot--crit'}`} />
      <span className="v">{isConnected ? 'WS · ONLINE' : 'WS · OFFLINE'}</span>
    </span>
    <span className="console-statusbar__sep">│</span>
    <span className="console-statusbar__seg">Engine · <span className="v">Random Forest 2017/2018</span></span>
    <span className="console-statusbar__sep">│</span>
    <span className="console-statusbar__seg">Pkt · <span className="v">{Number(totalTraffic).toLocaleString('tr-TR')}</span></span>
    <span className="console-statusbar__sep">│</span>
    <span className="console-statusbar__seg">Atk · <span className="v" style={{ color: totalAttacks > 0 ? 'var(--critical)' : 'var(--fg-1)' }}>{totalAttacks}</span></span>
    <span style={{ marginLeft: 'auto' }} className="console-statusbar__seg">host · <span className="v">guardian-prod-01</span></span>
    <span className="console-statusbar__sep">│</span>
    <span className="console-statusbar__seg">v2.4.1 · build 7c8a3f</span>
  </footer>
);
