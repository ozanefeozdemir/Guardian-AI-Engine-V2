import Icon from './Icon';
import { RiskBadge, ConnectionPill } from './primitives';

// Trafik alan grafiği — iki seri (trafik + saldırı). data = [{name, traffic, attack}].
export const TrafficAreaChart = ({ data, width = 800, height = 220 }) => {
  if (!data || data.length === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--fg-4)', font: '400 12px/1 var(--font-mono)' }}>
        Akış başlatılıyor...
      </div>
    );
  }

  const maxVal = Math.max(...data.flatMap((d) => [d.traffic, d.attack]), 5);
  const stepX = data.length > 1 ? width / (data.length - 1) : width;

  const buildPath = (key) => {
    const pts = data.map((d, i) => [i * stepX, height - (d[key] / maxVal) * (height - 30)]);
    const line = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p[0]} ${p[1]}`).join(' ');
    const area = `${line} L ${width} ${height} L 0 ${height} Z`;
    return { line, area };
  };
  const t = buildPath('traffic');
  const a = buildPath('attack');

  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ width: '100%', height, display: 'block' }}>
      <defs>
        <linearGradient id="g-traffic" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#43B5C0" stopOpacity="0.30" />
          <stop offset="100%" stopColor="#43B5C0" stopOpacity="0" />
        </linearGradient>
        <linearGradient id="g-attack" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#FF5A6B" stopOpacity="0.30" />
          <stop offset="100%" stopColor="#FF5A6B" stopOpacity="0" />
        </linearGradient>
      </defs>

      {[0.25, 0.5, 0.75].map((p) => (
        <line key={p} x1="0" y1={height * p} x2={width} y2={height * p} stroke="#232B3D" strokeDasharray="2 6" strokeWidth="1" />
      ))}

      <path d={t.area} fill="url(#g-traffic)" />
      <path d={t.line} fill="none" stroke="#43B5C0" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d={a.area} fill="url(#g-attack)" />
      <path d={a.line} fill="none" stroke="#FF5A6B" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
};

// Saldırı tipi dağılımı — donut. data = [{name, value}].
export const AttackDonut = ({ data, total = 0 }) => {
  const COLORS = ['#FF5A6B', '#F08A4B', '#F5B549', '#43B5C0', '#6DA6FF', '#A78BFA'];
  const cleaned = data && data.length ? data : [{ name: 'Bekleniyor', value: 1, ph: true }];
  const sum = cleaned.reduce((acc, b) => acc + b.value, 0) || 1;
  const radius = 70, inner = 50, cx = 100, cy = 100;
  let acc = -Math.PI / 2;

  const arc = (start, end) => {
    const x1 = cx + radius * Math.cos(start), y1 = cy + radius * Math.sin(start);
    const x2 = cx + radius * Math.cos(end), y2 = cy + radius * Math.sin(end);
    const x3 = cx + inner * Math.cos(end), y3 = cy + inner * Math.sin(end);
    const x4 = cx + inner * Math.cos(start), y4 = cy + inner * Math.sin(start);
    const large = end - start > Math.PI ? 1 : 0;
    return `M ${x1} ${y1} A ${radius} ${radius} 0 ${large} 1 ${x2} ${y2} L ${x3} ${y3} A ${inner} ${inner} 0 ${large} 0 ${x4} ${y4} Z`;
  };

  return (
    <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14 }}>
      <svg viewBox="0 0 200 200" style={{ width: 180, height: 180 }}>
        {cleaned.map((d, i) => {
          const sweep = (d.value / sum) * Math.PI * 2 * 0.96;
          const path = arc(acc, acc + sweep);
          acc += sweep + (Math.PI * 2 * 0.04) / cleaned.length;
          return <path key={i} d={path} fill={d.ph ? '#232C3E' : COLORS[i % COLORS.length]} />;
        })}
      </svg>
      <div style={{ position: 'absolute', top: 60, left: 0, right: 0, textAlign: 'center', pointerEvents: 'none' }}>
        <div style={{ font: '600 28px/1 var(--font-mono)', color: 'white', fontVariantNumeric: 'tabular-nums' }}>{total}</div>
        <div style={{ font: '500 10px/1 var(--font-mono)', color: 'var(--fg-3)', marginTop: 4, letterSpacing: '0.15em', textTransform: 'uppercase' }}>Tespit</div>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, justifyContent: 'center' }}>
        {cleaned.filter((d) => !d.ph).map((d, i) => (
          <div key={d.name} style={{ display: 'inline-flex', alignItems: 'center', gap: 6, font: '400 11px/1 var(--font-sans)', color: 'var(--fg-2)' }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: COLORS[i % COLORS.length] }} />
            {d.name}
          </div>
        ))}
      </div>
    </div>
  );
};

// Tehdit haritası — saldırı anında ping animasyonlu noktalar.
export const GeoMap = ({ active }) => (
  <div className="dcard" style={{ minHeight: 280, position: 'relative', overflow: 'hidden', padding: 16, display: 'flex', flexDirection: 'column' }}>
    <div className="dcard__title" style={{ position: 'relative', zIndex: 2 }}>
      <Icon name="globe" size={14} color={active ? 'var(--critical)' : 'var(--fg-4)'} style={active ? { animation: 'pulse 1.6s ease-in-out infinite' } : {}} />
      <span style={{ color: active ? 'var(--critical)' : 'var(--fg-3)', font: '500 11px/1 var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.12em' }}>
        {active ? 'Saldırı Tespit Edildi' : 'Canlı Tehdit Haritası'}
      </span>
    </div>
    <svg viewBox="0 0 200 100" preserveAspectRatio="xMidYMid meet" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: 0.20, color: 'var(--fg-4)' }}>
      <g fill="currentColor">
        <path d="M14,32 Q22,28 30,32 L36,30 L42,33 L48,30 L55,33 L62,30 L70,32 L78,30 L82,34 L86,30 L90,33 L94,30 L98,32 L102,30 L108,33 L114,30 L120,32 L126,29 L132,33 L138,30 L144,34 L150,30 L156,33 L162,30 L168,32 L174,29 L180,33 L186,30 L192,32 L186,40 L180,38 L174,42 L168,40 L162,43 L156,40 L150,44 L144,40 L138,43 L132,40 L126,44 L120,40 L114,43 L108,40 L102,44 L98,40 L94,43 L90,40 L86,44 L82,40 L78,43 L70,40 L62,44 L55,40 L48,43 L42,40 L36,44 L30,40 L22,43 L14,40 Z" />
        <path d="M30,52 L40,50 L52,53 L64,51 L78,54 L92,52 L108,55 L124,52 L142,55 L160,52 L178,54 L178,62 L160,60 L142,63 L124,60 L108,63 L92,60 L78,62 L64,59 L52,61 L40,58 L30,60 Z" />
      </g>
    </svg>
    {active && (
      <>
        <div style={{ position: 'absolute', top: '38%', left: '22%', width: 12, height: 12, borderRadius: '50%', background: 'var(--critical)', boxShadow: '0 0 14px var(--critical)' }} />
        <div style={{ position: 'absolute', top: '38%', left: '22%', width: 12, height: 12, borderRadius: '50%', background: 'var(--critical)', animation: 'ping 1.6s cubic-bezier(0,0,0.2,1) infinite' }} />
        <div style={{ position: 'absolute', top: '54%', left: '64%', width: 10, height: 10, borderRadius: '50%', background: 'var(--critical)', boxShadow: '0 0 12px var(--critical)' }} />
        <div style={{ position: 'absolute', top: '54%', left: '64%', width: 10, height: 10, borderRadius: '50%', background: 'var(--critical)', animation: 'ping 1.6s cubic-bezier(0,0,0.2,1) infinite', animationDelay: '0.3s' }} />
      </>
    )}
    <div style={{ position: 'relative', zIndex: 2, marginTop: 'auto', font: '400 10px/1.4 var(--font-mono)', color: 'var(--fg-4)', paddingTop: 80 }}>
      {active ? <>İzleniyor: 2 düğüm · konum kestirimi devam ediyor</> : <>Düğüm sayısı: 254 · son güncelleme şimdi</>}
    </div>
  </div>
);

// Canlı saldırı tablosu. alerts = [{id, time, type, confidence, src, dest, list_type}].
export const AlertTable = ({ alerts, isConnected }) => (
  <div className="dcard" style={{ padding: 0, display: 'flex', flexDirection: 'column', minHeight: 320, overflow: 'hidden', width: '100%' }}>
    <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--ink-4)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <div className="dcard__title">
        <Icon name="alert-triangle" size={16} color="var(--critical)" />
        Canlı Saldırı Akışı
      </div>
      <ConnectionPill isConnected={isConnected} />
    </div>
    <div style={{ overflow: 'auto', flex: 1 }}>
      <table className="table">
        <thead>
          <tr>
            <th>Zaman</th>
            <th>Risk / Güven</th>
            <th>Saldırı Tipi</th>
            <th>Kaynak IP</th>
            <th>Hedef Port</th>
          </tr>
        </thead>
        <tbody>
          {alerts.map((a) => (
            <tr key={a.id}>
              <td className="t-time">{a.time}</td>
              <td><RiskBadge confidence={a.confidence} listType={a.list_type} /></td>
              <td className="t-type">{a.type}</td>
              <td className="t-ip">{a.src}</td>
              <td className="t-port">:{a.dest}</td>
            </tr>
          ))}
          {alerts.length === 0 && (
            <tr><td colSpan={5} style={{ padding: 40, textAlign: 'center', color: 'var(--fg-4)', fontStyle: 'italic' }}>
              Henüz bir saldırı tespit edilmedi. Sistem izlemede...
            </td></tr>
          )}
        </tbody>
      </table>
    </div>
  </div>
);
