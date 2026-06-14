import {
  ConsoleRail, ConsoleTopBar, ConsoleStatusBar, KPICard, AiScoreCard,
} from './components/ui/primitives';
import { TrafficAreaChart, AttackDonut, GeoMap, AlertTable } from './components/ui/charts';
import AuthLogs from './components/AuthLogs';
import Icon from './components/ui/Icon';

const legendStyle = {
  display: 'inline-flex', alignItems: 'center', gap: 6,
  font: '500 10px/1 var(--font-mono)', color: 'var(--fg-3)',
  letterSpacing: '0.12em', textTransform: 'uppercase',
};

// v2.0 — Operations Console + bento. Tüm canlı veri App'teki useGuardianLive'dan props ile gelir.
export default function ModernDashboard({
  isConnected, alerts, stats, traffic, distribution,
  view, onView, onRoute, onLogout, username, role,
}) {
  const armed = stats.totalAttacks > 0;
  const justAttacked = alerts.length > 0 && (Date.now() - (alerts[0].tsMs || 0)) < 3000;

  return (
    <div className="dash-shell">
      <ConsoleRail route="dashboard" view={view} onRoute={onRoute} onView={onView} alertCount={alerts.length} username={username} role={role} />

      <div className="console-main">
        <ConsoleTopBar here="Tehdit Brifingi" crumbs={['Operasyon']} onLogout={onLogout}>
          <button className="console-topbar__util"><Icon name="search" size={13} /> Ara</button>
          <button className="console-topbar__util"><Icon name="refresh-cw" size={13} /></button>
          <button className="console-topbar__util primary"><Icon name="filter" size={13} /> Son 60 sn</button>
        </ConsoleTopBar>

        <div className="console-canvas">

          {/* Row 1 — Hero saldırı sayacı + KPI'lar + AI güven */}
          <div className="bento">
            <div className={`dcard b-c5${armed ? ' armed' : ' primary'} stat-hero`}>
              <div>
                <div className="kpi__eyebrow">{armed ? 'Aktif tehdit · tespit' : 'Bekleme · tespit yok'}</div>
                <div className="stat-hero__num">{String(stats.totalAttacks).padStart(2, '0')}</div>
                <div className="stat-hero__lede">
                  {armed
                    ? `${stats.totalAttacks} anomali işaretlendi. Operatör müdahalesi gerekebilir.`
                    : 'Ağ trafiği nominal sınırlar içinde. Sistem aktif izlemede.'}
                </div>
              </div>
              <Icon name={armed ? 'shield-alert' : 'shield-check'} size={56} color={armed ? 'var(--critical)' : 'var(--signal-400)'} className="stat-hero__icon" />
            </div>

            <div className="b-c3" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div className="dcard" style={{ flex: 1 }}>
                <KPICard
                  eyebrow="İşlenen Paket"
                  value={stats.totalTraffic.toLocaleString('tr-TR')}
                  meta={<><span className="dot dot--ok" style={{ width: 6, height: 6 }} /> Real-time</>}
                />
              </div>
              <div className="dcard" style={{ flex: 1 }}>
                <KPICard
                  eyebrow="Mod"
                  value={<span style={{ fontFamily: 'var(--font-sans)', fontSize: 22, fontWeight: 500, letterSpacing: '-0.012em' }}>AI · Sim</span>}
                  meta="Active Learning"
                />
              </div>
            </div>

            <div className="dcard b-c4"><AiScoreCard confidence={stats.lastConfidence} /></div>
          </div>

          {/* Row 2 — Trafik grafiği + donut */}
          <div className="bento">
            <div className="dcard b-c8">
              <div className="dcard__head">
                <div className="dcard__title"><Icon name="activity" size={13} color="var(--signal-400)" /> Ağ Trafiği Akışı</div>
                <div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
                  <span style={legendStyle}><span style={{ width: 7, height: 2, background: 'var(--signal-400)' }} /> Trafik</span>
                  <span style={legendStyle}><span style={{ width: 7, height: 2, background: 'var(--critical)' }} /> Saldırı</span>
                  <span className="dcard__meta">PKT/SN · 20s</span>
                </div>
              </div>
              <TrafficAreaChart data={traffic} height={210} />
            </div>

            <div className="dcard b-c4">
              <div className="dcard__head">
                <div className="dcard__title"><Icon name="pie-chart" size={13} color="var(--signal-400)" /> Tip Dağılımı</div>
                <div className="dcard__meta">{stats.totalAttacks} TESPİT</div>
              </div>
              <AttackDonut data={distribution} total={stats.totalAttacks} />
            </div>
          </div>

          {/* Row 3 — Canlı tablo + harita */}
          <div className="bento">
            <div className="b-c8" style={{ display: 'flex' }}>
              <AlertTable alerts={alerts} isConnected={isConnected} />
            </div>
            <div className="b-c4" style={{ display: 'flex' }}>
              <GeoMap active={justAttacked} />
            </div>
          </div>

          {/* Row 4 — Denetim logları */}
          <div className="bento">
            <div className="b-c12" style={{ display: 'flex' }}>
              <AuthLogs />
            </div>
          </div>

        </div>
      </div>

      <ConsoleStatusBar isConnected={isConnected} totalTraffic={stats.totalTraffic} totalAttacks={stats.totalAttacks} />
    </div>
  );
}
