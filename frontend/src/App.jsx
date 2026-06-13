import { useState } from 'react';
import Login from './components/Login';
import ModernDashboard from './ModernDashboard';
import ClassicDashboard from './ClassicDashboard';
import IPRules from './components/IPRules';
import { useGuardianLive } from './hooks/useGuardianLive';

export default function App() {
  const [authed, setAuthed] = useState(() => !!localStorage.getItem('guardian_token'));
  const [view, setView] = useState('modern');      // 'modern' | 'classic'
  const [route, setRoute] = useState('dashboard');  // 'dashboard' | 'iprules'

  // Tek paylaşılan canlı veri kaynağı (yalnızca giriş yapılınca bağlanır).
  const live = useGuardianLive(authed);

  const username = localStorage.getItem('guardian_username') || 'admin';
  const roleRaw = localStorage.getItem('guardian_role') || '';
  const role = roleRaw === 'admin' ? 'Yönetici' : (roleRaw || 'Operatör');

  const handleLogout = () => {
    localStorage.removeItem('guardian_token');
    localStorage.removeItem('guardian_role');
    localStorage.removeItem('guardian_username');
    setAuthed(false);
    setRoute('dashboard');
    setView('modern');
  };

  if (!authed) return <Login onAuthed={() => setAuthed(true)} />;

  // v1.0 (klasik) konsol kabuğunu atlar; kendi açık temalı düzeni vardır.
  if (view === 'classic') {
    return (
      <ClassicDashboard
        alerts={live.alerts}
        stats={live.stats}
        distribution={live.distribution}
        isConnected={live.isConnected}
        onView={setView}
        onLogout={handleLogout}
      />
    );
  }

  const shellProps = {
    view, onView: setView, onRoute: setRoute, onLogout: handleLogout,
    username, role, ...live,
  };

  if (route === 'iprules') return <IPRules {...shellProps} />;
  return <ModernDashboard {...shellProps} />;
}
