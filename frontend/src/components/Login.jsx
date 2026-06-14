import { useState } from 'react';
import axios from 'axios';
import { API_BASE } from '../config';
import { BrandMark, Field } from './ui/primitives';
import Icon from './ui/Icon';

// Görsel: Operations Console login. Veri akışı KORUNDU:
// axios POST /api/auth/login → { access_token, role }, localStorage persist.
export default function Login({ onAuthed }) {
  const [username, setUsername] = useState('admin');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setError('');
    if (!password) { setError('Şifre alanı zorunludur.'); return; }
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/api/auth/login`, { username, password });
      const { access_token, role } = res.data;
      localStorage.setItem('guardian_token', access_token);
      localStorage.setItem('guardian_role', role || '');
      localStorage.setItem('guardian_username', username);
      onAuthed();
    } catch (err) {
      setError(err.response?.data?.detail || 'Sunucu bağlantı hatası. Backend çalışıyor mu?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', background: 'var(--ink-0)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16, position: 'relative', overflow: 'hidden' }}>
      <div style={{ position: 'absolute', top: '20%', left: '18%', width: 380, height: 380, background: 'rgba(67,181,192,0.16)', borderRadius: '50%', filter: 'blur(110px)' }} />
      <div style={{ position: 'absolute', bottom: '15%', right: '15%', width: 380, height: 380, background: 'rgba(31,42,82,0.40)', borderRadius: '50%', filter: 'blur(110px)' }} />

      <form onSubmit={submit} style={{ width: '100%', maxWidth: 440, padding: 36, background: 'rgba(11,16,26,0.85)', backdropFilter: 'blur(20px)', borderRadius: 'var(--radius-xl)', border: '1px solid var(--ink-5)', boxShadow: '0 24px 64px -12px rgba(0,0,0,0.8)', position: 'relative', zIndex: 1 }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginBottom: 32 }}>
          <BrandMark size="splash" />
          <div style={{ marginTop: 18, font: '500 10px/1 var(--font-mono)', letterSpacing: '0.3em', color: 'var(--signal-400)', textTransform: 'uppercase' }}>
            Sistem Yetkilendirmesi
          </div>
        </div>

        {error && (
          <div style={{ padding: 12, marginBottom: 20, background: 'rgba(255,90,107,0.08)', border: '1px solid rgba(255,90,107,0.4)', borderRadius: 8, display: 'flex', gap: 10, alignItems: 'flex-start' }}>
            <Icon name="alert-circle" size={16} color="var(--critical)" style={{ marginTop: 1, flexShrink: 0 }} />
            <span style={{ font: '400 12px/1.4 var(--font-sans)', color: '#FF7A88' }}>{error}</span>
          </div>
        )}

        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          <Field label="Kullanıcı Adı" icon="user" value={username} onChange={(e) => setUsername(e.target.value)} placeholder="admin" />
          <Field label="Şifre" icon="lock" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••••" />
          <div className="field">
            <label className="field__label">Erişim Rolü</label>
            <div className="input-wrap">
              <Icon name="briefcase" size={16} className="input-wrap__icon" />
              <select
                className="input"
                defaultValue="admin"
                onChange={(e) => localStorage.setItem('selected_role_context', e.target.value)}
              >
                <option value="admin">Sistem Yöneticisi (Admin)</option>
                <option value="analyst">Güvenlik Analisti</option>
                <option value="viewer">Sadece İzleyici</option>
              </select>
            </div>
          </div>
        </div>

        <button type="submit" disabled={loading} className="btn btn--primary" style={{ width: '100%', justifyContent: 'center', padding: '14px 18px', marginTop: 24, opacity: loading ? 0.6 : 1 }}>
          {loading ? 'Doğrulanıyor...' : <>SİSTEME GİRİŞ YAP <Icon name="chevron-right" size={16} /></>}
        </button>
      </form>
    </div>
  );
}
