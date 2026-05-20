import React, { useState } from 'react';
import axios from 'axios';
import { Shield, Lock, User, AlertCircle, ChevronRight, Briefcase } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { API_BASE } from '../config';

export default function Login({ setAuth }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE}/api/auth/login`, {
        username,
        password
      });

      const { access_token, role } = response.data;
      
      localStorage.setItem('guardian_token', access_token);
      localStorage.setItem('guardian_role', role);
      localStorage.setItem('guardian_username', username);
      
      setAuth(true);
      navigate('/'); 
      
    } catch (err) {
      if (err.response && err.response.data && err.response.data.detail) {
        setError(err.response.data.detail);
      } else {
        setError('Sunucu bağlantı hatası. Backend çalışıyor mu?');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-4 relative overflow-hidden">
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-900/20 rounded-full blur-[100px]"></div>
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-900/20 rounded-full blur-[100px]"></div>
      
      <div className="w-full max-w-md bg-slate-900/80 backdrop-blur-xl p-8 rounded-2xl border border-slate-800 shadow-2xl z-10">
        <div className="flex flex-col items-center mb-8">
          <div className="bg-gradient-to-br from-cyan-600 to-blue-700 p-4 rounded-xl shadow-lg shadow-cyan-500/20 mb-4 animate-pulse">
            <Shield className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-3xl font-bold tracking-wider text-white">GUARDIAN</h1>
          <p className="text-cyan-500 text-xs font-mono tracking-[0.3em] mt-1">Sistem Yetkilendirmesi</p>
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/50 p-3 rounded-lg mb-6 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
            <p className="text-sm text-red-400 font-medium">{error}</p>
          </div>
        )}

        <form onSubmit={handleLogin} className="space-y-6">
          {/* Kullanıcı Adı */}
          <div className="space-y-2">
            <label className="text-xs font-mono text-slate-400 uppercase tracking-wider">Kullanıcı Adı</label>
            <div className="relative">
              <User className="w-5 h-5 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full bg-slate-950 border border-slate-700 text-slate-200 text-sm rounded-lg focus:ring-2 focus:ring-cyan-500 block pl-10 p-3 transition-all"
                placeholder="admin"
                required
              />
            </div>
          </div>

          {/* Şifre (Geri Getirdik) */}
          <div className="space-y-2">
            <label className="text-xs font-mono text-slate-400 uppercase tracking-wider">Şifre</label>
            <div className="relative">
              <Lock className="w-5 h-5 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-slate-950 border border-slate-700 text-slate-200 text-sm rounded-lg focus:ring-2 focus:ring-cyan-500 block pl-10 p-3 transition-all"
                placeholder="••••••••"
                required
              />
            </div>
          </div>

          {/* Erişim Rolü */}
          <div className="space-y-2">
            <label className="text-xs font-mono text-slate-400 uppercase tracking-wider">Erişim Rolü</label>
            <div className="relative">
              <Briefcase className="w-5 h-5 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
              <select 
                className="w-full bg-slate-950 border border-slate-700 text-slate-200 text-sm rounded-lg focus:ring-2 focus:ring-cyan-500 outline-none block pl-10 p-3 transition-all appearance-none cursor-pointer"
                onChange={(e) => localStorage.setItem('selected_role_context', e.target.value)}
              >
                <option value="admin">Sistem Yöneticisi (Admin)</option>
                <option value="analyst">Güvenlik Analisti</option>
                <option value="viewer">Sadece İzleyici</option>
              </select>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-bold py-3 px-4 rounded-lg flex items-center justify-center gap-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed group"
          >
            {loading ? (
              <span className="animate-pulse">Doğrulanıyor...</span>
            ) : (
              <>
                SİSTEME GİRİŞ YAP
                <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
}