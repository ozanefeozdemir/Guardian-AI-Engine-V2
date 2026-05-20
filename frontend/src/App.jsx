import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate, Link } from 'react-router-dom';
import ClassicDashboard from './ClassicDashboard';
import ModernDashboard from './ModernDashboard';
import Login from './components/Login';
import IPRules from './components/IPRules'; // Dosyanın components içinde olduğundan eminiz

export default function App() {
  const [view, setView] = useState('modern');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('guardian_token');
    if (token) {
      setIsAuthenticated(true);
    }
    setLoading(false);
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('guardian_token');
    localStorage.removeItem('guardian_role');
    localStorage.removeItem('guardian_username');
    setIsAuthenticated(false);
  };

  if (loading) {
    return <div className="min-h-screen bg-slate-950 flex items-center justify-center text-cyan-500 font-mono italic">INITIALIZING GUARDIAN CORE...</div>;
  }

  return (
    <BrowserRouter>
      <Routes>
        {/* Giriş Sayfası */}
        <Route
          path="/login"
          element={!isAuthenticated ? <Login setAuth={setIsAuthenticated} /> : <Navigate to="/" replace />}
        />

        {/* Ana Dashboard (Korumalı) */}
        <Route
          path="/"
          element={
            isAuthenticated ? (
              <div className="relative">
                {/* Çıkış Butonu */}
                <button
                  onClick={handleLogout}
                  className="fixed top-4 right-4 z-[9999] bg-red-600/80 hover:bg-red-500 text-white px-4 py-2 rounded-lg font-bold text-sm backdrop-blur border border-red-500/50 transition-all shadow-lg"
                >
                  Çıkış Yap
                </button>

                {/* ALT PANEL (Switcher + IP Rules Kapısı) */}
                <div className="fixed bottom-4 right-4 z-[9999] flex gap-2 bg-black/80 p-2 rounded-full border border-gray-700 shadow-2xl backdrop-blur-sm">
                  <button
                    onClick={() => setView('classic')}
                    className={`px-4 py-2 rounded-full text-sm font-bold transition-all ${view === 'classic' ? 'bg-blue-600 text-white shadow-lg' : 'text-gray-400 hover:text-white'}`}
                  >
                    v1.0 (Klasik)
                  </button>
                  <button
                    onClick={() => setView('modern')}
                    className={`px-4 py-2 rounded-full text-sm font-bold transition-all ${view === 'modern' ? 'bg-cyan-600 text-white shadow-lg' : 'text-gray-400 hover:text-white'}`}
                  >
                    v2.0 (Modern)
                  </button>
                  
                  {/* İŞTE O EKSİK BUTON BURADA */}
                  <Link
                    to="/ip-rules"
                    className="px-4 py-2 rounded-full text-sm font-bold bg-slate-800 text-slate-300 hover:bg-slate-700 transition-all border border-slate-600 flex items-center"
                  >
                    IP Kuralları →
                  </Link>
                </div>

                {view === 'classic' ? <ClassicDashboard /> : <ModernDashboard />}
              </div>
            ) : (
              <Navigate to="/login" replace />
            )
          }
        />

        {/* IP Rules Sayfası Rotası */}
        <Route
          path="/ip-rules"
          element={
            isAuthenticated ? (
              <div className="relative">
                <div className="fixed top-4 right-4 z-[9999] flex gap-2">
                  <Link 
                    to="/" 
                    className="bg-slate-800 hover:bg-slate-700 text-white px-4 py-2 rounded-lg font-bold text-sm border border-slate-600 transition-all shadow-xl backdrop-blur-md"
                  >
                    ← Dashboard'a Dön
                  </Link>
                  <button
                    onClick={handleLogout}
                    className="bg-red-600/80 hover:bg-red-500 text-white px-4 py-2 rounded-lg font-bold text-sm border border-red-500/50 transition-all"
                  >
                    Çıkış Yap
                  </button>
                </div>
                <IPRules />
              </div>
            ) : (
              <Navigate to="/login" replace />
            )
          }
        />
      </Routes>
    </BrowserRouter>
  );
}