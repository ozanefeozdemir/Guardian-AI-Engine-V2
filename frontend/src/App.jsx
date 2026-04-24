import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import ClassicDashboard from './ClassicDashboard';
import ModernDashboard from './ModernDashboard';
import Login from './components/Login'; // Yeni oluşturduğumuz login sayfası

export default function App() {
  const [view, setView] = useState('modern');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  // Sayfa ilk yüklendiğinde token var mı diye kontrol et
  useEffect(() => {
    const token = localStorage.getItem('guardian_token');
    if (token) {
      setIsAuthenticated(true);
    }
    setLoading(false);
  }, []);

  // Kullanıcı Çıkış Yapma Fonksiyonu
  const handleLogout = () => {
    localStorage.removeItem('guardian_token');
    localStorage.removeItem('guardian_role');
    localStorage.removeItem('guardian_username');
    setIsAuthenticated(false);
  };

  if (loading) {
    return <div className="min-h-screen bg-slate-950 flex items-center justify-center text-cyan-500">Yükleniyor...</div>;
  }

  return (
    <BrowserRouter>
      <Routes>
        {/* Giriş Sayfası Rotası */}
        <Route
          path="/login"
          element={
            !isAuthenticated ? (
              <Login setAuth={setIsAuthenticated} />
            ) : (
              <Navigate to="/" replace />
            )
          }
        />

        {/* Dashboard Rotası (Korumalı) */}
        <Route
          path="/"
          element={
            isAuthenticated ? (
              <div className="relative">
                {/* --- Çıkış Yap Butonu (Header'a eklenecek ama şimdilik burada) --- */}
                <button
                  onClick={handleLogout}
                  className="fixed top-4 right-1 z-[9999] bg-red-600/80 hover:bg-red-500 text-white px-4 py-2 rounded-lg font-bold text-sm backdrop-blur border border-red-500/50 transition-all"
                >
                  Çıkış Yap
                </button>

                {/* Mevcut View Switcher'ın */}
                <div className="fixed bottom-4 right-4 z-[9999] flex gap-2 bg-black/80 p-2 rounded-full border border-gray-700 shadow-2xl backdrop-blur-sm">
                  <button
                    onClick={() => setView('classic')}
                    className={`px-4 py-2 rounded-full text-sm font-bold transition-all ${view === 'classic'
                      ? 'bg-blue-600 text-white shadow-lg'
                      : 'text-gray-400 hover:text-white'
                      }`}
                  >
                    v1.0 (Klasik)
                  </button>
                  <button
                    onClick={() => setView('modern')}
                    className={`px-4 py-2 rounded-full text-sm font-bold transition-all ${view === 'modern'
                      ? 'bg-cyan-600 text-white shadow-lg'
                      : 'text-gray-400 hover:text-white'
                      }`}
                  >
                    v2.0 (Modern)
                  </button>
                </div>

                {view === 'classic' ? <ClassicDashboard /> : <ModernDashboard />}
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