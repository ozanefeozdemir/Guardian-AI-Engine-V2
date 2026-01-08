import React, { useState } from 'react';
import ClassicDashboard from './ClassicDashboard';
import ModernDashboard from './ModernDashboard';
import { Layout } from 'lucide-react';

export default function App() {
  const [view, setView] = useState('modern'); // Varsayılan olarak yeni açılır

  return (
    <div className="relative">
      {/* Geçiş Butonu (Ekranın sağ altında sabit durur) */}
      <div className="fixed bottom-4 right-4 z-[9999] flex gap-2 bg-black/80 p-2 rounded-full border border-gray-700 shadow-2xl backdrop-blur-sm">
        <button
          onClick={() => setView('classic')}
          className={`px-4 py-2 rounded-full text-sm font-bold transition-all ${
            view === 'classic' 
              ? 'bg-blue-600 text-white shadow-lg' 
              : 'text-gray-400 hover:text-white'
          }`}
        >
          v1.0 (Klasik)
        </button>
        <button
          onClick={() => setView('modern')}
          className={`px-4 py-2 rounded-full text-sm font-bold transition-all ${
            view === 'modern' 
              ? 'bg-cyan-600 text-white shadow-lg' 
              : 'text-gray-400 hover:text-white'
          }`}
        >
          v2.0 (Modern)
        </button>
      </div>

      {/* Seçili Dashboard'u Render Et */}
      {view === 'classic' ? <ClassicDashboard /> : <ModernDashboard />}
    </div>
  );
}