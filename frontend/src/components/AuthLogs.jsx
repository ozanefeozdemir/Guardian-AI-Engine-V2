import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { ShieldAlert, UserX, UserCheck, RefreshCw } from 'lucide-react';

export default function AuthLogs() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const response = await axios.get('http://localhost:8000/api/auth/logs');
      setLogs(response.data);
    } catch (error) {
      console.error('Loglar çekilemedi', error);
    } finally {
      setLoading(false);
    }
  };

  // Sayfa açıldığında logları çek
  useEffect(() => {
    fetchLogs();
  }, []);

  const getActionIcon = (action) => {
    if (action.includes('Başarılı')) return <UserCheck className="w-5 h-5 text-green-500" />;
    if (action.includes('Yanlış') || action.includes('BAŞARISIZ')) return <UserX className="w-5 h-5 text-red-500" />;
    return <ShieldAlert className="w-5 h-5 text-yellow-500" />;
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl mt-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <ShieldAlert className="text-cyan-500" />
          Sistem Denetim Logları (Audit Logs)
        </h2>
        <button onClick={fetchLogs} className="p-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 transition-colors">
          <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>
      
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm text-slate-300">
          <thead className="bg-slate-800/50 text-slate-400 font-mono text-xs uppercase">
            <tr>
              <th className="p-4 rounded-tl-lg">Eylem / Durum</th>
              <th className="p-4">Kullanıcı Adı</th>
              <th className="p-4">IP Adresi</th>
              <th className="p-4">Tarih / Saat</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/50">
            {logs.map((log) => (
              <tr key={log.id} className="hover:bg-slate-800/20 transition-colors">
                <td className="p-4 flex items-center gap-3">
                  {getActionIcon(log.action)}
                  <span className={log.action.includes('BAŞARISIZ') ? 'text-red-400 font-bold' : 'text-slate-300'}>
                    {log.action}
                  </span>
                </td>
                <td className="p-4 font-medium text-cyan-400">{log.username}</td>
                <td className="p-4 font-mono text-xs text-slate-500">{log.ip_address}</td>
                <td className="p-4 font-mono text-xs text-slate-400">
                  {new Date(log.timestamp).toLocaleString('tr-TR')}
                </td>
              </tr>
            ))}
            {logs.length === 0 && (
              <tr>
                <td colSpan="4" className="p-8 text-center text-slate-500 italic">Henüz kayıtlı bir denetim logu bulunmuyor.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}