import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { ShieldCheck, ShieldX, Trash2, Plus, Clock, Terminal } from 'lucide-react';
import { API_BASE } from '../config';

export default function IPRules() {
  const [rules, setRules] = useState([]);
  const [activeTab, setActiveTab] = useState('whitelist');
  const [newRule, setNewRule] = useState({ cidr: '', reason: '', expires_at: '' });
  const token = localStorage.getItem('guardian_token');

  const fetchRules = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/ip-rules?list_type=${activeTab}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setRules(res.data);
    } catch (err) { console.error("Kurallar çekilemedi", err); }
  };

  useEffect(() => { fetchRules(); }, [activeTab]);

  const handleAdd = async (e) => {
    e.preventDefault();
    
    // Backend'in beklediği temiz veriyi hazırlıyoruz
    const payload = {
      cidr: newRule.cidr.trim(),
      list_type: activeTab,
      reason: newRule.reason || null, // Boşsa null gönder
      expires_at: newRule.expires_at === "" ? null : newRule.expires_at // Boş metni null yap
    };

    try {
      const token = localStorage.getItem('guardian_token');
      await axios.post(`${API_BASE}/api/ip-rules`, payload, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      setNewRule({ cidr: '', reason: '', expires_at: '' });
      fetchRules();
    } catch (err) {
      // [object Object] hatasını önlemek için hatayı detaylı yazdırıyoruz
      const errorMsg = err.response?.data?.detail;
      console.error("422 Hata Detayı:", errorMsg);
      
      alert(typeof errorMsg === 'object' 
        ? "Format Hatası: Verileri kontrol edin (Örn: IP/32)" 
        : (errorMsg || "Kural eklenemedi"));
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm("Bu kuralı silmek istediğinize emin misiniz?")) return;
    try {
      await axios.delete(`${API_BASE}/api/ip-rules/${id}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      fetchRules();
    } catch (err) { console.error("Silme hatası", err); }
  };

  return (
    <div className="min-h-screen bg-slate-900 p-8 text-slate-200">
      <div className="max-w-5xl mx-auto space-y-8">
        <header className="flex justify-between items-end">
          <div>
            <h1 className="text-3xl font-bold text-white tracking-tight">IP Filtreleme Politikaları</h1>
            <p className="text-slate-400 text-sm mt-1">Ağ trafiği kurallarını CIDR formatında yönetin.</p>
          </div>
          <div className="flex bg-slate-800 p-1 rounded-xl border border-slate-700">
            <button onClick={() => setActiveTab('whitelist')} className={`px-6 py-2 rounded-lg text-sm font-bold transition-all ${activeTab === 'whitelist' ? 'bg-green-600 text-white shadow-lg' : 'text-slate-400 hover:text-white'}`}>Güvenli Liste</button>
            <button onClick={() => setActiveTab('blacklist')} className={`px-6 py-2 rounded-lg text-sm font-bold transition-all ${activeTab === 'blacklist' ? 'bg-red-600 text-white shadow-lg' : 'text-slate-400 hover:text-white'}`}>Yasaklı Liste</button>
          </div>
        </header>

        {/* Ekleme Formu */}
        <form onSubmit={handleAdd} className="bg-slate-800/50 p-6 rounded-2xl border border-slate-700 grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
          <div className="space-y-2">
            <label className="text-xs font-mono uppercase text-slate-500">IP / CIDR</label>
            <input required placeholder="192.168.1.0/24" className="w-full bg-slate-900 border border-slate-700 p-2.5 rounded-lg text-sm" 
              value={newRule.cidr} onChange={e => setNewRule({...newRule, cidr: e.target.value})} />
          </div>
          <div className="space-y-2 md:col-span-2">
            <label className="text-xs font-mono uppercase text-slate-500">Açıklama</label>
            <input placeholder="Güvenilir iç ağ trafiği..." className="w-full bg-slate-900 border border-slate-700 p-2.5 rounded-lg text-sm" 
              value={newRule.reason} onChange={e => setNewRule({...newRule, reason: e.target.value})} />
          </div>
          <button type="submit" className="bg-cyan-600 hover:bg-cyan-500 text-white font-bold py-2.5 rounded-lg flex items-center justify-center gap-2 transition-all">
            <Plus size={18}/> Kural Ekle
          </button>
        </form>

        {/* Tablo */}
        <div className="bg-slate-800 border border-slate-700 rounded-2xl overflow-hidden shadow-2xl">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-900/50 text-slate-400 font-mono text-xs uppercase">
              <tr>
                <th className="p-4">Kapsam (CIDR)</th>
                <th className="p-4">Gerekçe</th>
                <th className="p-4">Ekleyen</th>
                <th className="p-4">İşlem</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/50">
              {rules.map(rule => (
                <tr key={rule.id} className="hover:bg-slate-700/30 transition-colors group">
                  <td className="p-4 font-mono text-cyan-400 font-bold">{rule.cidr}</td>
                  <td className="p-4 text-slate-300">{rule.reason || '-'}</td>
                  <td className="p-4 text-slate-400 text-xs flex items-center gap-2"><Terminal size={12}/> {rule.created_by}</td>
                  <td className="p-4">
                    <button onClick={() => handleDelete(rule.id)} className="text-slate-500 hover:text-red-500 transition-colors"><Trash2 size={18}/></button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {rules.length === 0 && <div className="p-12 text-center text-slate-500 italic">Henüz tanımlanmış bir kural bulunmuyor.</div>}
        </div>
      </div>
    </div>
  );
}