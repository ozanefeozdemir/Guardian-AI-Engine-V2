import React, { useState, useEffect, useRef } from 'react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts';
import { 
  Shield, Zap, Activity, Globe, AlertTriangle, Server, 
  Database, Search, Wifi, WifiOff
} from 'lucide-react';
import AuthLogs from './components/AuthLogs';
// --- Sabitler ---
// DÜZELTME: Backend api.py dosyasında adres "/ws/alerts" olarak tanımlı
const WS_URL = 'ws://localhost:8000/ws/alerts';

// Bu satırın da böyle olduğundan emin ol:
const API_URL = 'http://localhost:8000/status';const MAX_CHART_POINTS = 20; 
const COLORS = ['#ef4444', '#f59e0b', '#06b6d4', '#8b5cf6'];

// --- Alt Bileşenler ---

const AiScore = ({ confidence }) => {
  const percentage = Math.round(confidence * 100);
  
  return (
    <div className="relative flex flex-col items-center justify-center p-4 bg-slate-800 rounded-xl border border-slate-700 h-full">
      <h3 className="text-slate-400 text-xs mb-2 font-mono uppercase tracking-widest">AI Güven (Son Saldırı)</h3>
      <div className="relative w-24 h-12 overflow-hidden mt-1">
        <div className="absolute top-0 left-0 w-24 h-24 rounded-full border-8 border-slate-600 box-border"></div>
        <div 
          className={`absolute top-0 left-0 w-24 h-24 rounded-full border-8 box-border transition-all duration-500 ease-out ${percentage > 80 ? 'border-red-500' : 'border-cyan-500'}`}
          style={{ clipPath: 'polygon(0 0, 100% 0, 100% 50%, 0 50%)', transform: `rotate(${(percentage - 100) * 1.8}deg)` }}
        ></div>
      </div>
      <div className="mt-[-5px] text-xl font-bold text-white">%{percentage}</div>
      <span className="text-[10px] text-slate-400 mt-1">Tespit Kesinliği</span>
    </div>
  );
};

const GeoMap = ({ active }) => (
  <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 h-full min-h-[200px] flex flex-col relative overflow-hidden group">
    <h3 className="text-slate-400 text-xs font-mono uppercase z-10 flex items-center gap-2">
      <Globe size={14} className={active ? "text-red-500 animate-pulse" : "text-slate-500"}/> 
      {active ? "SALDIRI TESPİT EDİLDİ" : "Canlı Tehdit Haritası"}
    </h3>
    <svg viewBox="0 0 100 50" className="w-full h-full absolute top-0 left-0 opacity-20 text-slate-500 fill-current transition-opacity">
      <path d="M20,15 Q25,10 30,15 T40,20 T50,15 T60,20 T70,15 T80,20 T90,15 V35 H20 Z" />
    </svg>
    
    {active && (
      <>
        <div className="absolute top-1/3 left-1/4 w-3 h-3 bg-red-500 rounded-full animate-ping"></div>
        <div className="absolute top-1/2 left-2/3 w-2 h-2 bg-red-500 rounded-full animate-ping" style={{ animationDelay: '0.2s' }}></div>
      </>
    )}
  </div>
);

const RiskBadge = ({ confidence }) => {
  let level = 'Düşük';
  let color = 'bg-blue-500/20 text-blue-400 border-blue-500/30';
  
  if (confidence > 0.8) { level = 'KRİTİK'; color = 'bg-red-500/20 text-red-400 border-red-500/30'; }
  else if (confidence > 0.5) { level = 'YÜKSEK'; color = 'bg-orange-500/20 text-orange-400 border-orange-500/30'; }
  
  return (
    <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${color}`}>
      {level}
    </span>
  );
};

export default function ModernDashboard() {
  // --- State Yönetimi ---
  const [isConnected, setIsConnected] = useState(false);
  const [systemStatus, setSystemStatus] = useState('Checking...');
  const [trafficData, setTrafficData] = useState([]); 
  const [alerts, setAlerts] = useState([]); 
  const [stats, setStats] = useState({ totalTraffic: 0, totalAttacks: 0, lastConfidence: 0 });
  const [attackDistribution, setAttackDistribution] = useState([{ name: 'Analiz Ediliyor...', value: 1 }]);
  
  // --- Refs (Performans için Buffer) ---
  const bufferRef = useRef({
    packetCount: 0,
    attackCount: 0,
    attackTypes: {} // Saldırı tiplerini burada biriktireceğiz
  });

  // 1. Sistem Sağlığını Kontrol Et
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch(API_URL);
        if (res.ok) {
           setSystemStatus('ONLINE');
        } else {
           setSystemStatus('API ERROR');
        }
      } catch (err) {
        setSystemStatus('OFFLINE');
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 10000); 
    return () => clearInterval(interval);
  }, []);

  // 2. WebSocket Bağlantısı ve Veri İşleme
  useEffect(() => {
    let ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log('[Guardian] WS Connected');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // -- DÜZELTME 3: Backend veri yapısına uyum --
        // Backend'den gelen veri: { timestamp, src_ip, dst_port, label, confidence, is_attack }
        
        bufferRef.current.packetCount += 1;
        
        if (data.is_attack) {
          bufferRef.current.attackCount += 1;
          
          // Saldırı tipini say
          const type = data.label || 'Unknown Attack';
          bufferRef.current.attackTypes[type] = (bufferRef.current.attackTypes[type] || 0) + 1;

          setStats(prev => ({
            totalTraffic: prev.totalTraffic + 1,
            totalAttacks: prev.totalAttacks + 1,
            lastConfidence: data.confidence
          }));

          const newAlert = {
            id: Date.now() + Math.random(),
            time: new Date(data.timestamp * 1000).toLocaleTimeString('tr-TR'),
            type: data.label,          // Backend: label
            confidence: data.confidence,
            features: {}, // Featurelar backend'den ham gelmiyor artık, bu boş kalabilir
            src: data.src_ip || 'Gizli IP',   // Backend: src_ip
            dest: data.dst_port || '?',       // Backend: dst_port
          };

          setAlerts(prev => [newAlert, ...prev].slice(0, 50)); 
        } else {
          setStats(prev => ({ ...prev, totalTraffic: prev.totalTraffic + 1 }));
        }
      } catch (e) {
        console.error("Veri işleme hatası:", e);
      }
    };

    ws.onclose = () => {
      console.log('[Guardian] WS Disconnected');
      setIsConnected(false);
      // Basit reconnect mantığı
      setTimeout(() => {
         if(!isConnected) { console.log("Reconnecting..."); } // React useEffect dependency nedeniyle tam reconnect burada biraz karmaşık, sayfa yenileme gerekebilir.
      }, 3000);
    };

    return () => {
      ws.close();
    };
  }, []);

  // 3. Grafikleri Güncelleme Döngüsü
  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date().toLocaleTimeString('tr-TR');
      
      const newPoint = {
        name: now,
        traffic: bufferRef.current.packetCount,
        attack: bufferRef.current.attackCount
      };

      setTrafficData(prev => [...prev, newPoint].slice(-MAX_CHART_POINTS));

      // Pasta grafiği güncelle (Biriken tiplere göre)
      const newDistribution = Object.keys(bufferRef.current.attackTypes).map(key => ({
        name: key,
        value: bufferRef.current.attackTypes[key]
      }));
      
      if (newDistribution.length > 0) {
        setAttackDistribution(newDistribution);
      }

      // Saniyelik sayaçları sıfırla
      bufferRef.current.packetCount = 0;
      bufferRef.current.attackCount = 0;
      
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 font-sans selection:bg-cyan-500 selection:text-white pb-20">
      
      {/* Header */}
      <header className="bg-slate-950 border-b border-slate-800 px-6 py-4 flex justify-between items-center sticky top-0 z-50 shadow-2xl shadow-black/50">
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-br from-cyan-600 to-blue-700 p-2 rounded-lg shadow-lg shadow-cyan-500/20">
            <Shield className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-wider text-white leading-none">GUARDIAN</h1>
            <span className="text-[10px] text-cyan-500 font-mono tracking-[0.2em] uppercase">AI Engine V2</span>
          </div>
        </div>
        <div className="flex gap-6 text-sm font-mono">
          <div className="flex flex-col items-end">
             <span className="text-xs text-slate-500">ENGINE STATUS</span>
             <span className={`flex items-center gap-2 font-bold ${systemStatus === 'ONLINE' ? 'text-green-400' : 'text-red-500'}`}>
                <Activity size={14}/> {systemStatus}
             </span>
          </div>
          <div className="flex flex-col items-end border-l border-slate-700 pl-4">
             <span className="text-xs text-slate-500">LIVE FEED</span>
             <span className={`flex items-center gap-2 font-bold ${isConnected ? 'text-green-400' : 'text-red-500'}`}>
                {isConnected ? <Wifi size={14}/> : <WifiOff size={14}/>} {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
             </span>
          </div>
        </div>
      </header>

      <main className="p-6 max-w-[1600px] mx-auto space-y-6">
        
        {/* ROW 1: KPI KARTLARI */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Toplam İşlenen Paket */}
          <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 flex items-center justify-between hover:border-cyan-500/50 transition-colors">
            <div>
              <p className="text-slate-400 text-xs uppercase font-bold mb-1">İşlenen Paket</p>
              <h2 className="text-2xl font-bold text-white font-mono">{stats.totalTraffic.toLocaleString()}</h2>
              <p className="text-xs text-blue-400 mt-1 flex items-center gap-1">
                <Activity size={10}/> Real-time
              </p>
            </div>
            <div className="p-3 bg-slate-700 rounded-lg text-cyan-400"><Database size={24}/></div>
          </div>

          {/* Tespit Edilen Saldırı */}
          <div className={`bg-slate-800 p-4 rounded-xl border flex items-center justify-between transition-colors ${stats.totalAttacks > 0 ? 'border-red-500/50 shadow-[0_0_15px_rgba(239,68,68,0.2)]' : 'border-slate-700'}`}>
            <div>
              <p className="text-slate-400 text-xs uppercase font-bold mb-1">Tespit Edilen</p>
              <h2 className={`text-2xl font-bold font-mono ${stats.totalAttacks > 0 ? 'text-red-400' : 'text-white'}`}>{stats.totalAttacks}</h2>
              <p className="text-xs text-red-400 mt-1">Saldırı Girişimi</p>
            </div>
            <div className="p-3 bg-slate-700 rounded-lg text-red-400"><Shield size={24}/></div>
          </div>

          {/* İzlenen Kaynak */}
          <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 flex items-center justify-between hover:border-blue-500/50 transition-colors">
            <div>
              <p className="text-slate-400 text-xs uppercase font-bold mb-1">Mod</p>
              <h2 className="text-lg font-bold text-white truncate">AI / Simülasyon</h2>
              <p className="text-xs text-green-400 mt-1">Active Learning</p>
            </div>
            <div className="p-3 bg-slate-700 rounded-lg text-blue-400"><Server size={24}/></div>
          </div>

          {/* AI Skoru */}
          <div className="h-full">
             <AiScore confidence={stats.lastConfidence} />
          </div>
        </div>

        {/* ROW 2: GRAFİKLER */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[350px]">
          
          {/* Sol: Alan Grafiği (Traffic) */}
          <div className="lg:col-span-2 bg-slate-800 p-5 rounded-xl border border-slate-700 flex flex-col">
             <div className="flex justify-between mb-4">
                <h3 className="text-slate-200 font-bold flex items-center gap-2"><Zap size={18} className="text-yellow-400"/> Ağ Trafiği Akışı (Paket/Sn)</h3>
                <div className="flex gap-2">
                  <span className="flex items-center gap-1 text-xs text-slate-400"><div className="w-2 h-2 bg-cyan-500 rounded-full"></div> Trafik</span>
                  <span className="flex items-center gap-1 text-xs text-slate-400"><div className="w-2 h-2 bg-red-500 rounded-full"></div> Saldırı</span>
                </div>
             </div>
             <div className="flex-1 w-full min-h-0">
               <ResponsiveContainer width="100%" height="100%">
                 <AreaChart data={trafficData}>
                   <defs>
                     <linearGradient id="colorTraffic" x1="0" y1="0" x2="0" y2="1">
                       <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3}/>
                       <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                     </linearGradient>
                     <linearGradient id="colorAttack" x1="0" y1="0" x2="0" y2="1">
                       <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                       <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                     </linearGradient>
                   </defs>
                   <XAxis dataKey="name" stroke="#475569" fontSize={11} tickLine={false} axisLine={false}/>
                   <YAxis stroke="#475569" fontSize={11} tickLine={false} axisLine={false}/>
                   <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false}/>
                   <Tooltip 
                     contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f1f5f9' }} 
                     itemStyle={{ color: '#cbd5e1' }}
                   />
                   <Area type="monotone" dataKey="traffic" stroke="#06b6d4" strokeWidth={2} fillOpacity={1} fill="url(#colorTraffic)" isAnimationActive={false} />
                   <Area type="monotone" dataKey="attack" stroke="#ef4444" strokeWidth={2} fillOpacity={1} fill="url(#colorAttack)" isAnimationActive={false} />
                 </AreaChart>
               </ResponsiveContainer>
             </div>
          </div>

          {/* Sağ: Pie Chart */}
          <div className="bg-slate-800 p-5 rounded-xl border border-slate-700 flex flex-col">
            <h3 className="text-slate-200 font-bold mb-4 flex items-center gap-2"><Search size={18} className="text-orange-400"/> Saldırı Tipi Dağılımı</h3>
            <div className="flex-1 w-full min-h-0 relative">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={attackDistribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {attackDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderRadius: '8px', border: 'none' }} />
                  <Legend verticalAlign="bottom" height={36} iconType="circle" wrapperStyle={{ fontSize: '12px' }}/>
                </PieChart>
              </ResponsiveContainer>
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-[60%] text-center">
                 <span className="text-2xl font-bold text-white">{stats.totalAttacks}</span>
                 <p className="text-[10px] text-slate-400 uppercase">Tespit</p>
              </div>
            </div>
          </div>
        </div>

        {/* ROW 3: HARİTA + DETAYLI GEÇMİŞ */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
           
           {/* Sol: Harita (Animasyonlu) */}
           <div className="lg:col-span-1 h-80">
              <GeoMap active={alerts.length > 0 && (new Date() - new Date(alerts[0]?.id)) < 2000} /> 
           </div>

           {/* Sağ: Tablo (Canlı Veri) */}
           <div className="lg:col-span-3 bg-slate-800 rounded-xl border border-slate-700 overflow-hidden flex flex-col h-80">
              <div className="p-4 border-b border-slate-700 flex justify-between items-center bg-slate-900/50">
                <h3 className="font-bold text-white flex items-center gap-2">
                  <AlertTriangle className="text-red-500" size={18}/> Canlı Saldırı Akışı
                </h3>
                <div className="flex items-center gap-2">
                   <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
                   <div className="text-xs text-slate-400 font-mono">{isConnected ? 'LIVE FEED' : 'OFFLINE'}</div>
                </div>
              </div>
              
              <div className="overflow-y-auto flex-1">
                <table className="w-full text-sm text-left">
                  <thead className="text-xs text-slate-400 uppercase bg-slate-900 sticky top-0 z-10">
                    <tr>
                      <th className="px-4 py-3">Zaman</th>
                      <th className="px-4 py-3">Risk/Güven</th>
                      <th className="px-4 py-3">Saldırı Tipi</th>
                      <th className="px-4 py-3">Kaynak IP</th>
                      <th className="px-4 py-3">Hedef Port</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-700">
                    {alerts.map((alert) => (
                      <tr key={alert.id} className="hover:bg-slate-700/50 transition cursor-pointer group">
                        <td className="px-4 py-3 font-mono text-slate-400 text-xs">{alert.time}</td>
                        <td className="px-4 py-3"><RiskBadge confidence={alert.confidence} /></td>
                        <td className="px-4 py-3 text-red-300 font-bold">{alert.type}</td>
                        <td className="px-4 py-3 font-mono text-slate-300 text-xs">{alert.src}</td>
                        <td className="px-4 py-3 font-mono text-cyan-300 text-xs">{alert.dest}</td>
                      </tr>
                    ))}
                    {alerts.length === 0 && (
                      <tr>
                        <td colSpan="5" className="px-6 py-8 text-center text-slate-500 italic">
                          Henüz bir saldırı tespit edilmedi. Sistem izlemede...
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
           </div>
        </div>
	<AuthLogs />
      </main>
    </div>
  );
}