import React, { useState, useEffect } from 'react';
import { 
  LayoutDashboard, 
  History, 
  Settings, 
  ShieldCheck, 
  Bell, 
  AlertTriangle, 
  BarChartHorizontal,
  Circle,
  Database
} from 'lucide-react';

// --- Sahte Veri ve Yardımcı Fonksiyonlar ---

// Geçmiş veriler için sahte data
const mockHistoricalAlerts = [
  { id: 101, timestamp: '2025-11-17 14:30:15', srcIp: '192.168.1.105', destIp: '10.0.0.5', risk: 'Kritik', desc: 'Tespit: Potansiyel DDoS Saldırısı (UDP Flood)' },
  { id: 102, timestamp: '2025-11-17 14:25:02', srcIp: '203.0.113.12', destIp: '10.0.0.5', risk: 'Yüksek', desc: 'Tespit: SSH Brute Force Denemesi' },
  { id: 103, timestamp: '2025-11-17 14:10:45', srcIp: '192.168.1.201', destIp: '10.0.0.22', risk: 'Orta', desc: 'Tespit: Bilinmeyen Port Taraması' },
  { id: 104, timestamp: '2025-11-17 13:55:19', srcIp: '172.16.0.5', destIp: '10.0.0.8', risk: 'Düşük', desc: 'Tespit: Anormal Ağ Aktivitesi' },
  { id: 105, timestamp: '2025-11-17 13:40:11', srcIp: '192.168.1.105', destIp: '10.0.0.5', risk: 'Kritik', desc: 'Tespit: Potansiyel DDoS Saldırısı (ICMP Flood)' },
  { id: 106, timestamp: '2025-11-17 13:20:05', srcIp: '203.0.113.12', destIp: '10.0.0.5', risk: 'Yüksek', desc: 'Tespit: SSH Brute Force Denemesi' },
  { id: 107, timestamp: '2025-11-17 13:15:22', srcIp: '10.0.0.5', destIp: '198.51.100.2', risk: 'Orta', desc: 'Tespit: Veri Sızıntısı Şüphesi (Anormal Upload)' },
];

// Canlı veri akışı için rastgele uyarı üreteci
const riskLevels = ['Kritik', 'Yüksek', 'Orta', 'Düşük'];
const attackTypes = [
  'Potansiyel DDoS Saldırısı (UDP Flood)',
  'SSH Brute Force Denemesi',
  'Bilinmeyen Port Taraması',
  'SQL Injection Denemesi',
  'Anormal Ağ Aktivitesi (ICMP)',
  'Veri Sızıntısı Şüphesi'
];

const createRandomAlert = () => {
  const risk = riskLevels[Math.floor(Math.random() * riskLevels.length)];
  const desc = attackTypes[Math.floor(Math.random() * attackTypes.length)];
  return {
    id: new Date().getTime(),
    timestamp: new Date().toLocaleTimeString('tr-TR'),
    srcIp: `192.168.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
    destIp: `10.0.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
    risk: risk,
    desc: `Tespit: ${desc}`,
  };
};

// --- Bileşenler ---

// 1. Sol Sidebar
const Sidebar = ({ currentPage, setCurrentPage, isConnected }) => {
  const navItems = [
    { name: 'Ana Panel', icon: LayoutDashboard, page: 'dashboard' },
    { name: 'Uyarı Geçmişi', icon: History, page: 'history' },
    { name: 'Veri Kaynakları', icon: Database, page: 'data' },
    { name: 'Ayarlar', icon: Settings, page: 'settings' },
  ];

  return (
    <div className="flex flex-col w-64 h-screen px-4 py-8 bg-slate-900 text-slate-200 sticky top-0">
      {/* Logo */}
      <div className="flex items-center justify-center mb-10">
        <ShieldCheck className="w-10 h-10 text-cyan-400" />
        <span className="ml-3 text-2xl font-bold text-white">Guardian</span>
      </div>

      {/* Navigasyon */}
      <nav className="flex-1 space-y-2">
        {navItems.map((item) => (
          <button
            key={item.name}
            onClick={() => setCurrentPage(item.page)}
            className={`flex items-center w-full px-4 py-3 rounded-lg transition-colors duration-200 ${
              currentPage === item.page
                ? 'bg-cyan-600 text-white'
                : 'hover:bg-slate-800 hover:text-white'
            }`}
          >
            <item.icon className="w-5 h-5" />
            <span className="ml-4 font-medium">{item.name}</span>
          </button>
        ))}
      </nav>

      {/* Bağlantı Durumu */}
      <div className="mt-auto">
        <div className={`flex items-center px-4 py-2 rounded-lg ${
          isConnected ? 'bg-green-800' : 'bg-red-800'
        }`}>
          <Circle className={`w-3 h-3 ${
            isConnected ? 'text-green-300' : 'text-red-300'
          }`} fill="currentColor" />
          <span className="ml-3 text-sm font-medium text-slate-100">
            {isConnected ? 'Bağlantı Aktif' : 'Bağlantı Kesildi'}
          </span>
        </div>
      </div>
    </div>
  );
};

// Risk Seviyesine göre etiket (badge)
const RiskBadge = ({ risk }) => {
  const colors = {
    'Kritik': 'bg-red-600 text-white',
    'Yüksek': 'bg-orange-500 text-white',
    'Orta': 'bg-yellow-400 text-slate-900',
    'Düşük': 'bg-blue-500 text-white',
  };
  return (
    <span className={`px-3 py-1 text-xs font-semibold rounded-full ${colors[risk] || 'bg-gray-400'}`}>
      {risk}
    </span>
  );
};

// 2. Ana Panel Sayfası
const DashboardPage = ({ liveAlerts }) => {
  
  // Sahte istatistik verisi
  const chartData = [
    { type: 'DDoS', count: 42, color: 'bg-red-500' },
    { type: 'Brute Force', count: 28, color: 'bg-orange-500' },
    { type: 'Port Tarama', count: 71, color: 'bg-yellow-400' },
    { type: 'Diğer', count: 15, color: 'bg-blue-400' },
  ];
  const maxCount = Math.max(...chartData.map(d => d.count));

  return (
    <div className="p-8 space-y-6">
      <h1 className="text-3xl font-bold text-slate-900">Ana Panel</h1>
      
      {/* KPI Kartları */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <KpiCard 
          title="Toplam Uyarı (24s)" 
          value={liveAlerts.length + 104} 
          icon={AlertTriangle} 
          iconColor="text-red-500" 
        />
        <KpiCard 
          title="Kritik Uyarılar (24s)" 
          value={liveAlerts.filter(a => a.risk === 'Kritik').length + 12} 
          icon={ShieldCheck} 
          iconColor="text-orange-500" 
        />
        <KpiCard 
          title="İzlenen Cihazlar" 
          value="254" 
          icon={Database} 
          iconColor="text-blue-500" 
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Canlı Uyarı Akışı */}
        <div className="lg:col-span-2 bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold text-slate-800 mb-4 flex items-center">
            <Bell className="w-5 h-5 mr-2 text-cyan-600" />
            Anlık Uyarı Akışı
          </h2>
          <div className="overflow-y-auto h-96 space-y-3 pr-2">
            {liveAlerts.length === 0 && (
              <p className="text-slate-500 text-center mt-10">Yeni uyarı bekleniyor...</p>
            )}
            {liveAlerts.map((alert) => (
              <div key={alert.id} className="flex items-center p-3 bg-slate-50 rounded-lg animate-pulse-once">
                <div className="mr-3">
                  <RiskBadge risk={alert.risk} />
                </div>
                <div className="flex-1 text-sm">
                  <p className="text-slate-700 font-medium">{alert.desc}</p>
                  <p className="text-slate-500">
                    {alert.srcIp} &rarr; {alert.destIp}
                  </p>
                </div>
                <div className="text-xs text-slate-400 ml-4">
                  {alert.timestamp}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Uyarı Dağılımı */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold text-slate-800 mb-4 flex items-center">
            <BarChartHorizontal className="w-5 h-5 mr-2 text-cyan-600" />
            Uyarı Tipi Dağılımı (24s)
          </h2>
          <div className="space-y-4 pt-4">
            {chartData.map((data) => (
              <div key={data.type}>
                <div className="flex justify-between items-center mb-1 text-sm">
                  <span className="font-medium text-slate-700">{data.type}</span>
                  <span className="font-bold text-slate-600">{data.count}</span>
                </div>
                <div className="w-full h-4 bg-slate-200 rounded-full overflow-hidden">
                  <div 
                    className={`${data.color} h-4 rounded-full`}
                    style={{ width: `${(data.count / maxCount) * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

// KPI Kartı Bileşeni
const KpiCard = ({ title, value, icon: Icon, iconColor }) => (
  <div className="bg-white p-6 rounded-lg shadow-md flex items-center justify-between">
    <div>
      <p className="text-sm font-medium text-slate-500 uppercase">{title}</p>
      <p className="text-3xl font-bold text-slate-900">{value}</p>
    </div>
    <div className={`p-3 rounded-full bg-opacity-20 ${iconColor.replace('text-', 'bg-')}`}>
      <Icon className={`w-6 h-6 ${iconColor}`} />
    </div>
  </div>
);

// 3. Uyarı Geçmişi Sayfası
const HistoryPage = ({ allAlerts }) => {
  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold text-slate-900 mb-6">Uyarı Geçmişi</h1>
      <div className="bg-white shadow-md rounded-lg overflow-hidden">
        <table className="min-w-full divide-y divide-slate-200">
          <thead className="bg-slate-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Zaman Damgası</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Risk</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Kaynak IP</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Hedef IP</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Açıklama</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-slate-200">
            {allAlerts.map((alert) => (
              <tr key={alert.id} className="hover:bg-slate-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600">{alert.timestamp}</td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <RiskBadge risk={alert.risk} />
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-slate-800">{alert.srcIp}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-slate-800">{alert.destIp}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-700 max-w-sm truncate">{alert.desc}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// 4. Diğer Sayfalar (Placeholder)
const PlaceholderPage = ({ title }) => (
  <div className="p-8">
    <h1 className="text-3xl font-bold text-slate-900">{title}</h1>
    <p className="mt-4 text-slate-600">Bu sayfa şu anda yapım aşamasındadır.</p>
  </div>
);


// --- Ana App Bileşeni ---

export default function ClassicDashboard() {
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [isConnected, setIsConnected] = useState(true);
  const [liveAlerts, setLiveAlerts] = useState([]);
  const [allAlerts] = useState(mockHistoricalAlerts);

  // Sahte WebSocket bağlantı durumu simülasyonu
  useEffect(() => {
    const connectionInterval = setInterval(() => {
      // %10 ihtimalle bağlantı kesilsin/gelsin
      if (Math.random() < 0.1) {
        setIsConnected(prev => !prev);
      }
    }, 5000);
    return () => clearInterval(connectionInterval);
  }, []);

  // Sahte canlı uyarı akışı simülasyonu
  useEffect(() => {
    if (!isConnected) return; // Bağlantı yoksa yeni uyarı gelmesin

    const alertInterval = setInterval(() => {
      setLiveAlerts(prevAlerts => [
        createRandomAlert(),
        ...prevAlerts
      ].slice(0, 20)); // Son 20 uyarıyı tut
    }, Math.random() * 3000 + 1000); // 1-4 saniyede bir yeni uyarı

    return () => clearInterval(alertInterval);
  }, [isConnected]);

  // CSS Animasyonu için (Canlı akışta yeni gelen item için)
  useEffect(() => {
    const style = document.createElement('style');
    style.innerHTML = `
      @keyframes pulse-once {
        0% { background-color: #ecfdf5; } /* Hafif yeşil */
        50% { background-color: #d1fae5; }
        100% { background-color: #f8fafc; } /* Orijinal arka plan */
      }
      .animate-pulse-once {
        animation: pulse-once 1s ease-out;
      }
    `;
    document.head.appendChild(style);
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  // Sayfa render etme
  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <DashboardPage liveAlerts={liveAlerts} />;
      case 'history':
        return <HistoryPage allAlerts={allAlerts} />;
      case 'data':
        return <PlaceholderPage title="Veri Kaynakları" />;
      case 'settings':
        return <PlaceholderPage title="Ayarlar" />;
      default:
        return <DashboardPage liveAlerts={liveAlerts} />;
    }
  };

  return (
    <div className="flex min-h-screen bg-slate-100 font-sans">
      <Sidebar 
        currentPage={currentPage} 
        setCurrentPage={setCurrentPage}
        isConnected={isConnected} 
      />
      <main className="flex-1 overflow-y-auto">
        {renderPage()}
      </main>
    </div>
  );
}