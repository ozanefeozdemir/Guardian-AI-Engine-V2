import React, { useState, useEffect } from 'react';
import { Shield, Activity, Server, Database, Lock, Cpu, Network, Users, ChevronRight, Github } from 'lucide-react';

const GuardianLanding = () => {
  const [isScrolled, setIsScrolled] = useState(false);

  // Scroll takibi için
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="min-h-screen font-sans text-slate-800 bg-slate-50">
      
      {/* --- NAVBAR --- */}
      <nav className={`fixed w-full z-50 transition-all duration-300 ${isScrolled ? 'bg-white shadow-md py-4' : 'bg-transparent py-6'}`}>
        <div className="container mx-auto px-6 flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <Shield className={`w-8 h-8 ${isScrolled ? 'text-blue-700' : 'text-white'}`} />
            <span className={`text-2xl font-bold tracking-tighter ${isScrolled ? 'text-slate-900' : 'text-white'}`}>
              GUARDIAN
            </span>
          </div>
          <div className="hidden md:flex space-x-8">
            {['Hakkında', 'Mimari', 'Teknoloji', 'Ekip'].map((item) => (
              <a 
                key={item} 
                href={`#${item.toLowerCase()}`} 
                className={`text-sm font-medium hover:text-cyan-400 transition-colors ${isScrolled ? 'text-slate-600' : 'text-slate-200'}`}
              >
                {item}
              </a>
            ))}
          </div>
          <button className={`px-5 py-2 rounded-full font-semibold text-sm transition-all ${isScrolled ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-white text-blue-900 hover:bg-cyan-50'}`}>
            Dashboard'a Git
          </button>
        </div>
      </nav>

      {/* --- HERO SECTION --- */}
      <header className="relative bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 h-screen flex items-center overflow-hidden">
        {/* Arkaplan Efekti */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-0 left-0 w-96 h-96 bg-cyan-500 rounded-full blur-3xl mix-blend-screen animate-pulse"></div>
          <div className="absolute bottom-0 right-0 w-96 h-96 bg-blue-600 rounded-full blur-3xl mix-blend-screen"></div>
        </div>

        <div className="container mx-auto px-6 relative z-10 text-center md:text-left">
          <div className="md:w-2/3">
            <div className="inline-flex items-center px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-400/30 text-cyan-300 text-xs font-semibold mb-6">
              <span className="w-2 h-2 rounded-full bg-cyan-400 mr-2 animate-pulse"></span>
              BİL493 Bitirme Projesi
            </div>
            <h1 className="text-5xl md:text-7xl font-bold text-white leading-tight mb-6">
              Yapay Zekânın Gözüyle <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-400">
                Ağınızı Koruyun
              </span>
            </h1>
            <p className="text-lg md:text-xl text-slate-300 mb-8 max-w-2xl leading-relaxed">
              Guardian, imza tabanlı sistemlerin yetersiz kaldığı "Sıfırıncı Gün" saldırılarına karşı derin öğrenme tabanlı dinamik bir kalkan oluşturur.
            </p>
            <div className="flex flex-col md:flex-row space-y-4 md:space-y-0 md:space-x-4">
              <button className="px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-lg text-white font-bold shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40 transition-all flex items-center justify-center">
                Proje Detayları <ChevronRight className="ml-2 w-5 h-5" />
              </button>
              <button className="px-8 py-4 bg-slate-800 border border-slate-700 rounded-lg text-slate-300 font-bold hover:bg-slate-700 transition-all flex items-center justify-center">
                <Github className="mr-2 w-5 h-5" /> GitHub
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* --- ABOUT / PROBLEM & SOLUTION --- */}
      <section id="hakkında" className="py-24 bg-white">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-slate-900 mb-4">Neden Guardian?</h2>
            <p className="text-slate-600 max-w-2xl mx-auto">
              Geleneksel güvenlik duvarları reaktiftir. Guardian ise proaktif bir yaklaşım sunarak ağ trafiğindeki anomalileri gerçekleştiği anda tespit eder.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-10">
            {[
              {
                icon: <Activity className="w-10 h-10 text-cyan-500" />,
                title: "Sıfırıncı Gün Tehditleri",
                desc: "İmza tabanlı sistemlerin yakalayamadığı, daha önce görülmemiş saldırı türlerini Autoencoder mimarisi ile tespit eder."
              },
              {
                icon: <Lock className="w-10 h-10 text-blue-500" />,
                title: "Proaktif Koruma",
                desc: "Saldırı gerçekleştikten sonra değil, anomali başladığı anda uyarı üreterek sisteme dinamik bir kalkan sağlar."
              },
              {
                icon: <Network className="w-10 h-10 text-indigo-500" />,
                title: "Gerçek Zamanlı Analiz",
                desc: "Redis ve WebSocket altyapısı sayesinde 2 saniyenin altında gecikme ile ağ trafiğini analiz eder ve raporlar."
              }
            ].map((feature, idx) => (
              <div key={idx} className="p-8 bg-slate-50 rounded-2xl border border-slate-100 hover:shadow-xl transition-shadow duration-300">
                <div className="bg-white w-16 h-16 rounded-xl flex items-center justify-center mb-6 shadow-sm">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-bold text-slate-900 mb-3">{feature.title}</h3>
                <p className="text-slate-600 leading-relaxed">{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* --- ARCHITECTURE / PROCESS --- */}
      <section id="mimari" className="py-24 bg-slate-900 text-white relative overflow-hidden">
        <div className="container mx-auto px-6 relative z-10">
          <div className="mb-16 md:flex justify-between items-end">
            <div>
              <h2 className="text-3xl font-bold mb-2">Sistem Mimarisi</h2>
              <p className="text-slate-400">Veri akışının her aşamasında modern teknolojiler.</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 relative">
            {/* Bağlantı Çizgisi (Desktop) */}
            <div className="hidden md:block absolute top-1/2 left-0 w-full h-1 bg-gradient-to-r from-cyan-500/20 via-blue-500/20 to-indigo-500/20 -translate-y-1/2 z-0"></div>

            {[
              { 
                step: "01", 
                title: "Veri Toplama", 
                tech: "Scapy & Python", 
                desc: "Canlı ağ trafiği dinlenir ve özellik vektörleri çıkarılır." 
              },
              { 
                step: "02", 
                title: "Analiz Motoru", 
                tech: "PyTorch & Autoencoder", 
                desc: "Derin öğrenme modeli trafiği analiz eder, anomalileri yakalar." 
              },
              { 
                step: "03", 
                title: "İletişim & API", 
                tech: "Redis & FastAPI", 
                desc: "Uyarılar mesaj kuyruğuna iletilir ve veritabanına işlenir." 
              },
              { 
                step: "04", 
                title: "Görselleştirme", 
                tech: "React Dashboard", 
                desc: "WebSocket üzerinden anlık uyarılar ekrana yansıtılır." 
              }
            ].map((item, idx) => (
              <div key={idx} className="relative z-10 bg-slate-800/50 backdrop-blur-sm p-6 rounded-xl border border-slate-700 hover:border-cyan-500/50 transition-colors">
                <div className="text-4xl font-black text-slate-700 mb-4 opacity-50">{item.step}</div>
                <h3 className="text-lg font-bold text-white mb-1">{item.title}</h3>
                <div className="text-xs font-semibold text-cyan-400 mb-3">{item.tech}</div>
                <p className="text-sm text-slate-400">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* --- SDG GOALS --- */}
      <section className="py-16 bg-blue-50">
        <div className="container mx-auto px-6 text-center">
          <p className="text-sm font-semibold text-blue-600 uppercase tracking-widest mb-4">Birleşmiş Milletler Sürdürülebilir Kalkınma Amaçları</p>
          <div className="flex flex-wrap justify-center gap-4">
            {["Hedef 9: Sanayi & Yenilikçilik", "Hedef 11: Sürdürülebilir Şehirler", "Hedef 16: Güçlü Kurumlar"].map((goal, i) => (
              <span key={i} className="px-4 py-2 bg-white rounded-lg shadow-sm text-slate-700 font-medium border border-blue-100">
                {goal}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* --- TEAM --- */}
      <section id="ekip" className="py-24 bg-white">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-center text-slate-900 mb-16">Proje Ekibi</h2>
          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            
            {/* Üye 1 */}
            <div className="group text-center">
              <div className="w-32 h-32 mx-auto bg-slate-200 rounded-full mb-6 overflow-hidden relative group-hover:ring-4 ring-cyan-400 transition-all">
                 <div className="absolute inset-0 flex items-center justify-center text-slate-400">
                    <Users size={48} />
                 </div>
              </div>
              <h3 className="text-xl font-bold text-slate-900">Alperen Melih Göncü</h3>
              <p className="text-blue-600 font-medium mb-2">Yapay Zeka & Model</p>
              <p className="text-sm text-slate-500 px-4">
                Autoencoder mimarisi, veri ön işleme ve PyTorch model eğitimi.
              </p>
            </div>

            {/* Üye 2 */}
            <div className="group text-center">
              <div className="w-32 h-32 mx-auto bg-slate-200 rounded-full mb-6 overflow-hidden relative group-hover:ring-4 ring-blue-500 transition-all">
                <div className="absolute inset-0 flex items-center justify-center text-slate-400">
                    <Server size={48} />
                 </div>
              </div>
              <h3 className="text-xl font-bold text-slate-900">Ozan Efe Özdemir</h3>
              <p className="text-blue-600 font-medium mb-2">Backend & DevOps</p>
              <p className="text-sm text-slate-500 px-4">
                FastAPI mimarisi, PostgreSQL veritabanı, Redis ve Docker dağıtımı.
              </p>
            </div>

            {/* Üye 3 */}
            <div className="group text-center">
              <div className="w-32 h-32 mx-auto bg-slate-200 rounded-full mb-6 overflow-hidden relative group-hover:ring-4 ring-indigo-500 transition-all">
                <div className="absolute inset-0 flex items-center justify-center text-slate-400">
                    <Cpu size={48} />
                 </div>
              </div>
              <h3 className="text-xl font-bold text-slate-900">Cem Mete Sarıtaş</h3>
              <p className="text-blue-600 font-medium mb-2">Frontend & UI/UX</p>
              <p className="text-sm text-slate-500 px-4">
                React Dashboard geliştirme, WebSocket entegrasyonu ve veri görselleştirme.
              </p>
            </div>

          </div>
        </div>
      </section>

      {/* --- FOOTER --- */}
      <footer className="bg-slate-900 text-slate-400 py-12 border-t border-slate-800">
        <div className="container mx-auto px-6 flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <div className="flex items-center space-x-2 mb-2">
              <Shield className="w-6 h-6 text-blue-500" />
              <span className="text-xl font-bold text-white">GUARDIAN</span>
            </div>
            <p className="text-sm">Akıllı Ağ Güvenliği Çözümü</p>
          </div>
          <div className="text-sm">
            &copy; 2025 Guardian Project. BİL493 Kapsamında Geliştirilmiştir.
          </div>
        </div>
      </footer>
    </div>
  );
};

export default GuardianLanding;