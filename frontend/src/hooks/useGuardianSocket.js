import { useState, useEffect, useRef } from 'react';
import { WS_BASE } from '../config';

// Backend Docker üzerinde 8000 portundan yayın yapıyor
const SOCKET_URL = `${WS_BASE}/ws`;

export const useGuardianSocket = () => {
  const [data, setData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const socketRef = useRef(null);

  useEffect(() => {
    const connect = () => {
      socketRef.current = new WebSocket(SOCKET_URL);

      socketRef.current.onopen = () => {
        console.log("🟢 Guardian Engine'e Bağlanıldı!");
        setIsConnected(true);
      };

      socketRef.current.onmessage = (event) => {
        try {
          const parsedData = JSON.parse(event.data);
          // Gelen veriyi state'e atıyoruz
          setData(parsedData);
        } catch (error) {
          console.error("Veri okuma hatası:", error);
        }
      };

      socketRef.current.onclose = () => {
        console.log("🔴 Bağlantı Koptu. Tekrar deneniyor...");
        setIsConnected(false);
        // Bağlantı koparsa 3 saniye sonra tekrar dene (Reconnection Logic)
        setTimeout(connect, 3000);
      };

      socketRef.current.onerror = (error) => {
        console.error("Socket Hatası:", error);
        socketRef.current.close();
      };
    };

    connect();

    // Component kapanırken socket'i temizle
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, []);

  return { data, isConnected };
};