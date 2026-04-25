/**
 * API ve WebSocket URL'lerini ortama göre belirler.
 * 
 * Docker (production build + nginx): relative URL kullanır → nginx proxy'e gider
 * Local dev (npm run dev):           localhost:8000'e direkt gider
 */

const isDev = import.meta.env.DEV;

// HTTP API base URL
export const API_BASE = isDev ? 'http://localhost:8000' : '';

// WebSocket base URL
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
export const WS_BASE = isDev
  ? 'ws://localhost:8000'
  : `${wsProtocol}//${window.location.host}`;
