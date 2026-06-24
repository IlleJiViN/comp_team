import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import './index.css';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const userIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
});

const resultIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
});

const PRESET_LOCATIONS = [
  { name: '신촌역 (나)', latitude: 37.5551, longitude: 126.9369 },
  { name: '강남역 (친구A)', latitude: 37.4979, longitude: 127.0276 },
  { name: '홍대입구역 (친구B)', latitude: 37.5568, longitude: 126.9242 },
];

function MapBoundsUpdater({ users, results }) {
  const map = useMap();
  useEffect(() => {
    if (users.length === 0 && results.length === 0) return;
    const bounds = L.latLngBounds();
    users.forEach(u => bounds.extend([u.latitude, u.longitude]));
    results.filter(r => r.latitude !== undefined && r.longitude !== undefined).forEach(r => bounds.extend([r.latitude, r.longitude]));
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [map, users, results]);
  return null;
}

function OpenMap({ users, results, center }) {
  const mapCenter = center ? [center.latitude, center.longitude] : [37.5551, 126.9369];

  return (
    <div className="map-container-wrapper">
      <MapContainer center={mapCenter} zoom={12} style={{ width: '100%', height: '350px', zIndex: 1 }}>
        <TileLayer
          attribution='&copy; OpenStreetMap'
          url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
        />
        <MapBoundsUpdater users={users} results={results} />
        
        {users.map(u => (
          <Marker key={u.name} position={[u.latitude, u.longitude]} icon={userIcon}>
            <Popup><strong>👤 {u.name}</strong></Popup>
          </Marker>
        ))}

        {results.map((r, idx) => {
           if(r.latitude && r.longitude) {
              return (
                <Marker key={r.id} position={[r.latitude, r.longitude]} icon={resultIcon}>
                  <Popup><strong>{idx + 1}. {r.name}</strong></Popup>
                </Marker>
              )
           }
           return null;
        })}
      </MapContainer>
    </div>
  );
}

const Avatar = ({ src, name, size = 24 }) => {
  const [error, setError] = useState(false);
  const safeSrc = src && src.startsWith('http://') ? src.replace('http://', 'https://') : src;

  useEffect(() => {
    setError(false);
  }, [src]);

  if (safeSrc && !error) {
    return (
      <img 
        src={safeSrc} 
        alt={name} 
        onError={() => setError(true)} 
        style={{ width: `${size}px`, height: `${size}px`, borderRadius: '50%', objectFit: 'cover', display: 'block' }} 
      />
    );
  }
  
  const charCodeSum = name ? name.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) : 0;
  const bgColor = `hsl(${charCodeSum % 360}, 65%, 65%)`;
  
  return (
    <div style={{ 
      width: `${size}px`, 
      height: `${size}px`, 
      borderRadius: '50%', 
      backgroundColor: bgColor, 
      color: 'white', 
      display: 'flex', 
      justifyContent: 'center', 
      alignItems: 'center', 
      fontSize: `${size * 0.45}px`, 
      fontWeight: 600,
      textTransform: 'uppercase'
    }}>
      {name ? name.charAt(0) : '?'}
    </div>
  );
};

function App() {
  const [theme, setTheme] = useState('light');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [latency, setLatency] = useState(0);
  const [hasSearched, setHasSearched] = useState(false);
  const [mapCenter, setMapCenter] = useState(null);
  const [users, setUsers] = useState([PRESET_LOCATIONS[0], PRESET_LOCATIONS[2]]);
  const [aiMessage, setAiMessage] = useState("");
  const [customLoc, setCustomLoc] = useState("");
  const [roomId, setRoomId] = useState(null);
  const [currentUser, setCurrentUser] = useState(() => {
    const saved = localStorage.getItem('currentUser');
    return saved ? JSON.parse(saved) : null;
  });

  // Save/Remove user session in localStorage
  useEffect(() => {
    if (currentUser) {
      localStorage.setItem('currentUser', JSON.stringify(currentUser));
    } else {
      localStorage.removeItem('currentUser');
    }
  }, [currentUser]);

  // Handle URL callback for Social Login
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const code = urlParams.get('code');
    const state = urlParams.get('state');

    if (code) {
      const isKakao = state === 'kakao';
      const endpoint = isKakao ? 'http://localhost:8001/auth/kakao/token' : 'http://localhost:8001/auth/naver/token';
      
      fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, state })
      })
      .then(res => res.json())
      .then(data => {
        if (data.user_id) {
          setCurrentUser(data);
          const pendingRoom = localStorage.getItem('join_room_pending');
          if (pendingRoom) {
            setRoomId(pendingRoom);
            localStorage.removeItem('join_room_pending');
            window.history.replaceState({}, document.title, `/?room=${pendingRoom}`);
          } else {
            window.history.replaceState({}, document.title, "/");
          }
        } else {
          console.error("Login failed:", data);
        }
      })
      .catch(err => console.error("Login error:", err));
    }
  }, []);

  // Read room from URL query on initial load
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const roomParam = urlParams.get('room');
    if (roomParam) {
      setRoomId(roomParam);
    }
  }, []);

  // Sync / Join Room when currentUser or roomId changes
  useEffect(() => {
    if (!roomId) return;
    
    // Sync Room Query string to URL
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('room') !== roomId) {
      urlParams.set('room', roomId);
      window.history.replaceState({}, document.title, "?" + urlParams.toString());
    }

    // Join room endpoint call
    const joinRoom = async () => {
      if (!currentUser) return;
      // Get host/user starting point location
      const userLoc = users.find(u => u.name === currentUser.name) || PRESET_LOCATIONS[0];
      
      try {
        await fetch('http://localhost:8001/room/join', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            room_id: roomId,
            user_id: currentUser.user_id,
            latitude: userLoc.latitude,
            longitude: userLoc.longitude
          })
        });
      } catch (err) {
        console.error("Failed to join room:", err);
      }
    };
    
    joinRoom();

    // Poll room members list
    const pollMembers = async () => {
      try {
        const res = await fetch(`http://localhost:8001/room/${roomId}/members`);
        const data = await res.json();
        if (data.members) {
          const memberLocs = data.members.map(m => ({
            name: m.name,
            latitude: m.latitude,
            longitude: m.longitude,
            profile_image: m.profile_image,
            user_id: m.user_id
          }));
          if (memberLocs.length > 0) {
            setUsers(memberLocs);
          }
        }
      } catch (err) {
        console.error("Error polling room members:", err);
      }
    };

    pollMembers();
    const interval = setInterval(pollMembers, 3000);
    return () => clearInterval(interval);
  }, [roomId, currentUser]);

  // Update backend location when our user's location changes in a room
  const updateRoomLocation = async (lat, lng) => {
    if (!roomId || !currentUser) return;
    try {
      await fetch('http://localhost:8001/room/update_location', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          room_id: roomId,
          user_id: currentUser.user_id,
          latitude: lat,
          longitude: lng
        })
      });
    } catch (err) {
      console.error("Failed to update room location:", err);
    }
  };

  const handleSetMyLocation = async (lat, lng) => {
    if (roomId && currentUser) {
      // Update local state first for instant response
      setUsers(prev => prev.map(u => u.user_id === currentUser.user_id ? { ...u, latitude: lat, longitude: lng } : u));
      await updateRoomLocation(lat, lng);
    }
  };

  const handleLogout = () => {
    setCurrentUser(null);
    localStorage.removeItem('currentUser');
    setRoomId(null);
    window.history.replaceState({}, document.title, "/");
    setUsers([PRESET_LOCATIONS[0], PRESET_LOCATIONS[2]]); // Reset to presets
  };

  const handleNaverLogin = async () => {
    const urlParams = new URLSearchParams(window.location.search);
    const roomParam = urlParams.get('room');
    if (roomParam) {
      localStorage.setItem('join_room_pending', roomParam);
    }
    
    try {
      const res = await fetch('http://localhost:8001/auth/naver/login');
      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }
      const data = await res.json();
      if (data.url) {
        window.location.href = data.url;
      } else {
        alert("네이버 로그인 URL을 가져오지 못했습니다. 백엔드 설정을 확인하세요.");
      }
    } catch (err) {
      console.error("Failed to fetch login URL", err);
      alert(`네이버 로그인 연결 실패: ${err.message}`);
    }
  };

  const handleKakaoLogin = async () => {
    const urlParams = new URLSearchParams(window.location.search);
    const roomParam = urlParams.get('room');
    if (roomParam) {
      localStorage.setItem('join_room_pending', roomParam);
    }

    try {
      const res = await fetch('http://localhost:8001/auth/kakao/login');
      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }
      const data = await res.json();
      if (data.url) {
        window.location.href = data.url;
      } else {
        alert("카카오 로그인 URL을 가져오지 못했습니다. 백엔드 설정을 확인하세요.");
      }
    } catch (err) {
      console.error("Failed to fetch Kakao login URL", err);
      alert(`카카오 로그인 연결 실패: ${err.message}`);
    }
  };

  const addCustomLocation = async (e) => {
    e.preventDefault();
    if(!customLoc.trim()) return;
    try {
      const res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(customLoc + ' 서울')}`);
      const data = await res.json();
      if(data && data.length > 0) {
        setUsers([...users, { name: customLoc, latitude: parseFloat(data[0].lat), longitude: parseFloat(data[0].lon) }]);
        setCustomLoc("");
      } else {
        alert('위치를 찾을 수 없습니다.');
      }
    } catch(err) {
      alert('위치 검색 중 오류가 발생했습니다.');
    }
  };

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  const handleAddUser = (e) => {
    const loc = PRESET_LOCATIONS.find(l => l.name === e.target.value);
    if (loc && !users.find(u => u.name === loc.name)) {
      setUsers([...users, loc]);
    }
    e.target.value = ""; 
  };

  const handleRemoveUser = (nameToRemove) => {
    setUsers(users.filter(u => u.name !== nameToRemove));
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim() || users.length === 0) {
      alert("검색어와 최소 1명 이상의 위치가 필요합니다.");
      return;
    }

    setLoading(true);
    setStreaming(true);
    setHasSearched(true);
    setAiMessage("");
    setResults([]);
    
    try {
      const res = await fetch('http://localhost:8001/search_rag', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query,
          top_k: 5,
          user_locations: users.map(u => ({ name: u.name, lat: u.latitude, lng: u.longitude }))
        })
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        // Process SSE lines
        const lines = buffer.split('\n\n');
        buffer = lines.pop(); // Keep the last incomplete chunk in buffer
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            if (!dataStr) continue;
            
            try {
              const data = JSON.parse(dataStr);
              if (data.type === 'results') {
                setResults(data.results || []);
                setLatency(data.elapsed_sec ? Math.round(data.elapsed_sec * 1000) : 0);
                setLoading(false); // Stop main loader, start showing streaming AI text
              } else if (data.type === 'chunk') {
                setAiMessage(prev => prev + data.text);
              } else if (data.type === 'done') {
                setStreaming(false);
              }
            } catch(e) {
              console.error("Failed to parse SSE JSON", e, dataStr);
            }
          }
        }
      }
    } catch (error) {
      console.error("Search failed:", error);
      alert("서버 연결에 실패했습니다.");
      setLoading(false);
      setStreaming(false);
    }
  };

  // Parse AI summaries from aiMessage
  const aiSummaries = {};
  if (aiMessage) {
    const parts = aiMessage.split(/\[ID:\s*(\d+)\]/);
    for (let i = 1; i < parts.length; i += 2) {
      const id = parseInt(parts[i], 10);
      const text = parts[i+1] ? parts[i+1].trim() : '';
      if (id && text) {
        aiSummaries[id] = text;
      }
    }
  }

  return (
    <div className="container">
      <header className="header">
        <div className="logo">
          <h1>SpotSync AI</h1>
        </div>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          {currentUser ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'var(--bg-primary)', padding: '0.4rem 1rem', borderRadius: '24px', border: '1px solid var(--border-color)', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
                <Avatar src={currentUser.profile_image} name={currentUser.name} size={24} />
                <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>{currentUser.name}</span>
              </div>
              <button onClick={handleLogout} style={{ background: 'transparent', border: '1px solid var(--border-color)', color: 'var(--text-color)', padding: '0.4rem 0.8rem', borderRadius: '24px', fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer' }}>
                로그아웃
              </button>
            </div>
          ) : (
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button onClick={handleKakaoLogin} style={{ background: '#FEE500', color: '#000000', border: 'none', padding: '0.5rem 1rem', borderRadius: '24px', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                카카오 로그인
              </button>
              <button onClick={handleNaverLogin} style={{ background: '#03C75A', color: 'white', border: 'none', padding: '0.5rem 1rem', borderRadius: '24px', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                네이버 로그인
              </button>
            </div>
          )}
          <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle Theme">
            {theme === 'light' ? '🌙' : '☀️'}
          </button>
        </div>
      </header>

      {roomId && (
        <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', padding: '1rem 1.5rem', borderRadius: '16px', marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', boxShadow: '0 2px 6px rgba(0,0,0,0.02)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '1.2rem' }}>🔗</span>
            <div>
              <div style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)' }}>실시간 모임방에 참가 중입니다</div>
              <div style={{ fontSize: '0.8rem', color: '#888' }}>방 코드: <strong>{roomId}</strong></div>
            </div>
          </div>
          <button 
            onClick={() => {
              const roomUrl = `${window.location.origin}/?room=${roomId}`;
              if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(roomUrl).then(() => {
                  alert("초대 링크가 클립보드에 복사되었습니다!");
                }).catch(err => {
                  alert("클립보드 복사 실패: " + err);
                });
              } else {
                const textArea = document.createElement("textarea");
                textArea.value = roomUrl;
                document.body.appendChild(textArea);
                textArea.select();
                try {
                  document.execCommand('copy');
                  alert("초대 링크가 클립보드에 복사되었습니다!");
                } catch (err) {
                  alert("클립보드 복사 실패");
                }
                document.body.removeChild(textArea);
              }
            }}
            style={{ background: 'var(--primary-color)', color: 'white', border: 'none', padding: '0.5rem 1rem', borderRadius: '20px', fontSize: '0.85rem', fontWeight: 600, cursor: 'pointer' }}
          >
            링크 공유하기
          </button>
        </div>
      )}

      <div className="toss-card users-container" style={{ padding: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 600 }}>🙌 모임 멤버 위치 ({users.length}명)</h3>
          <button 
            style={{ background: 'var(--primary-color)', color: 'white', border: 'none', padding: '0.4rem 0.8rem', borderRadius: '20px', fontSize: '0.8rem', cursor: 'pointer', fontWeight: 600 }} 
            onClick={async () => {
              if (!currentUser) {
                alert('소셜 로그인(카카오/네이버) 후에 모임방을 만들어 친구를 초대할 수 있습니다!');
                return;
              }
              if (roomId) {
                const roomUrl = `${window.location.origin}/?room=${roomId}`;
                if (navigator.clipboard && navigator.clipboard.writeText) {
                  navigator.clipboard.writeText(roomUrl).then(() => {
                    alert(`초대 링크가 클립보드에 복사되었습니다! 친구에게 공유해 보세요.\n${roomUrl}`);
                  });
                } else {
                  const textArea = document.createElement("textarea");
                  textArea.value = roomUrl;
                  document.body.appendChild(textArea);
                  textArea.select();
                  document.execCommand('copy');
                  document.body.removeChild(textArea);
                  alert(`초대 링크가 클립보드에 복사되었습니다! 친구에게 공유해 보세요.\n${roomUrl}`);
                }
                return;
              }
              // Create room
              const userLoc = users.find(u => u.name === currentUser.name) || PRESET_LOCATIONS[0];
              try {
                const res = await fetch('http://localhost:8001/room/create', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    host_id: currentUser.user_id,
                    latitude: userLoc.latitude,
                    longitude: userLoc.longitude
                  })
                });
                const data = await res.json();
                if (data.room_id) {
                  setRoomId(data.room_id);
                  const roomUrl = `${window.location.origin}/?room=${data.room_id}`;
                  if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(roomUrl).then(() => {
                      alert(`🎉 모임방이 생성되었습니다!\n초대 링크가 복사되었습니다. 친구들에게 전달하세요:\n${roomUrl}`);
                    });
                  } else {
                    const textArea = document.createElement("textarea");
                    textArea.value = roomUrl;
                    document.body.appendChild(textArea);
                    textArea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textArea);
                    alert(`🎉 모임방이 생성되었습니다!\n초대 링크가 복사되었습니다. 친구들에게 전달하세요:\n${roomUrl}`);
                  }
                } else {
                  alert("방 생성 실패: " + data.error);
                }
              } catch (err) {
                alert("방 생성 중 오류 발생: " + err.message);
              }
            }}
          >
            🔗 친구 초대
          </button>
        </div>

        {!currentUser && roomId && (
          <div style={{ background: 'rgba(254, 229, 0, 0.08)', padding: '1rem', borderRadius: '12px', marginBottom: '1.2rem', border: '1px solid rgba(254, 229, 0, 0.4)', display: 'flex', flexDirection: 'column', gap: '0.6rem', alignItems: 'center', textAlign: 'center' }}>
            <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-color)' }}>
              🔒 이 모임방에 참가하여 내 위치를 실시간으로 공유해 보세요!
            </span>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button onClick={handleKakaoLogin} style={{ background: '#FEE500', color: '#000', border: 'none', padding: '0.4rem 0.8rem', borderRadius: '16px', fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                카카오 로그인
              </button>
              <button onClick={handleNaverLogin} style={{ background: '#03C75A', color: '#FFF', border: 'none', padding: '0.4rem 0.8rem', borderRadius: '16px', fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                네이버 로그인
              </button>
            </div>
          </div>
        )}
        
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.8rem', marginBottom: '1.5rem' }}>
          {users.map((u, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', background: 'var(--bg-primary)', padding: '0.5rem 1rem', borderRadius: '24px', border: '1px solid var(--border-color)', boxShadow: '0 2px 4px rgba(0,0,0,0.02)' }}>
              <div style={{ marginRight: '8px' }}>
                <Avatar src={u.profile_image} name={u.name} size={24} />
              </div>
              <span style={{ fontSize: '0.9rem', fontWeight: 500, marginRight: '8px' }}>{u.name}</span>
              {(!roomId || (currentUser && u.user_id !== currentUser.user_id)) && (
                <button style={{ background: 'none', border: 'none', color: '#999', cursor: 'pointer', padding: 0 }} onClick={() => handleRemoveUser(u.name)}>✕</button>
              )}
            </div>
          ))}
        </div>
        
        <div style={{ background: 'var(--bg-primary)', padding: '1rem', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
          {roomId && currentUser ? (
            <>
              <h4 style={{ margin: '0 0 0.8rem 0', fontSize: '0.9rem', color: 'var(--text-color)' }}>📍 내 출발 위치 변경하기</h4>
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.8rem' }}>
                {PRESET_LOCATIONS.map(p => (
                  <button key={p.name} onClick={() => handleSetMyLocation(p.latitude, p.longitude)} style={{ flex: 1, padding: '0.6rem', border: '1px solid var(--border-color)', background: 'transparent', borderRadius: '8px', cursor: 'pointer', fontSize: '0.85rem', color: 'var(--text-color)' }}>
                    {p.name.replace(' (나)', '').replace(' (친구A)', '').replace(' (친구B)', '')}
                  </button>
                ))}
              </div>
              <form onSubmit={async (e) => {
                e.preventDefault();
                if(!customLoc.trim()) return;
                try {
                  const res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(customLoc + ' 서울')}`);
                  const data = await res.json();
                  if(data && data.length > 0) {
                    const lat = parseFloat(data[0].lat);
                    const lng = parseFloat(data[0].lon);
                    await handleSetMyLocation(lat, lng);
                    setCustomLoc("");
                  } else {
                    alert('위치를 찾을 수 없습니다.');
                  }
                } catch(err) {
                  alert('위치 검색 중 오류가 발생했습니다.');
                }
              }} style={{ display: 'flex', gap: '0.5rem' }}>
                <input 
                  type="text" 
                  placeholder="주변 역이나 도로명으로 검색 (예: 합정역)" 
                  value={customLoc}
                  onChange={e => setCustomLoc(e.target.value)}
                  style={{ flex: 1, padding: '0.8rem', border: '1px solid var(--border-color)', borderRadius: '8px', background: 'transparent', color: 'var(--text-primary)', fontSize: '0.9rem' }}
                />
                <button type="submit" style={{ padding: '0 1.2rem', background: 'var(--text-primary)', color: 'var(--bg-color)', border: 'none', borderRadius: '8px', fontWeight: 600, cursor: 'pointer' }}>변경</button>
              </form>
            </>
          ) : (
            <>
              <h4 style={{ margin: '0 0 0.8rem 0', fontSize: '0.9rem', color: 'var(--text-color)' }}>📍 위치 추가하기</h4>
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                {PRESET_LOCATIONS.filter(p => !users.find(u => u.name === p.name)).map(p => (
                  <button key={p.name} onClick={() => setUsers([...users, p])} style={{ flex: 1, padding: '0.6rem', border: '1px solid var(--border-color)', background: 'transparent', borderRadius: '8px', cursor: 'pointer', fontSize: '0.85rem', color: 'var(--text-color)' }}>
                    {p.name}
                  </button>
                ))}
              </div>
              <form onSubmit={addCustomLocation} style={{ display: 'flex', marginTop: '0.8rem', gap: '0.5rem' }}>
                <input 
                  type="text" 
                  placeholder="직접 위치 검색 (예: 합정역, 신논현역)" 
                  value={customLoc}
                  onChange={e => setCustomLoc(e.target.value)}
                  style={{ flex: 1, padding: '0.8rem', border: '1px solid var(--border-color)', borderRadius: '8px', background: 'transparent', color: 'var(--text-primary)', fontSize: '0.9rem' }}
                />
                <button type="submit" style={{ padding: '0 1.2rem', background: 'var(--text-primary)', color: 'var(--bg-color)', border: 'none', borderRadius: '8px', fontWeight: 600, cursor: 'pointer' }}>검색</button>
              </form>
            </>
          )}
        </div>
      </div>

      <form className="toss-card" onSubmit={handleSearch}>
        <div className="search-input-wrapper">
          <input
            type="text"
            className="search-input"
            placeholder="어떤 장소를 찾으시나요? (예: 분위기 좋은 카페)"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
          <button type="submit" className="search-btn" disabled={loading || !query.trim() || users.length === 0}>
            {loading ? <span className="loader"></span> : '모임 장소 찾기'}
          </button>
        </div>
      </form>

      <OpenMap users={users} results={results} center={mapCenter} />

      {hasSearched && !loading && results.length > 0 && (
        <div style={{marginTop: '2rem'}}>
          <div className="results-header">
            <span>추천 장소 Top {results.length}</span>
            {latency > 0 && <span className="latency">검색: {latency}ms</span>}
          </div>

          <div>
            {results.map((place, idx) => (
              <div className="toss-card place-card" key={place.id}>
                <div className="place-card-top">
                  <div>
                    <h3 className="place-title">{idx + 1}. {place.name}</h3>
                    <span className="place-category">{place.category}</span>
                  </div>
                </div>
                <div className="place-address" style={{ marginBottom: '0.8rem' }}>
                  📍 {place.address}
                </div>
                <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
                  <a href={`https://map.kakao.com/link/search/${encodeURIComponent(place.address + ' ' + place.name)}`} target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none', fontSize: '0.8rem', padding: '0.4rem 0.8rem', background: '#FEE500', color: '#000', borderRadius: '12px', fontWeight: 600 }}>
                    카카오맵
                  </a>
                  <a href={`https://map.naver.com/v5/search/${encodeURIComponent(place.address + ' ' + place.name)}`} target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none', fontSize: '0.8rem', padding: '0.4rem 0.8rem', background: '#03C75A', color: '#FFF', borderRadius: '12px', fontWeight: 600 }}>
                    네이버지도
                  </a>
                </div>
                <div className="place-score">
                  <span style={{ marginRight: '10px' }}>🤖 AI 매칭: {(place.score * 100).toFixed(0)}점</span>
                  {place.distance_to_midpoint_m != null && (
                    <span style={{ color: '#0066ff', fontWeight: '500' }}>
                      🎯 중간 지점에서 {place.distance_to_midpoint_m}m
                    </span>
                  )}
                </div>

                {place.travel_times && place.travel_times.length > 0 && (
                  <div style={{ marginTop: '0.8rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    {place.travel_times.map((t, tidx) => (
                      <div key={tidx} style={{ background: 'var(--bg-secondary)', padding: '0.4rem 0.8rem', borderRadius: '8px', fontSize: '0.85rem', color: 'var(--text-color)', border: '1px solid var(--border-color)' }}>
                        🚗 <strong>{t.name}</strong>에서 {t.minutes}분
                      </div>
                    ))}
                  </div>
                )}
                
                {aiSummaries[place.id] && (
                  <div className="place-ai-summary" style={{ marginTop: '0.5rem', padding: '1rem', backgroundColor: 'var(--bg-primary)', borderRadius: '12px', fontSize: '0.95rem', lineHeight: '1.5', color: 'var(--text-primary)', borderLeft: '4px solid var(--accent)' }}>
                    💡 {aiSummaries[place.id]}
                    {streaming && <span className="ai-typing-cursor"></span>}
                  </div>
                )}


              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
