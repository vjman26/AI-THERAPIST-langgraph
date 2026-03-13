import ReactMarkdown from "react-markdown";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend
} from "recharts";
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const WS_URL = API_BASE.replace(/^http/, "ws") + "/ws/telemetry";
const MAX_SERIES = 240;

const formatNumber = (value, digits = 2) => {
  if (value === null || value === undefined) {
    return "--";
  }
  if (typeof value === "number") {
    return value.toFixed(digits);
  }
  return value;
};

const formatTime = (ts) => {
  const date = new Date(ts);
  return `${date.getMinutes()}:${String(date.getSeconds()).padStart(2, "0")}`;
};

const pushSeries = (prev, key, point) => {
  const next = prev[key].concat(point);
  if (next.length > MAX_SERIES) {
    return next.slice(next.length - MAX_SERIES);
  }
  return next;
};

const baseLineProps = {
  dot: false,
  strokeWidth: 2,
  isAnimationActive: false
};

export default function App() {
  const [telemetry, setTelemetry] = useState(null);
  const [chatOpen, setChatOpen] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Engineer view online. Ask for race strategy or telemetry insights." }
  ]);
  const [series, setSeries] = useState({
    speed: [],
    fuel: [],
    gapAhead: [],
    gapBehind: [],
    ers: [],
    tyreWear: [],
    damage: []
  });
  const wsRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.status !== "ok") {
          return;
        }
        setTelemetry(payload);
        const now = Date.now();
        setSeries((prev) => ({
          speed: pushSeries(prev, "speed", {
            t: now,
            value: payload.speed_kph ?? null
          }),
          fuel: pushSeries(prev, "fuel", {
            t: now,
            value: payload.fuel_kg ?? null,
            laps: payload.fuel_laps ?? null
          }),
          gapAhead: pushSeries(prev, "gapAhead", {
            t: now,
            value: payload.gap_ahead_s ?? null
          }),
          gapBehind: pushSeries(prev, "gapBehind", {
            t: now,
            value: payload.gap_behind_s ?? null
          }),
          ers: pushSeries(prev, "ers", {
            t: now,
            value: payload.ers_energy ?? null
          }),
          tyreWear: pushSeries(prev, "tyreWear", {
            t: now,
            fl: payload.tyre_wear?.[0] ?? null,
            fr: payload.tyre_wear?.[1] ?? null,
            rl: payload.tyre_wear?.[2] ?? null,
            rr: payload.tyre_wear?.[3] ?? null
          }),
          damage: pushSeries(prev, "damage", {
            t: now,
            fl: payload.front_left_wing_damage ?? null,
            fr: payload.front_right_wing_damage ?? null,
            rear: payload.rear_wing_damage ?? null,
            engine: payload.engine_damage ?? null
          })
        }));
      } catch (err) {
        console.error(err);
      }
    };
    ws.onclose = () => {
      wsRef.current = null;
    };
    return () => {
      ws.close();
    };
  }, []);

  const statusCards = useMemo(() => {
    if (!telemetry) {
      return [];
    }
    return [
      { label: "Session", value: telemetry.session_type ?? "--" },
      { label: "Lap", value: telemetry.lap ?? "--" },
      { label: "Position", value: telemetry.position ?? "--" },
      { label: "Tyres", value: telemetry.tyre_compound ?? "--" },
      { label: "Weather", value: telemetry.weather ?? "--" },
      { label: "Safety Car", value: telemetry.safety_car ?? "--" },
      { label: "Fuel (kg)", value: formatNumber(telemetry.fuel_kg, 2) },
      { label: "Fuel Laps", value: formatNumber(telemetry.fuel_laps, 2) },
      { label: "ERS Energy", value: formatNumber(telemetry.ers_energy, 2) },
      { label: "ERS Mode", value: telemetry.ers_mode ?? "--" }
    ];
  }, [telemetry]);

  const submitChat = async () => {
    const trimmed = chatInput.trim();
    if (!trimmed) {
      return;
    }
    const nextMessages = messages.concat({ role: "user", content: trimmed });
    setMessages(nextMessages);
    setChatInput("");
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: trimmed,
        history: nextMessages.filter((item) => item.role !== "assistant_seed")
      })
    });
    const data = await response.json();
    setMessages((prev) => prev.concat({ role: "assistant", content: data.response }));
  };

  return (
    <div className={`app ${chatOpen ? "chat-open" : ""}`}>
      <header className="header">
        <div>
          <p className="eyebrow">Engineer View</p>
          <h1>F1 Telemetry Command Center</h1>
          <p className="subtle">
            Live UDP telemetry feed with strategy context, tuned for practice, qualifying, and race.
          </p>
        </div>
        <button className="chat-toggle" onClick={() => setChatOpen((open) => !open)}>
          {chatOpen ? "Close Chat" : "Open Chat"}
        </button>
      </header>

      <section className="status-grid">
        {statusCards.length === 0 ? (
          <div className="status-empty">Waiting for telemetry on UDP 127.0.0.1:20777</div>
        ) : (
          statusCards.map((card) => (
            <div key={card.label} className="status-card">
              <p>{card.label}</p>
              <h3>{card.value}</h3>
            </div>
          ))
        )}
      </section>

      <section className="chart-grid">
        <div className="chart-card">
          <h3>Speed (kph)</h3>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={series.speed}>
              <XAxis dataKey="t" tickFormatter={formatTime} />
              <YAxis />
              <Tooltip labelFormatter={formatTime} />
              <Legend />
              <Line {...baseLineProps} dataKey="value" name="Speed" stroke="#ff6b4a" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-card">
          <h3>Fuel (kg)</h3>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={series.fuel}>
              <XAxis dataKey="t" tickFormatter={formatTime} />
              <YAxis />
              <Tooltip labelFormatter={formatTime} />
              <Legend />
              <Line {...baseLineProps} dataKey="value" name="Fuel kg" stroke="#ffd166" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-card">
          <h3>Gaps (s)</h3>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={series.gapAhead}>
              <XAxis dataKey="t" tickFormatter={formatTime} />
              <YAxis />
              <Tooltip labelFormatter={formatTime} />
              <Legend />
              <Line {...baseLineProps} dataKey="value" name="Gap Ahead" stroke="#4ecdc4" />
            </LineChart>
          </ResponsiveContainer>
          <ResponsiveContainer width="100%" height={120}>
            <LineChart data={series.gapBehind}>
              <XAxis dataKey="t" tickFormatter={formatTime} hide />
              <YAxis hide />
              <Tooltip labelFormatter={formatTime} />
              <Legend />
              <Line {...baseLineProps} dataKey="value" name="Gap Behind" stroke="#1a659e" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-card">
          <h3>ERS Energy</h3>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={series.ers}>
              <XAxis dataKey="t" tickFormatter={formatTime} />
              <YAxis />
              <Tooltip labelFormatter={formatTime} />
              <Legend />
              <Line {...baseLineProps} dataKey="value" name="ERS MJ" stroke="#7bd389" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-card wide">
          <h3>Tyre Wear (%)</h3>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={series.tyreWear}>
              <XAxis dataKey="t" tickFormatter={formatTime} />
              <YAxis />
              <Tooltip labelFormatter={formatTime} />
              <Legend />
              <Line {...baseLineProps} dataKey="fl" name="FL" stroke="#ef476f" />
              <Line {...baseLineProps} dataKey="fr" name="FR" stroke="#ffd166" />
              <Line {...baseLineProps} dataKey="rl" name="RL" stroke="#06d6a0" />
              <Line {...baseLineProps} dataKey="rr" name="RR" stroke="#118ab2" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-card">
          <h3>Damage (%)</h3>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={series.damage}>
              <XAxis dataKey="t" tickFormatter={formatTime} />
              <YAxis />
              <Tooltip labelFormatter={formatTime} />
              <Legend />
              <Line {...baseLineProps} dataKey="fl" name="FL Wing" stroke="#ff9f1c" />
              <Line {...baseLineProps} dataKey="fr" name="FR Wing" stroke="#ffbf69" />
              <Line {...baseLineProps} dataKey="rear" name="Rear Wing" stroke="#2ec4b6" />
              <Line {...baseLineProps} dataKey="engine" name="Engine" stroke="#6d597a" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <aside className={`chat-drawer ${chatOpen ? "open" : ""}`}>
        <div className="chat-header">
          <h2>Race Engineer Chat</h2>
          <button className="chat-toggle ghost" onClick={() => setChatOpen(false)}>
            Close
          </button>
        </div>
        <div className="chat-body">
          {messages.map((msg, idx) => (
            <div key={idx} className={`chat-message ${msg.role}`}>
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
          ))}
        </div>
        <div className="chat-input">
          <input
            value={chatInput}
            onChange={(event) => setChatInput(event.target.value)}
            placeholder="Ask for strategy, pit windows, fuel, tyres..."
          />
          <button onClick={submitChat}>Send</button>
        </div>
      </aside>
    </div>
  );
}



