// frontend/src/ForecastView.tsx
import React, { useEffect, useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";
import { computeErrorMetrics } from "./forecast";

type Tech = "pv" | "wind";

interface ApiForecastPoint {
  time: string;   // ISO string from backend
  tech: Tech;
  actual: number | null;
  forecast: number;
}

interface ChartPoint {
  time: Date;
  label: string;
  pvActual: number | null;
  pvForecast: number | null;
  windActual: number | null;
  windForecast: number | null;
}

const API_BASE = "http://127.0.0.1:8000/Backend/forecast";

const ForecastView: React.FC = () => {
  const [horizon, setHorizon] = useState<number>(48); // hours
  const [allPoints, setAllPoints] = useState<ChartPoint[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // -------- Fetch forecast data from backend once on mount --------
  useEffect(() => {
  const fetchData = async () => {
    setLoading(true);
    setError(null);

    try {
      // FIXED WINDOW: Jan 1–4, 2025 (72 hours)
      const start = new Date("2025-01-01T00:00:00Z");
      const end = new Date("2025-01-04T00:00:00Z");

      const iso = (d: Date) => d.toISOString();

      const urlPv = `${API_BASE}?tech=pv&start=${encodeURIComponent(
        iso(start),
      )}&end=${encodeURIComponent(iso(end))}`;

      const urlWind = `${API_BASE}?tech=wind&start=${encodeURIComponent(
        iso(start),
      )}&end=${encodeURIComponent(iso(end))}`;

        const [pvResp, windResp] = await Promise.all([
          fetch(urlPv),
          fetch(urlWind),
        ]);

        if (!pvResp.ok || !windResp.ok) {
          throw new Error("Backend returned an error status");
        }

        const pvData = (await pvResp.json()) as ApiForecastPoint[];
        const windData = (await windResp.json()) as ApiForecastPoint[];

        // Merge PV + wind by timestamp
        const merged: Record<string, ChartPoint> = {};

        const upsert = (p: ApiForecastPoint) => {
          const key = p.time; // ISO string
          if (!merged[key]) {
            const t = new Date(p.time);
            merged[key] = {
              time: t,
              label: t.toISOString().slice(5, 16), // "MM-DDTHH:MM"
              pvActual: null,
              pvForecast: null,
              windActual: null,
              windForecast: null,
            };
          }
          const cp = merged[key];
          if (p.tech === "pv") {
            cp.pvActual = p.actual;
            cp.pvForecast = p.forecast;
          } else {
            cp.windActual = p.actual;
            cp.windForecast = p.forecast;
          }
        };

        pvData.forEach(upsert);
        windData.forEach(upsert);

        const points = Object.values(merged).sort(
          (a, b) => a.time.getTime() - b.time.getTime(),
        );

        setAllPoints(points);
      } catch (err) {
        console.error(err);
        setError("Could not load forecast data from backend.");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // -------- Slice according to horizon ----------------------------

  const sliced: ChartPoint[] = useMemo(
    () => allPoints.slice(0, horizon),
    [allPoints, horizon],
  );

  // -------- Metrics (filter out missing actuals) ------------------

  const pvMetrics = useMemo(() => {
    const yTrue: number[] = [];
    const yPred: number[] = [];
    sliced.forEach((p) => {
      if (p.pvActual != null && p.pvForecast != null) {
        yTrue.push(p.pvActual);
        yPred.push(p.pvForecast);
      }
    });
    return computeErrorMetrics(yTrue, yPred);
  }, [sliced]);

  const windMetrics = useMemo(() => {
    const yTrue: number[] = [];
    const yPred: number[] = [];
    sliced.forEach((p) => {
      if (p.windActual != null && p.windForecast != null) {
        yTrue.push(p.windActual);
        yPred.push(p.windForecast);
      }
    });
    return computeErrorMetrics(yTrue, yPred);
  }, [sliced]);

  // -------- Chart data for Recharts -------------------------------

  const chartData = sliced.map((p) => ({
    timeLabel: p.label,
    pvActual: p.pvActual != null ? Math.round(p.pvActual) : null,
    pvForecast: p.pvForecast != null ? Math.round(p.pvForecast) : null,
    windActual: p.windActual != null ? Math.round(p.windActual) : null,
    windForecast: p.windForecast != null ? Math.round(p.windForecast) : null,
  }));

  // ----------------------------------------------------------------

  return (
    <div>
      <h2 style={{ marginBottom: 8 }}>Spain PV &amp; Wind Forecast</h2>

      <div style={{ marginBottom: 16 }}>
        {loading && (
          <div style={{ fontSize: 13, opacity: 0.8 }}>
            Loading forecast data from backend…
          </div>
        )}
        {error && (
          <div style={{ fontSize: 13, color: "#e74c3c" }}>{error}</div>
        )}
        <label>
          Horizon (hours):{" "}
          <input
            type="range"
            min={24}
            max={72}
            step={6}
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
          />
          <span style={{ marginLeft: 8 }}>{horizon} h</span>
        </label>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "2.3fr 1fr",
          gap: "16px",
        }}
      >
        {/* Chart */}
        <div
          style={{
            background: "#101826",
            borderRadius: 12,
            padding: 12,
          }}
        >
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <CartesianGrid stroke="#1f2a3d" strokeDasharray="3 3" />
              <XAxis
                dataKey="timeLabel"
                tick={{ fontSize: 10 }}
                minTickGap={20}
              />
              <YAxis
                label={{
                  value: "Power (MW)",
                  angle: -90,
                  position: "insideLeft",
                }}
              />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="pvActual"
                name="PV actual"
                stroke="#f1c40f"
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 4 }}
                connectNulls={false}
              />
              <Line
                type="monotone"
                dataKey="pvForecast"
                name="PV forecast"
                stroke="#f39c12"
                strokeDasharray="4 4"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 3 }}
                connectNulls={false}
              />
              <Line
                type="monotone"
                dataKey="windActual"
                name="Wind actual"
                stroke="#3498db"
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 4 }}
                connectNulls={false}
              />
              <Line
                type="monotone"
                dataKey="windForecast"
                name="Wind forecast"
                stroke="#2980b9"
                strokeDasharray="4 4"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 3 }}
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Metrics */}
        <div
          style={{
            background: "#101826",
            borderRadius: 12,
            padding: 12,
            fontSize: 14,
          }}
        >
          <h3>Model performance</h3>

          <h4 style={{ marginTop: 12 }}>PV</h4>
          <MetricRow label="MAE (MW)" value={pvMetrics.mae} />
          <MetricRow label="RMSE (MW)" value={pvMetrics.rmse} />
          <MetricRow label="Bias (MW)" value={pvMetrics.bias} />
          <MetricRow label="MAPE (%)" value={pvMetrics.mape} />

          <h4 style={{ marginTop: 16 }}>Wind</h4>
          <MetricRow label="MAE (MW)" value={windMetrics.mae} />
          <MetricRow label="RMSE (MW)" value={windMetrics.rmse} />
          <MetricRow label="Bias (MW)" value={windMetrics.bias} />
          <MetricRow label="MAPE (%)" value={windMetrics.mape} />
        </div>
      </div>
    </div>
  );
};

const MetricRow: React.FC<{
  label: string;
  value: number | null;
}> = ({ label, value }) => (
  <div
    style={{
      display: "flex",
      justifyContent: "space-between",
      marginBottom: 4,
    }}
  >
    <span style={{ opacity: 0.85 }}>{label}</span>
    <span style={{ fontVariantNumeric: "tabular-nums" }}>
      {value === null || Number.isNaN(value)
        ? "N/A"
        : value.toFixed(1)}
    </span>
  </div>
);

export default ForecastView;

