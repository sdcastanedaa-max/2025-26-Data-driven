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
import { useForecastWindow } from "./ForecastWindowContext";

type SeriesPoint = {
  time: Date;
  pvActual: number | null;
  pvForecast: number;
  windActual: number | null;
  windForecast: number;
};

const BACKEND_BASE = "http://127.0.0.1:8000/Backend";

// --------- helpers ------------------------------------------------

const DEFAULT_START = new Date(Date.UTC(2025, 0, 1)); // 2025-01-01 00:00Z
const DEFAULT_HOURS = 72;

function toInputDateString(d: Date): string {
  // "YYYY-MM-DD" in UTC
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

// ------------------------------------------------------------------

const ForecastView: React.FC = () => {
  const { start, end, hours, setWindow } = useForecastWindow();

  // Make sure we always have a valid window, even if context is still default
  const effectiveStart = useMemo(
    () => start ?? DEFAULT_START,
    [start],
  );

  const effectiveHours = hours ?? DEFAULT_HOURS;

  const effectiveEnd = useMemo(() => {
    if (end) return end;
    return new Date(
      effectiveStart.getTime() + effectiveHours * 3600 * 1000,
    );
  }, [end, effectiveStart, effectiveHours]);

  // ---- UI state for selectors (kept in sync with effectiveStart) ----

  const [calendarValue, setCalendarValue] = useState<string>(
    toInputDateString(effectiveStart),
  );
  const [dayStr, setDayStr] = useState<string>(
    String(effectiveStart.getUTCDate()).padStart(2, "0"),
  );
  const [monthStr, setMonthStr] = useState<string>(
    String(effectiveStart.getUTCMonth() + 1).padStart(2, "0"),
  );
  const [yearStr, setYearStr] = useState<string>(
    String(effectiveStart.getUTCFullYear()),
  );

  // whenever effectiveStart changes (e.g. from Grid view), sync the controls
  useEffect(() => {
    const d = effectiveStart;
    setCalendarValue(toInputDateString(d));
    setDayStr(String(d.getUTCDate()).padStart(2, "0"));
    setMonthStr(String(d.getUTCMonth() + 1).padStart(2, "0"));
    setYearStr(String(d.getUTCFullYear()));
  }, [effectiveStart.getTime()]);

  // ---- data from backend ----

  const [data, setData] = useState<SeriesPoint[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function fetchData() {
      setLoading(true);
      setError(null);

      const startISO = effectiveStart.toISOString();
      const endISO = effectiveEnd.toISOString();

      try {
        const urlPv = `${BACKEND_BASE}/forecast?tech=pv&start=${encodeURIComponent(
          startISO,
        )}&end=${encodeURIComponent(endISO)}`;
        const urlWind = `${BACKEND_BASE}/forecast?tech=wind&start=${encodeURIComponent(
          startISO,
        )}&end=${encodeURIComponent(endISO)}`;

        const [respPv, respWind] = await Promise.all([
          fetch(urlPv),
          fetch(urlWind),
        ]);

        if (!respPv.ok || !respWind.ok) {
          throw new Error(
            `Backend error: PV ${respPv.status}, wind ${respWind.status}`,
          );
        }

        const pvJson = await respPv.json();
        const windJson = await respWind.json();

        if (cancelled) return;

        // Index wind points by ISO time string
        const windByTime = new Map<string, any>();
        for (const w of windJson) {
          windByTime.set(w.time, w);
        }

        const merged: SeriesPoint[] = pvJson.map((p: any) => {
          const t = new Date(p.time);
          const w = windByTime.get(p.time);
          return {
            time: t,
            pvActual:
              p.actual === null || p.actual === undefined
                ? null
                : Number(p.actual),
            pvForecast: Number(p.forecast),
            windActual:
              w && w.actual !== null && w.actual !== undefined
                ? Number(w.actual)
                : null,
            windForecast: w ? Number(w.forecast) : 0,
          };
        });

        setData(merged);
      } catch (e: any) {
        console.error("Error fetching forecast data", e);
        if (!cancelled) {
          setError("Could not load forecast data from backend.");
          setData([]);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchData();
    return () => {
      cancelled = true;
    };
  }, [effectiveStart.getTime(), effectiveEnd.getTime()]);

  // ---- metrics & chart data ----

  const chartData = useMemo(
    () =>
      data.map((p) => ({
        timeLabel: p.time.toISOString().slice(5, 16), // "MM-DDTHH:MM"
        pvActual: p.pvActual === null ? null : Math.round(p.pvActual),
        pvForecast: Math.round(p.pvForecast),
        windActual:
          p.windActual === null ? null : Math.round(p.windActual),
        windForecast: Math.round(p.windForecast),
      })),
    [data],
  );

  const pvMetrics = useMemo(() => {
    const actual = data
      .map((p) => p.pvActual)
      .filter((v): v is number => v !== null);
    if (actual.length === 0) return { mae: 0, rmse: 0, bias: 0, mape: null };
    const forecast = data.map((p) => p.pvForecast);
    return computeErrorMetrics(actual, forecast);
  }, [data]);

  const windMetrics = useMemo(() => {
    const actual = data
      .map((p) => p.windActual)
      .filter((v): v is number => v !== null);
    if (actual.length === 0) return { mae: 0, rmse: 0, bias: 0, mape: null };
    const forecast = data.map((p) => p.windForecast);
    return computeErrorMetrics(actual, forecast);
  }, [data]);

  // ---- handlers for selectors ------------------------------------

  const applyManualDate = () => {
    const y = Number(yearStr);
    const m = Number(monthStr);
    const d = Number(dayStr);
    if (
      !Number.isFinite(y) ||
      !Number.isFinite(m) ||
      !Number.isFinite(d)
    ) {
      return;
    }
    // UTC midnight
    const newStart = new Date(Date.UTC(y, m - 1, d, 0, 0, 0));
    if (isNaN(newStart.getTime())) return;

    setWindow(newStart, effectiveHours);
  };

  const handleCalendarChange: React.ChangeEventHandler<
    HTMLInputElement
  > = (e) => {
    const value = e.target.value;
    setCalendarValue(value);
    if (!value) return;
    const newStart = new Date(`${value}T00:00:00Z`);
    if (!isNaN(newStart.getTime())) {
      setWindow(newStart, effectiveHours);
    }
  };

  // ---- render ----------------------------------------------------

  return (
    <div
      style={{
        width: "100%",
        maxWidth: "100%",         // match your GridView / App main
        margin: "0 auto",
        boxSizing: "border-box",
      }}
    >
      <h2 style={{ marginBottom: 8 }}>Spain PV &amp; Wind Forecast</h2>

      <p style={{ fontSize: 12, opacity: 0.8, marginBottom: 4 }}>
        Window: {toInputDateString(effectiveStart)} →{" "}
        {toInputDateString(effectiveEnd)} ({effectiveHours} hours)
      </p>

      {error && (
        <p style={{ color: "#e74c3c", marginBottom: 8 }}>
          {error}
        </p>
      )}

      {/* Date selectors */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 16,
          marginBottom: 16,
          alignItems: "center",
        }}
      >
        <div>
          <div style={{ fontSize: 12, opacity: 0.8 }}>
            Start date (calendar)
          </div>
          <input
            type="date"
            value={calendarValue}
            onChange={handleCalendarChange}
            style={{
              marginTop: 4,
              padding: "4px 6px",
              borderRadius: 4,
              border: "1px solid #2c3e50",
              background: "#050814",
              color: "#f5f5f5",
            }}
          />
        </div>

        <div>
          <div style={{ fontSize: 12, opacity: 0.8 }}>
            Start date (manual)
          </div>
          <div style={{ display: "flex", gap: 6, marginTop: 4 }}>
            <input
              type="text"
              value={dayStr}
              onChange={(e) => setDayStr(e.target.value)}
              style={{ width: 32 }}
              maxLength={2}
              placeholder="DD"
            />
            <input
              type="text"
              value={monthStr}
              onChange={(e) => setMonthStr(e.target.value)}
              style={{ width: 32 }}
              maxLength={2}
              placeholder="MM"
            />
            <input
              type="text"
              value={yearStr}
              onChange={(e) => setYearStr(e.target.value)}
              style={{ width: 52 }}
              maxLength={4}
              placeholder="YYYY"
            />
            <button
              onClick={applyManualDate}
              style={{
                padding: "2px 10px",
                borderRadius: 4,
                border: "1px solid #75896b",
                background: "#75896b",
                color: "#000",
                cursor: "pointer",
                fontSize: 12,
              }}
            >
              Apply
            </button>
          </div>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "2.3fr 1fr",
          gap: 16,
        }}
      >
        {/* Chart */}
        <div
          style={{
            background: "#232920",
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
                connectNulls
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
              />
              <Line
                type="monotone"
                dataKey="windActual"
                name="Wind actual"
                stroke="#3498db"
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 4 }}
                connectNulls
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
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Metrics */}
        <div
          style={{
            background: "#232920",
            borderRadius: 12,
            padding: 12,
            fontSize: 14,
            color: "#f5f5f5",
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

      {loading && (
        <p style={{ marginTop: 8, fontSize: 12, opacity: 0.8 }}>
          Loading data…
        </p>
      )}
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
      {value === null ? "N/A" : value.toFixed(1)}
    </span>
  </div>
);

export default ForecastView;