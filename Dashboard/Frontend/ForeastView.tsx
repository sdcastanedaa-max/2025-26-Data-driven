import React, { useMemo, useState } from "react";
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
import {
  generateDummyForecast,
  computeErrorMetrics,
} from "./forecast";

const ForecastView: React.FC = () => {
  const [horizon, setHorizon] = useState<number>(48);

  const data = useMemo(() => generateDummyForecast(72), []);

  const sliced = useMemo(
    () => data.slice(0, horizon),
    [data, horizon],
  );

  const pvMetrics = useMemo(
    () =>
      computeErrorMetrics(
        sliced.map((p) => p.pvActual),
        sliced.map((p) => p.pvForecast),
      ),
    [sliced],
  );

  const windMetrics = useMemo(
    () =>
      computeErrorMetrics(
        sliced.map((p) => p.windActual),
        sliced.map((p) => p.windForecast),
      ),
    [sliced],
  );

  const chartData = sliced.map((p) => ({
    timeLabel: p.time.toISOString().slice(5, 16), // "MM-DDTHH:MM"
    pvActual: Math.round(p.pvActual),
    pvForecast: Math.round(p.pvForecast),
    windActual: Math.round(p.windActual),
    windForecast: Math.round(p.windForecast),
  }));

  return (
    <div>
      <h2 style={{ marginBottom: 8 }}>Spain PV &amp; Wind Forecast</h2>

      <div style={{ marginBottom: 16 }}>
        <label>
          Horizon (hours):{" "}
          <input
            type="range"
            min={24}
            max={72}
            step={6}
            value={horizon}
            onChange={(e) =>
              setHorizon(Number(e.target.value))
            }
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
          <MetricRow
            label="MAPE (%)"
            value={pvMetrics.mape}
          />

          <h4 style={{ marginTop: 16 }}>Wind</h4>
          <MetricRow label="MAE (MW)" value={windMetrics.mae} />
          <MetricRow label="RMSE (MW)" value={windMetrics.rmse} />
          <MetricRow label="Bias (MW)" value={windMetrics.bias} />
          <MetricRow
            label="MAPE (%)"
            value={windMetrics.mape}
          />
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
      {value === null ? "N/A" : value.toFixed(1)}
    </span>
  </div>
);

export default ForecastView;
