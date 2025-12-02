// frontend/src/ForecastView.tsx
import React, { useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from "recharts";

// ----------------- Types & helpers --------------------

interface BasePoint {
  tLabel: string; // x-axis label
  pv: number;    // PV actual
  wind: number;  // Wind actual
}

interface ErrorMetrics {
  mae: number;
  rmse: number;
  bias: number;
  mape: number | null;
}

// same normal noise helper as before
function randn(mean = 0, std = 1): number {
  const u = 1 - Math.random();
  const v = 1 - Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return mean + z * std;
}

// 72h of hourly PV/Wind data
function generateBaseData(nHours: number = 72): BasePoint[] {
  const now = new Date();
  now.setMinutes(0, 0, 0);

  const points: BasePoint[] = [];

  for (let i = 0; i < nHours; i++) {
    const t = new Date(now.getTime() + i * 3600 * 1000);
    const hourOfDay = (t.getHours() + t.getMinutes() / 60) % 24;

    // PV: daytime curve, peak ~15
    const pvShape = Math.max(
      0,
      Math.sin(((hourOfDay - 6) / 12) * Math.PI),
    );
    const pv = 15 * pvShape + randn(0, 0.5);

    // Wind: slower oscillation around ~20
    const windShape =
      0.7 +
      0.15 * Math.sin((2 * Math.PI * i) / 48) +
      0.1 * Math.sin((2 * Math.PI * i) / 8);
    const wind = 20 * Math.max(0.1, windShape) + randn(0, 0.8);

    points.push({
      tLabel: t.toISOString().slice(5, 16), // "MM-DDTHH:MM"
      pv,
      wind,
    });
  }

  return points;
}

// simple metric helper
function computeErrorMetrics(
  yTrue: number[],
  yPred: number[],
): ErrorMetrics {
  const n = yTrue.length;
  if (n === 0 || n !== yPred.length) {
    return { mae: 0, rmse: 0, bias: 0, mape: null };
  }

  let mae = 0;
  let mse = 0;
  let bias = 0;
  let mapeSum = 0;
  let mapeCount = 0;

  for (let i = 0; i < n; i++) {
    const t = yTrue[i];
    const p = yPred[i];
    const diff = p - t;

    mae += Math.abs(diff);
    mse += diff * diff;
    bias += diff;

    if (t !== 0) {
      mapeSum += Math.abs(diff / t);
      mapeCount += 1;
    }
  }

  mae /= n;
  const rmse = Math.sqrt(mse / n);
  bias /= n;
  const mape = mapeCount > 0 ? (mapeSum / mapeCount) * 100 : null;

  return { mae, rmse, bias, mape };
}

// ----------------- Component -------------------------

const ForecastView: React.FC = () => {
  const [horizon, setHorizon] = useState<number>(48);

  // base PV/Wind data (72h) – same shape as the simple working example
  const baseData = useMemo(() => generateBaseData(72), []);

  // slice for current horizon
  const data = useMemo(
    () => baseData.slice(0, horizon),
    [baseData, horizon],
  );

  // define "forecast" as a simple biased version of actual
  const pvActualArr = data.map((p) => p.pv);
  const pvForecastArr = data.map((p) => p.pv * 0.95 + 1.0); // MW-ish bias
  const windActualArr = data.map((p) => p.wind);
  const windForecastArr = data.map((p) => p.wind * 1.05 - 1.0);

  const pvMetrics = useMemo(
    () => computeErrorMetrics(pvActualArr, pvForecastArr),
    [pvActualArr, pvForecastArr],
  );

  const windMetrics = useMemo(
    () => computeErrorMetrics(windActualArr, windForecastArr),
    [windActualArr, windForecastArr],
  );

  console.log("Forecast points:", data.length, data[0]);

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
          <LineChart
            width={800}
            height={400}
            data={data}
          >
            <CartesianGrid stroke="#1f2a3d" strokeDasharray="3 3" />
            <XAxis
              dataKey="tLabel"
              tick={{ fontSize: 10 }}
              minTickGap={20}
            />
            <YAxis
              label={{
                value: "Power (arbitrary units)",
                angle: -90,
                position: "insideLeft",
              }}
            />
            <Tooltip />
            <Legend />
            {/* Actual series use the plain keys 'pv' and 'wind' */}
            <Line
              type="monotone"
              dataKey="pv"
              name="PV actual"
              stroke="#f1c40f"
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 4 }}
            />
            <Line
              type="monotone"
              dataKey={(p: BasePoint) => p.pv * 0.95 + 1.0}
              name="PV forecast"
              stroke="#f39c12"
              strokeDasharray="4 4"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 3 }}
            />
            <Line
              type="monotone"
              dataKey="wind"
              name="Wind actual"
              stroke="#3498db"
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 4 }}
            />
            <Line
              type="monotone"
              dataKey={(p: BasePoint) => p.wind * 1.05 - 1.0}
              name="Wind forecast"
              stroke="#2980b9"
              strokeDasharray="4 4"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 3 }}
            />
          </LineChart>
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
          <MetricRow label="MAE" value={pvMetrics.mae} />
          <MetricRow label="RMSE" value={pvMetrics.rmse} />
          <MetricRow label="Bias" value={pvMetrics.bias} />
          <MetricRow
            label="MAPE (%)"
            value={pvMetrics.mape}
          />

          <h4 style={{ marginTop: 16 }}>Wind</h4>
          <MetricRow label="MAE" value={windMetrics.mae} />
          <MetricRow label="RMSE" value={windMetrics.rmse} />
          <MetricRow label="Bias" value={windMetrics.bias} />
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
      {value === null ? "N/A" : value.toFixed(2)}
    </span>
  </div>
);

export default ForecastView;


