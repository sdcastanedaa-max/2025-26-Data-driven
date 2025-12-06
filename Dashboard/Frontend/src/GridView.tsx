// frontend/src/GridView.tsx
import { useSimTime } from "./simTime";
import React, { useEffect, useMemo, useState } from "react";
import GridCanvas from "./GridCanvas";
import { simulateLineFlows } from "./simulation";
import { LINES, NODES } from "./gridData";
import type { GridNode, NodeType } from "./gridData";
import { useForecastWindow } from "./ForecastWindowContext";

interface TooltipInfo {
  node: GridNode;
  x: number;
  y: number;
}

type NodeStatus = {
  netPowerMW: number;
  isDisconnected: boolean;
  isOverloaded: boolean;
};

type FlowMode = "sim" | "forecast";

const LEGEND_ITEMS: { key: NodeType; label: string }[] = [
  { key: "pv", label: "PV farm" },
  { key: "wind", label: "Wind farm" },
  { key: "load_ind", label: "Industrial load" },
  { key: "load_res", label: "Commercial / residential load" },
  { key: "bess", label: "BESS (battery)" },
  { key: "fossil", label: "Fossil plant" },
  { key: "nuclear", label: "Nuclear plant" },
  { key: "substation", label: "Substation" },
];

// same backend root as ForecastView
const BACKEND_BASE = "http://127.0.0.1:8000/Backend";

type ForecastPoint = {
  tMs: number;        // timestamp in ms
  pvForecast: number; // MW
  windForecast: number; // MW
};

// ---- simple real-time clock hook (HH:MM:SS, local) -----------------------

function useNowClock(): string {
  const [timeStr, setTimeStr] = useState<string>(() => {
    const d = new Date();
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
  });

  useEffect(() => {
    const id = window.setInterval(() => {
      const d = new Date();
      const hh = String(d.getHours()).padStart(2, "0");
      const mm = String(d.getMinutes()).padStart(2, "0");
      const ss = String(d.getSeconds()).padStart(2, "0");
      setTimeStr(`${hh}:${mm}:${ss}`);
    }, 1000);

    return () => window.clearInterval(id);
  }, []);

  return timeStr;
}

const GridView: React.FC = () => {
  const tHours = useSimTime();
  const { start, hours } = useForecastWindow();
  const nowClock = useNowClock();

  const [hoverInfo, setHoverInfo] = useState<TooltipInfo | null>(null);
  const [pinnedInfo, setPinnedInfo] = useState<TooltipInfo | null>(null);

  const [flowMode, setFlowMode] = useState<FlowMode>("sim");

  const [typeVisibility, setTypeVisibility] = useState<
    Record<NodeType, boolean>
  >(() => {
    const init: Record<NodeType, boolean> = {} as any;
    LEGEND_ITEMS.forEach((item) => {
      init[item.key] = true;
    });
    return init;
  });

  const toggleTypeVisibility = (key: NodeType) => {
    setTypeVisibility((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  // ------------------------------------------------------------
  // 1) Load PV/Wind forecast for the current forecast window
  // ------------------------------------------------------------
  const [forecastSeries, setForecastSeries] = useState<ForecastPoint[]>([]);
  const [forecastError, setForecastError] = useState<string | null>(null);

  useEffect(() => {
    if (!start) return;

    const effectiveHours = hours ?? 72;
    const startISO = start.toISOString();
    const endISO = new Date(
      start.getTime() + effectiveHours * 3600 * 1000,
    ).toISOString();

    let cancelled = false;

    async function loadForecast() {
      setForecastError(null);

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

        // index wind by ISO time
        const windByTime = new Map<string, any>();
        for (const w of windJson) {
          windByTime.set(w.time, w);
        }

        const merged: ForecastPoint[] = pvJson.map((p: any) => {
          const t = new Date(p.time).getTime();
          const w = windByTime.get(p.time);
          return {
            tMs: t,
            pvForecast: Number(p.forecast),
            windForecast: w ? Number(w.forecast) : 0,
          };
        });

        // sort by time just in case
        merged.sort((a, b) => a.tMs - b.tMs);
        setForecastSeries(merged);
      } catch (e: any) {
        console.error("Error fetching grid forecast data", e);
        if (!cancelled) {
          setForecastError("Could not load forecast data for grid view.");
          setForecastSeries([]);
        }
      }
    }

    loadForecast();
    return () => {
      cancelled = true;
    };
  }, [start?.getTime(), hours]);

  // ------------------------------------------------------------
  // 2) For the current simulated time, get interpolated totals
  // ------------------------------------------------------------

  const forecastTotals = useMemo(() => {
    if (
      flowMode !== "forecast" ||
      !start ||
      forecastSeries.length === 0
    ) {
      return undefined;
    }

    const tAbs = start.getTime() + tHours * 3600 * 1000;

    // clamp outside range
    if (tAbs <= forecastSeries[0].tMs) {
      const p = forecastSeries[0];
      return { pvMW: p.pvForecast, windMW: p.windForecast };
    }
    const last = forecastSeries[forecastSeries.length - 1];
    if (tAbs >= last.tMs) {
      return { pvMW: last.pvForecast, windMW: last.windForecast };
    }

    // find bounding points
    for (let i = 0; i < forecastSeries.length - 1; i++) {
      const p0 = forecastSeries[i];
      const p1 = forecastSeries[i + 1];
      if (tAbs >= p0.tMs && tAbs <= p1.tMs) {
        const span = p1.tMs - p0.tMs || 1;
        const alpha = (tAbs - p0.tMs) / span;

        const pvMW =
          p0.pvForecast + alpha * (p1.pvForecast - p0.pvForecast);
        const windMW =
          p0.windForecast + alpha * (p1.windForecast - p0.windForecast);

        return { pvMW, windMW };
      }
    }

    // fallback (shouldn't normally hit)
    return { pvMW: last.pvForecast, windMW: last.windForecast };
  }, [flowMode, start?.getTime(), tHours, forecastSeries]);

  // ------------------------------------------------------------
  // 3) Compute flows for canvas + table
  // ------------------------------------------------------------

  const flows = useMemo(
    () =>
      simulateLineFlows(tHours, {
        useForecast: flowMode === "forecast" && !!forecastTotals,
        forecastTotals,
      }),
    [tHours, flowMode, forecastTotals],
  );

  // ------------------------------------------------------------
  // 4) Node metrics from flows
  // ------------------------------------------------------------

  const nodeStatus: Record<string, NodeStatus> = {};
  NODES.forEach((n) => {
    nodeStatus[n.id] = {
      netPowerMW: 0,
      isDisconnected: true,
      isOverloaded: false,
    };
  });

  for (const line of LINES) {
    const f = flows[line.id];
    const flow = f.flow_MW;

    nodeStatus[line.from].netPowerMW -= flow;
    nodeStatus[line.to].netPowerMW += flow;

    if (Math.abs(flow) > 1e-3) {
      nodeStatus[line.from].isDisconnected = false;
      nodeStatus[line.to].isDisconnected = false;
    }
    if (f.color === "red") {
      nodeStatus[line.from].isOverloaded = true;
      nodeStatus[line.to].isOverloaded = true;
    }
  }

  NODES.forEach((n) => {
    const st = nodeStatus[n.id];
    const limit = 0.9 * n.capacity_MW;
    if (Math.abs(st.netPowerMW) > limit) {
      st.isOverloaded = true;
    }
  });

  const lineRows = LINES.map((line) => {
    const f = flows[line.id];
    const status =
      f.color === "red"
        ? "Emergency"
        : f.color === "yellow"
        ? "Warning"
        : "Normal";
    return {
      id: line.id,
      from: line.from,
      to: line.to,
      rating: line.rating_MW,
      flow: f.flow_MW,
      loading: f.loading,
      status,
    };
  }).sort((a, b) => b.loading - a.loading);

  const activeTooltip = pinnedInfo ?? hoverInfo;

  const handleHover = (info: {
    node: GridNode | null;
    x: number;
    y: number;
  }) => {
    if (!info.node) {
      setHoverInfo(null);
      return;
    }
    setHoverInfo({ node: info.node, x: info.x, y: info.y });
  };

  const handleClickNode = (info: {
    node: GridNode | null;
    x: number;
    y: number;
  }) => {
    if (!info.node) {
      setPinnedInfo(null);
      return;
    }
    setPinnedInfo({ node: info.node, x: info.x, y: info.y });
  };

  return (
    <div>
      <h2 style={{ marginBottom: 8 }}>Spain Grid View</h2>
      <p style={{ opacity: 0.8, marginBottom: 8 }}>
        Synthetic grid with wind, PV, fossil, nuclear, BESS and loads. Pulses
        move along lines in the direction of power flow; rare overloads and
        faults are shown in yellow/red.
      </p>

      {/* mode selector */}
      <div style={{ marginBottom: 12, fontSize: 13 }}>
        <span style={{ marginRight: 8, opacity: 0.8 }}>Flow mode:</span>
        <button
          onClick={() => setFlowMode("sim")}
          style={{
            padding: "4px 10px",
            borderRadius: "999px 0 0 999px",
            border:
              flowMode === "sim"
                ? "1px solid #42523e"
                : "1px solid #42523e",
            background:
              flowMode === "sim" ? "#75896b" : "#232920",
            color: "#f5f5f5",
            cursor: "pointer",
            fontSize: 12,
          }}
        >
          Simulation
        </button>
        <button
          onClick={() => setFlowMode("forecast")}
          style={{
            padding: "4px 10px",
            borderRadius: "0 999px 999px 0",
            border:
              flowMode === "forecast"
                ? "1px solid #75896b"
                : "1px solid #444c5a",
            borderLeft: "none",
            background:
              flowMode === "forecast" ? "#75896b" : "#232920",
            color: "#f5f5f5",
            cursor: "pointer",
            fontSize: 12,
          }}
        >
          Forecast-driven
        </button>
        <span style={{ marginLeft: 10, opacity: 0.7, fontSize: 11 }}>
          (driven by PV &amp; wind forecast at current time)
        </span>
      </div>

      {forecastError && flowMode === "forecast" && (
        <p style={{ color: "#e74c3c", fontSize: 12, marginBottom: 4 }}>
          {forecastError}
        </p>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "2.1fr 1.2fr",
          gap: 16,
          marginTop: 12,
        }}
      >
        {/* LEFT: canvas + tooltip + legend */}
        <div style={{ position: "relative" }}>
          <GridCanvas
            tHours={tHours}
            flows={flows}
            onHover={handleHover}
            onClickNode={handleClickNode}
            typeVisibility={typeVisibility}
          />

          {activeTooltip && (
            <Tooltip
              info={activeTooltip}
              status={nodeStatus[activeTooltip.node.id]}
            />
          )}

          {/* legend under map */}
          <div style={{ marginTop: 12, fontSize: 12 }}>
            <div style={{ opacity: 0.8, marginBottom: 4 }}>
              Asset legend (click to dim / undim):
            </div>
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: 8,
              }}
            >
              {LEGEND_ITEMS.map((item) => {
                const active = typeVisibility[item.key];
                return (
                  <button
                    key={item.key}
                    onClick={() => toggleTypeVisibility(item.key)}
                    style={{
                      padding: "4px 8px",
                      fontSize: 12,
                      borderRadius: 999,
                      border: "1px solid #42523e",
                      cursor: "pointer",
                      background: active ? "#75896b" : "#232920",
                      color: "#f5f5f5",
                      opacity: active ? 1 : 0.6,
                    }}
                  >
                    {item.label}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* RIGHT: table */}
        <div
          style={{
            background: "#232920",
            borderRadius: 12,
            padding: 12,
            fontSize: 13,
            maxHeight: 600,
            overflow: "auto",
          }}
        >
          <h3>Line loading snapshot</h3>
          <p style={{ opacity: 0.75, fontSize: 12 }}>
            Live clock (local): <strong>{nowClock}</strong>
          </p>

          {activeTooltip && (
            <p style={{ opacity: 0.9, fontSize: 12, marginTop: 4 }}>
              Selected asset:{" "}
              <strong>{activeTooltip.node.name}</strong>{" "}
              <span style={{ opacity: 0.8 }}>
                ({activeTooltip.node.id})
              </span>
            </p>
          )}

          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              marginTop: 8,
            }}
          >
            <thead>
              <tr>
                <Th>Line</Th>
                <Th>From</Th>
                <Th>To</Th>
                <Th>Rating (MW)</Th>
                <Th>Flow (MW)</Th>
                <Th>Loading (%)</Th>
                <Th>Status</Th>
              </tr>
            </thead>
            <tbody>
              {lineRows.map((row) => (
                <tr key={row.id}>
                  <Td>{row.id}</Td>
                  <Td>{row.from}</Td>
                  <Td>{row.to}</Td>
                  <Td>{row.rating}</Td>
                  <Td>{row.flow.toFixed(1)}</Td>
                  <Td>{(row.loading * 100).toFixed(1)}</Td>
                  <Td>
                    <span
                      style={{
                        color:
                          row.status === "Emergency"
                            ? "#e74c3c"
                            : row.status === "Warning"
                            ? "#f1c40f"
                            : "#2ecc71",
                      }}
                    >
                      {row.status}
                    </span>
                  </Td>
                </tr>
              ))}
            </tbody>
          </table>

          <div style={{ marginTop: 12, fontSize: 12, opacity: 0.8 }}>
            <div>● Green: normal loading</div>
            <div>● Yellow: high loading (warning)</div>
            <div>● Red: near/over rating or fault (emergency)</div>
          </div>
        </div>
      </div>
    </div>
  );
};

const Th: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <th
    style={{
      textAlign: "left",
      padding: "4px 6px",
      borderBottom: "1px solid #75896b",
      fontWeight: 600,
    }}
  >
    {children}
  </th>
);

const Td: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <td
    style={{
      padding: "3px 6px",
      borderBottom: "1px solid #75896b",
      fontVariantNumeric: "tabular-nums",
    }}
  >
    {children}
  </td>
);

const Tooltip: React.FC<{
  info: TooltipInfo;
  status: NodeStatus;
}> = ({ info, status }) => {
  const node = info.node;
  const net = status.netPowerMW;

  let powerText: string;
  if (net > 1) {
    powerText = `Net import: ${net.toFixed(1)} MW`;
  } else if (net < -1) {
    powerText = `Net export: ${Math.abs(net).toFixed(1)} MW`;
  } else {
    powerText = "Balanced (≈ 0 MW net)";
  }

  let statusText: string;
  let statusColor: string;
  if (status.isDisconnected) {
    statusText = "Disconnected";
    statusColor = "#95a5a6";
  } else if (status.isOverloaded) {
    statusText = "Connected – Overloaded";
    statusColor = "#e74c3c";
  } else {
    statusText = "Connected";
    statusColor = "#2ecc71";
  }

  return (
    <div
      style={{
        position: "absolute",
        left: info.x + 16,
        top: info.y + 16,
        transform: "translate(-50%, -100%)",
        background: "rgba(10, 16, 30, 0.95)",
        borderRadius: 8,
        padding: "8px 10px",
        fontSize: 12,
        pointerEvents: "none",
        boxShadow: "0 8px 16px rgba(0,0,0,0.35)",
        border: "1px solid #1f2a3d",
        maxWidth: 220,
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{node.name}</div>
      <div style={{ opacity: 0.85, marginBottom: 2 }}>
        ID: <code>{node.id}</code>
      </div>
      <div style={{ opacity: 0.85, marginBottom: 2 }}>
        Capacity: {node.capacity_MW} MW
      </div>
      <div style={{ opacity: 0.9, marginBottom: 2 }}>{powerText}</div>
      <div style={{ color: statusColor }}>{statusText}</div>
    </div>
  );
};

export default GridView;