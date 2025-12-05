// frontend/src/simulation.ts
import type { GridNode, GridLine } from "./gridData";
import { NODES, LINES } from "./gridData";

export type FlowStatus = "green" | "yellow" | "red";

export interface LineFlow {
  flow_MW: number;
  loading: number;   // |flow| / rating
  color: FlowStatus; // green / yellow / red
  direction: number; // +1 or -1 (from -> to sign)
}

// -----------------------------------------------------------------------------
// FORECAST TOTALS + ALLOCATION
// -----------------------------------------------------------------------------

export type ForecastTotals = {
  pvMW: number | null;
  windMW: number | null;
};

/**
 * Given system-level forecast totals (MW) for PV and Wind,
 * allocate them to individual nodes in proportion to installed capacity.
 *
 * Returns a map: nodeId -> injection_MW (only PV/Wind nodes get > 0).
 */
export function allocateForecastToNodes(
  totals: ForecastTotals | undefined
): Record<string, number> {
  const injections: Record<string, number> = {};

  // Default all nodes to 0
  NODES.forEach((n) => {
    injections[n.id] = 0;
  });

  if (!totals) return injections;

  // ------- PV allocation -------
  const pvNodes = NODES.filter((n) => n.type === "pv");
  const totalPvCap = pvNodes.reduce(
    (acc, n) => acc + (n.capacity_MW ?? 0),
    0
  );

  if (totals.pvMW != null && totalPvCap > 0) {
    pvNodes.forEach((n) => {
      const cap = n.capacity_MW ?? 0;
      if (cap <= 0) return;

      const share = cap / totalPvCap;
      let p = share * totals.pvMW;

      // clamp to [0, capacity]
      if (p < 0) p = 0;
      if (p > cap) p = cap;

      injections[n.id] = p;
    });
  }

  // ------- Wind allocation -------
  const windNodes = NODES.filter((n) => n.type === "wind");
  const totalWindCap = windNodes.reduce(
    (acc, n) => acc + (n.capacity_MW ?? 0),
    0
  );

  if (totals.windMW != null && totalWindCap > 0) {
    windNodes.forEach((n) => {
      const cap = n.capacity_MW ?? 0;
      if (cap <= 0) return;

      const share = cap / totalWindCap;
      let p = share * totals.windMW;

      if (p < 0) p = 0;
      if (p > cap) p = cap;

      // "+=" just in case some node is hybrid PV+wind in the toy world
      injections[n.id] += p;
    });
  }

  return injections;
}

// -----------------------------------------------------------------------------
// STEP A: NODE-LEVEL INJECTIONS (SIM MODE)
// -----------------------------------------------------------------------------
//
// Positive = generation (MW), negative = load (MW).
// Values are clamped to ±capacity_MW so assets can't violate their rating.
// -----------------------------------------------------------------------------

export function computeSimInjections(tHours: number): Record<string, number> {
  const hour = tHours % 24;
  const inj: Record<string, number> = {};

  for (const n of NODES) {
    let p = 0;
    const cap = n.capacity_MW ?? 0;

    if (n.type === "pv") {
      // simple PV bell curve: zero at night, peak at midday
      const pvShape = Math.max(
        0,
        Math.sin(((hour - 6) / 24) * 2 * Math.PI)
      );
      p = pvShape * cap; // generation ≥ 0
    } else if (n.type === "wind") {
      // wind: gently varying around ~60–80% of capacity
      const w =
        0.6 +
        0.2 * Math.sin((2 * Math.PI * tHours) / 24) +
        0.1 * Math.sin((2 * Math.PI * tHours) / 6);
      p = Math.max(0, w) * cap;
    } else if (n.type === "fossil" || n.type === "nuclear") {
      // baseload-ish around 70–90% of capacity
      const base = 0.75 + 0.1 * Math.sin((2 * Math.PI * tHours) / 24);
      p = base * cap;
    } else if (n.type === "load_res" || n.type === "load_ind") {
      // loads: negative injections with daily pattern
      const d =
        0.7 +
        0.15 * Math.sin((2 * Math.PI * hour) / 24) +
        0.05 * Math.sin((2 * Math.PI * hour) / 2);
      p = -d * cap; // negative = consumption
    } else {
      // substations / bess: neutral by default in this toy
      p = 0;
    }

    // hard clamp so nothing exceeds nameplate rating
    if (cap > 0) {
      p = Math.min(Math.max(p, -cap), cap);
    }

    inj[n.id] = p;
  }

  return inj;
}

// -----------------------------------------------------------------------------
// EVENT SYSTEM: overloads + faults
// -----------------------------------------------------------------------------

interface LineEvent {
  lineId: string;
  kind: "overload" | "fault";
  startTHours: number;
  endTHours: number;
}

let activeEvent: LineEvent | null = null;
let lastEventCheckTHours: number | null = null;

function maybeUpdateEvent(tHours: number, lines: GridLine[]) {
  // keep existing event if still active
  if (activeEvent && tHours < activeEvent.endTHours) {
    return;
  }
  // clear if expired
  if (activeEvent && tHours >= activeEvent.endTHours) {
    activeEvent = null;
  }

  // only consider starting new event every ~0.1 simulated hours
  if (
    lastEventCheckTHours !== null &&
    tHours - lastEventCheckTHours < 0.1
  ) {
    return;
  }
  lastEventCheckTHours = tHours;

  // small probability that *any* event starts
  const pNewEvent = 0.02; // ~2% per check (rare)
  if (Math.random() > pNewEvent) return;

  const idx = Math.floor(Math.random() * lines.length);
  const line = lines[idx];

  const isFault = Math.random() < 0.25; // 25% of events are faults
  const kind: "overload" | "fault" = isFault ? "fault" : "overload";

  // very short durations so emergencies clear quickly
  const duration = kind === "fault" ? 0.08 : 0.12; // hours

  activeEvent = {
    lineId: line.id,
    kind,
    startTHours: tHours,
    endTHours: tHours + duration,
  };
}

// -----------------------------------------------------------------------------
// STEP B: MAIN SIMULATION – FLOWS FROM NODE INJECTIONS
// -----------------------------------------------------------------------------
//
// simulateLineFlows(tHours)                         -> sim mode
// simulateLineFlows(tHours, { useForecast, forecastTotals }) -> RES-driven
// -----------------------------------------------------------------------------

export function simulateLineFlows(
  tHours: number,
  opts?: {
    useForecast?: boolean;
    forecastTotals?: ForecastTotals;
  }
): Record<string, LineFlow> {
  const nodeById: Record<string, GridNode> = {};
  NODES.forEach((n) => {
    nodeById[n.id] = n;
  });

  const useForecast = opts?.useForecast ?? false;
  const forecastTotals = opts?.forecastTotals;

  // 1) Per-node injections: either forecast-based or simulated
  const injections: Record<string, number> =
    useForecast && forecastTotals
      ? allocateForecastToNodes(forecastTotals)
      : computeSimInjections(tHours);

  // 2) Update rare events (overloads/faults)
  maybeUpdateEvent(tHours, LINES);

  const flows: Record<string, LineFlow> = {};

  // 3) Build line flows from injection differences
  for (const line of LINES) {
    const nFrom = nodeById[line.from];
    const nTo = nodeById[line.to];

    const pFrom = injections[nFrom.id] ?? 0;
    const pTo = injections[nTo.id] ?? 0;

    // raw flow proportional to injection imbalance
    let rawFlow = (pFrom - pTo) * 0.5; // 0.5 is a tuning knob

    // clamp to line rating
    rawFlow = Math.min(Math.max(rawFlow, -line.rating_MW), line.rating_MW);

    let loading = Math.abs(rawFlow) / line.rating_MW;

    // --- Apply event (overload or fault) -------------------------
    let eventColorOverride: FlowStatus | null = null;
    let forceZero = false;

    if (activeEvent && activeEvent.lineId === line.id) {
      const ev = activeEvent;
      const dur = Math.max(ev.endTHours - ev.startTHours, 1e-6);
      const prog = Math.min(
        Math.max((tHours - ev.startTHours) / dur, 0),
        1
      ); // 0..1

      if (ev.kind === "overload") {
        // overload: start strong red, fade through yellow back to base
        const boost = 0.4 * (1 - prog); // strong at start, 0 at end
        loading = loading * (1 + boost);
        if (prog < 0.3) eventColorOverride = "red";
        else if (prog < 0.8) eventColorOverride = "yellow";
      } else {
        // fault: open line briefly, then constrained, then back to normal
        if (prog < 0.3) {
          forceZero = true;
          eventColorOverride = "red";
        } else if (prog < 0.7) {
          loading = Math.max(0.3, loading * 0.5);
          eventColorOverride = "yellow";
        }
      }
    }

    // final loading cap
    let finalLoading = forceZero ? 0 : loading;
    finalLoading = Math.min(finalLoading, 1.2); // safety cap

    const flow_MW = forceZero
      ? 0
      : Math.sign(rawFlow || 1) * finalLoading * line.rating_MW;

    // --- Color assignment ---------------------------------------
    let color: FlowStatus;
    if (eventColorOverride) {
      color = eventColorOverride;
    } else if (finalLoading > 1.0) {
      color = "red";
    } else if (finalLoading > 0.85) {
      color = "yellow";
    } else {
      color = "green";
    }

    flows[line.id] = {
      flow_MW,
      loading: finalLoading,
      color,
      direction: Math.sign(flow_MW || 1),
    };
  }

  return flows;
}

