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
// NEW: forecast totals + allocation helper
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

      // += just in case a node is "hybrid" in your toy world
      injections[n.id] += p;
    });
  }

  return injections;
}

// -----------------------------------------------------------------------------
// NODE SIGNS
//   +1 : pure generators (wind, pv, fossil, nuclear)
//   -1 : pure loads (load_res, load_ind)
//    0 : substations and BESS (can do both, neutral hubs)
// -----------------------------------------------------------------------------

const nodeSign = (type: GridNode["type"]): number => {
  if (["wind", "pv", "fossil", "nuclear"].includes(type)) return 1;
  if (["load_res", "load_ind"].includes(type)) return -1;
  // substations + bess are neutral
  return 0;
};

// small deterministic hash per-line for diversity
function hashString(str: string): number {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = (h * 31 + str.charCodeAt(i)) | 0;
  }
  return h;
}

// -----------------------------------------------------------------------------
// EVENT SYSTEM: overloads + faults
//   - At most 1 active event at a time
//   - "overload": line goes red briefly, then yellow, then back to normal
//   - "fault"   : line opens (0 MW) for a very short time, then limited yellow,
//                 then back to normal
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
// MAIN SIMULATION
// -----------------------------------------------------------------------------
//
// Extended signature:
//   simulateLineFlows(tHours)                      // old behavior
//   simulateLineFlows(tHours, { useForecast, forecastTotals }) // new option
// -----------------------------------------------------------------------------

export function simulateLineFlows(
  tHours: number,
  opts?: {
    useForecast?: boolean;
    forecastTotals?: ForecastTotals;
  }
): Record<string, LineFlow> {
  const nodeById: Record<string, GridNode> = {};
  NODES.forEach((n) => (nodeById[n.id] = n));

  const useForecast = opts?.useForecast ?? false;
  const forecastTotals = opts?.forecastTotals;

  // capacity-weighted node injections for PV/Wind if forecast mode is on
  const forecastNodeInjection: Record<string, number> | null =
    useForecast && forecastTotals
      ? allocateForecastToNodes(forecastTotals)
      : null;

  maybeUpdateEvent(tHours, LINES);

  const flows: Record<string, LineFlow> = {};
  const hour = tHours % 24.0;

  for (const line of LINES) {
    const nFrom = nodeById[line.from];
    const nTo = nodeById[line.to];

    const signFrom = nodeSign(nFrom.type);
    const signTo = nodeSign(nTo.type);

    // GENERATION → LOAD direction preference
    const direction =
      Math.sign(signFrom - signTo) !== 0
        ? Math.sign(signFrom - signTo)
        : 1; // default

    // --- Base "healthy" loading profile (steady, mostly green) ---

    const baseScale =
      (Math.abs(nFrom.capacity_MW) + Math.abs(nTo.capacity_MW)) / 2.0;

    // PV: daytime bump
    const pvFactor = Math.max(
      0,
      Math.sin(((hour - 6.0) / 24.0) * 2.0 * Math.PI),
    );

    // technology-specific variability
    let techFactor: number;
    if (nFrom.type === "pv" || nTo.type === "pv") {
      techFactor = 0.3 + 0.7 * pvFactor; // PV low at night, high mid-day
    } else if (nFrom.type === "wind" || nTo.type === "wind") {
      techFactor =
        0.6 +
        0.2 * Math.sin((2.0 * Math.PI * tHours) / 24.0) +
        0.1 * Math.sin((2.0 * Math.PI * tHours) / 6.0);
    } else if (
      nFrom.type === "fossil" ||
      nTo.type === "fossil" ||
      nFrom.type === "nuclear" ||
      nTo.type === "nuclear"
    ) {
      techFactor =
        0.8 +
        0.1 * Math.sin((2.0 * Math.PI * tHours) / 24.0); // baseload-ish
    } else {
      techFactor =
        0.5 + 0.1 * Math.sin((2.0 * Math.PI * tHours) / 12.0);
    }

    // line-specific factor
    const hash = Math.abs(hashString(line.id)) % 1000;
    const lineFactor = 0.9 + 0.2 * (hash / 1000.0); // ~0.9..1.1

    // small wiggle just for liveliness
    const wiggle = 1.0 + 0.03 * Math.sin(2.0 * Math.PI * tHours);

    // choose magnitude so base loading typically 20–70% of rating
    const baseMagnitude =
      0.18 * baseScale * techFactor * lineFactor * wiggle;

    let baseFlow = direction * baseMagnitude;
    let baseLoading = Math.abs(baseFlow) / line.rating_MW;

    // clamp baseLoading to avoid natural permanent yellow/red
    baseLoading = Math.min(baseLoading, 0.85);

    // --- NEW: adjust loading based on forecast injections on PV/Wind ----
    if (forecastNodeInjection) {
      // For each endpoint, compute how "full" PV/Wind plants are vs capacity
      const ratioForNode = (n: GridNode): number => {
        if (n.type !== "pv" && n.type !== "wind") {
          return 1.0; // non-RES node: neutral
        }
        const cap = n.capacity_MW || 0;
        if (cap <= 0) return 1.0;

        const inj = forecastNodeInjection[n.id] ?? 0;
        const r = inj / cap;
        // clamp to 0..1.5 to avoid crazy extremes
        return Math.min(Math.max(r, 0), 1.5);
      };

      const rFrom = ratioForNode(nFrom);
      const rTo = ratioForNode(nTo);
      const avgRatio = (rFrom + rTo) / 2.0;

      // scale loading based on avgRatio:
      //  - ~0 generation => ~0.5x of normal
      //  - full generation => ~1.2x of normal (but still capped later)
      const scale = 0.5 + 0.7 * avgRatio; // 0.5 .. ~1.55
      baseLoading *= scale;
    }

    // --- Apply event (overload or fault) -------------------------

    let eventColorOverride: FlowStatus | null = null;
    let forceZero = false;

    if (activeEvent && activeEvent.lineId === line.id) {
      const ev = activeEvent;
      const dur = Math.max(ev.endTHours - ev.startTHours, 1e-6);
      const prog = Math.min(
        Math.max((tHours - ev.startTHours) / dur, 0),
        1,
      ); // 0..1 within event window

      if (ev.kind === "overload") {
        // overload: start strong red, fade through yellow back to base
        const boost = 0.6 * (1 - prog); // strong at start, 0 at end
        baseLoading *= 1 + boost;

        if (prog < 0.4) eventColorOverride = "red";
        else if (prog < 0.8) eventColorOverride = "yellow";
        // last 20%: back to normal color from loading
      } else {
        // fault: open line briefly, then limited yellow, then normal
        if (prog < 0.3) {
          // just tripped: open
          forceZero = true;
          eventColorOverride = "red";
        } else if (prog < 0.7) {
          // reclosed but constrained
          baseLoading = Math.max(0.3, baseLoading * 0.4);
          eventColorOverride = "yellow";
        } else {
          // final 30%: back to normal
        }
      }
    }

    // --- Final loading, enforcing min >= 15% except during open fault ----

    let loading: number;
    if (forceZero) {
      loading = 0;
    } else {
      loading = baseLoading;
      loading = Math.min(loading, 1.2); // safety cap
      // keep every *healthy* line at least 15% loaded
      loading = Math.max(0.15, loading);
    }

    const flow_MW = direction * loading * line.rating_MW;

    // --- Color assignment ---------------------------------------

    let color: FlowStatus;
    if (eventColorOverride) {
      color = eventColorOverride;
    } else if (loading > 1.0) {
      color = "red";
    } else if (loading > 0.85) {
      color = "yellow";
    } else {
      color = "green";
    }

    flows[line.id] = {
      flow_MW,
      loading,
      color,
      direction,
    };
  }

  return flows;
}
