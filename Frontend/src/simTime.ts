// src/simTime.ts
import { useEffect, useState } from "react";

type Subscriber = (tHours: number) => void;

let subscribers = new Set<Subscriber>();
let started = false;
let currentTHours = 0;

// start a single global RAF loop (independent of React components)
function startClock() {
  if (started) return;
  started = true;

  let lastTs = performance.now();

  const loop = (ts: number) => {
    const dtSeconds = (ts - lastTs) / 1000;
    lastTs = ts;

    // 60x real-time: 1 real second = 1 simulated minute
    currentTHours += (dtSeconds * 60) / 3600;

    // notify any listeners
    subscribers.forEach((fn) => fn(currentTHours));

    requestAnimationFrame(loop);
  };

  requestAnimationFrame(loop);
}

// Hook that lets any component read the global sim time
export function useSimTime(): number {
  const [tHours, setTHours] = useState<number>(currentTHours);

  useEffect(() => {
    startClock();

    const sub: Subscriber = (t) => setTHours(t);
    subscribers.add(sub);
    return () => {
      subscribers.delete(sub);
    };
  }, []);

  return tHours;
}
