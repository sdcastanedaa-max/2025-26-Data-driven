// frontend/src/simTime.tsx
import React, {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import { useForecastWindow } from "./ForecastWindowContext";

type SimTimeContextValue = {
  tHours: number;
};

const SimTimeContext = createContext<SimTimeContextValue | undefined>(
  undefined,
);

export const SimTimeProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const { start } = useForecastWindow();
  const [tHours, setTHours] = useState(0);
  const lastTsRef = useRef<number | null>(null);

  // Reset the sim clock whenever the forecast window start changes
  useEffect(() => {
    setTHours(0);
    lastTsRef.current = null;
  }, [start?.getTime()]);

  useEffect(() => {
    let frameId: number;

    const loop = (ts: number) => {
      if (lastTsRef.current === null) {
        lastTsRef.current = ts;
      }
      const dtSeconds = (ts - lastTsRef.current) / 1000;
      lastTsRef.current = ts;

      // 60x speed: 1 real second = 1 simulated minute
      setTHours((prev) => prev + (dtSeconds * 60) / 3600);

      frameId = requestAnimationFrame(loop);
    };

    frameId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frameId);
  }, []);

  return (
    <SimTimeContext.Provider value={{ tHours }}>
      {children}
    </SimTimeContext.Provider>
  );
};

export const useSimTime = (): number => {
  const ctx = useContext(SimTimeContext);
  if (!ctx) {
    throw new Error("useSimTime must be used inside <SimTimeProvider>");
  }
  return ctx.tHours;
};

