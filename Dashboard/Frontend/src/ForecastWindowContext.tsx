// frontend/src/ForecastWindowContext.tsx
import React, {
  createContext,
  useContext,
  useState,
  type ReactNode,
} from "react";

type ForecastWindowContextValue = {
  start: Date | null;
  end: Date | null;
  hours: number | null;
  setWindow: (start: Date, hours: number) => void;
};

// Default 3-day window starting 2025-01-01 (UTC)
const DEFAULT_START = new Date(Date.UTC(2025, 0, 1, 0, 0, 0));
const DEFAULT_HOURS = 72;

const ForecastWindowContext =
  createContext<ForecastWindowContextValue | undefined>(undefined);

export const ForecastWindowProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [start, setStart] = useState<Date | null>(DEFAULT_START);
  const [hours, setHours] = useState<number | null>(DEFAULT_HOURS);
  const [end, setEnd] = useState<Date | null>(
    new Date(DEFAULT_START.getTime() + DEFAULT_HOURS * 3600 * 1000),
  );

  const setWindow = (newStart: Date, newHours: number) => {
    setStart(newStart);
    setHours(newHours);
    setEnd(new Date(newStart.getTime() + newHours * 3600 * 1000));
  };

  const value: ForecastWindowContextValue = {
    start,
    end,
    hours,
    setWindow,
  };

  return (
    <ForecastWindowContext.Provider value={value}>
      {children}
    </ForecastWindowContext.Provider>
  );
};

export const useForecastWindow = (): ForecastWindowContextValue => {
  const ctx = useContext(ForecastWindowContext);
  if (!ctx) {
    throw new Error(
      "useForecastWindow must be used inside ForecastWindowProvider",
    );
  }
  return ctx;
};

