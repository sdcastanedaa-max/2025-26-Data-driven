// frontend/src/ForecastWindowContext.tsx
import React, {
  createContext,
  useContext,
  useState,
  type ReactNode,
} from "react";

type ForecastWindowContextValue = {
  startDate: Date;
  horizonHours: number;
  setStartDate: (d: Date) => void;
  setHorizonHours: (h: number) => void;
};

// Default: 3-day window starting 2025-01-01
const DEFAULT_START = new Date("2025-01-01T00:00:00");
const DEFAULT_HORIZON = 72;

const ForecastWindowContext =
  createContext<ForecastWindowContextValue | undefined>(undefined);

export const ForecastWindowProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [startDate, setStartDate] = useState<Date>(DEFAULT_START);
  const [horizonHours, setHorizonHours] =
    useState<number>(DEFAULT_HORIZON);

  const value: ForecastWindowContextValue = {
    startDate,
    horizonHours,
    setStartDate,
    setHorizonHours,
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
