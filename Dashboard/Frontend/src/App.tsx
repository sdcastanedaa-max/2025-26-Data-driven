// frontend/src/App.tsx
import React, { useState } from "react";
import ForecastView from "./ForecastView";
import GridView from "./GridView";

type View = "forecast" | "grid";

const App: React.FC = () => {
  const [view, setView] = useState<View>("forecast");

  return (
    <div
      style={{
        background: "#050814",
        minHeight: "100vh",
        width: "100vw",
        color: "#f5f5f5",
        fontFamily: "system-ui, sans-serif",
        overflowX: "hidden",
      }}
    >
      <header
        style={{
          padding: "12px 20px",
          borderBottom: "1px solid #1f2a3d",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          <h1 style={{ margin: 0, fontSize: 20 }}>
            Spain RES Forecast &amp; Grid Demo
          </h1>
          <div style={{ fontSize: 12, opacity: 0.8 }}>
            PV / Wind forecast models + animated grid visualisation
          </div>
        </div>

        <nav
          style={{
            display: "flex",
            gap: 8,
          }}
        >
          <TabButton
            active={view === "forecast"}
            onClick={() => setView("forecast")}
          >
            Forecast
          </TabButton>
          <TabButton
            active={view === "grid"}
            onClick={() => setView("grid")}
          >
            Grid
          </TabButton>
        </nav>
      </header>

            <main
        style={{
          padding: "16px 20px",
          display: "flex",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            width: "100%",
            maxWidth: "1400px", // both Forecast & Grid will use this
          }}
        >
          {view === "forecast" ? <ForecastView /> : <GridView />}
        </div>
      </main>
    </div>
  );
};

const TabButton: React.FC<{
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}> = ({ active, onClick, children }) => (
  <button
    onClick={onClick}
    style={{
      padding: "6px 12px",
      borderRadius: 999,
      border: active ? "1px solid #f1c40f" : "1px solid #1f2a3d",
      background: active ? "#1f2838" : "#101826",
      color: "#f5f5f5",
      cursor: "pointer",
      fontSize: 13,
    }}
  >
    {children}
  </button>
);

export default App;

