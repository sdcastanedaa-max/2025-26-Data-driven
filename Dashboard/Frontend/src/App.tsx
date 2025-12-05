// Frontend/src/App.tsx
import React, { useState } from "react";
import ForecastView from "./ForecastView";
import GridView from "./GridView";
import { SimTimeProvider } from "./simTime";
import { ForecastWindowProvider } from "./ForecastWindowContext";

const App: React.FC = () => {
  const [tab, setTab] = useState<"forecast" | "grid">("forecast");

  return (
    <ForecastWindowProvider>
      <SimTimeProvider>
        <div
          style={{
            minHeight: "100vh",
            width: "100vw",        // force full viewport width
            overflowX: "hidden",
            background: "#161a14",
            color: "#f5f5f5",
          }}
        >
          <header
            style={{
              maxWidth: 1900,
              margin: "0 auto",
              padding: "16px 24px 8px",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <h1 style={{ fontSize: 20, fontWeight: 600 }}>
              Demo Dashboard
            </h1>

            <div
              style={{
                display: "inline-flex",
                gap: 8,
                background: "#232920",
                padding: 4,
                borderRadius: 999,
                border: "1px solid #42523e",
              }}
            >
              <button
                onClick={() => setTab("forecast")}
                style={{
                  padding: "4px 16px",
                  borderRadius: 999,
                  border: "none",
                  cursor: "pointer",
                  background:
                    tab === "forecast" ? "#75896b" : "transparent",
                  color: tab === "forecast" ? "#000" : "#f5f5f5",
                  fontSize: 13,
                  fontWeight: 500,
                }}
              >
                Forecast
              </button>
              <button
                onClick={() => setTab("grid")}
                style={{
                  padding: "4px 16px",
                  borderRadius: 999,
                  border: "none",
                  cursor: "pointer",
                  background: tab === "grid" ? "#75896b" : "transparent",
                  color: tab === "grid" ? "#000" : "#f5f5f5",
                  fontSize: 13,
                  fontWeight: 500,
                }}
              >
                Grid
              </button>
            </div>
          </header>

          <main
            style={{
              width: "100%",
              maxWidth: 1900,
              margin: "0 auto",
              padding: "0 24px 24px",
              boxSizing: "border-box",
            }}
          >
            {tab === "forecast" ? <ForecastView /> : <GridView />}
          </main>
        </div>
      </SimTimeProvider>
    </ForecastWindowProvider>
  );
};

export default App;
