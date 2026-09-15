import { useState } from "react";
import Home from "./components/Home";
import TacticalSimulator from "./components/TacticalSimulator";
import PassNetwork from "./components/PassNetwork";
import PlayerHeatMap from "./components/PlayerHeatmap";
import ComputerVision from "./components/ComputerVision";

export type Screen =
  | "home"
  | "simulator"
  | "passNetwork"
  | "heatmaps"
  | "vision";

const NAV_ITEMS: { id: Screen; label: string }[] = [
  { id: "simulator", label: "ML Simulator" },
  { id: "passNetwork", label: "Pass Networks" },
  { id: "heatmaps", label: "Heatmaps" },
  { id: "vision", label: "Vision" },
];

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<Screen>("home");

  const renderScreen = () => {
    switch (currentScreen) {
      case "home":
        return <Home onNavigate={setCurrentScreen} />;
      case "simulator":
        return <TacticalSimulator />;
      case "passNetwork":
        return <PassNetwork />;
      case "heatmaps":
        return <PlayerHeatMap />;
      case "vision":
        return <ComputerVision />;
      default:
        return <Home onNavigate={setCurrentScreen} />;
    }
  };

  return (
    <div className="min-h-screen w-full bg-slate-950 flex flex-col font-sans text-slate-50">
      <nav className="bg-slate-950/95 backdrop-blur border-b border-slate-800/80 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <button
              className="flex items-center gap-2.5 group"
              onClick={() => setCurrentScreen("home")}
            >
              <div className="h-8 w-8 rounded-md bg-slate-900 border border-slate-700 flex items-center justify-center">
                <svg
                  className="w-4 h-4 text-amber-400"
                  viewBox="0 0 24 24"
                  fill="none"
                >
                  <rect
                    x="1"
                    y="1"
                    width="22"
                    height="22"
                    rx="2"
                    stroke="currentColor"
                    strokeWidth="1.3"
                  />
                  <line
                    x1="12"
                    y1="1"
                    x2="12"
                    y2="23"
                    stroke="currentColor"
                    strokeWidth="1.3"
                  />
                  <circle
                    cx="12"
                    cy="12"
                    r="4.5"
                    stroke="currentColor"
                    strokeWidth="1.3"
                  />
                </svg>
              </div>
              <span className="font-display font-semibold text-[15px] tracking-tight text-slate-100 hidden sm:block">
                Football Analytics{" "}
              </span>
            </button>

            <div className="flex items-center gap-1">
              {NAV_ITEMS.map((item) => {
                const isActive = currentScreen === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => setCurrentScreen(item.id)}
                    className={`relative px-3 py-2 text-sm font-medium transition-colors duration-150 ${
                      isActive
                        ? "text-slate-50"
                        : "text-slate-400 hover:text-slate-100"
                    }`}
                  >
                    <span className="hidden md:inline">{item.label}</span>
                    <span className="md:hidden">
                      {item.label.split(" ")[0]}
                    </span>
                    {isActive && (
                      <span className="absolute left-3 right-3 -bottom-[1px] h-[2px] bg-amber-400 rounded-full" />
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </nav>

      <main className="flex-grow flex flex-col relative overflow-hidden">
        {renderScreen()}
      </main>
    </div>
  );
}
