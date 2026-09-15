import React, { useState, useEffect, useRef, useCallback } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

interface TacticalState {
  defensiveHeight: number;
  pressureIntensity: number;
  possession: number;
  width: number;
}

interface Metrics {
  xgFor: number;
  xgAgainst: number;
  shapFor?: any;
  shapAgainst?: any;
}

const PITCH_L = 105;
const PITCH_W = 68;

interface ControlSliderProps {
  label: string;
  val: number;
  min: number;
  max: number;
  statKey: keyof TacticalState;
  desc?: string;
  onChange: (
    e: React.ChangeEvent<HTMLInputElement>,
    key: keyof TacticalState,
  ) => void;
}

const ControlSlider: React.FC<ControlSliderProps> = ({
  label,
  val,
  min,
  max,
  statKey,
  desc,
  onChange,
}) => (
  <div className="flex flex-col gap-2">
    <div className="flex justify-between items-center">
      <label className="text-sm font-medium text-slate-300">{label}</label>
      <span className="text-sm font-bold text-sky-400 bg-sky-400/10 px-2 py-0.5 rounded">
        {val}
        {statKey === "possession"
          ? "%"
          : statKey === "pressureIntensity"
            ? ""
            : "m"}
      </span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      value={val}
      onChange={(e) => onChange(e, statKey)}
      style={{ accentColor: "#38bdf8" }}
      className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer"
    />
    {desc && <p className="text-xs text-slate-500">{desc}</p>}
  </div>
);

const ShapExplanation = ({ data, isFor }: { data: any; isFor: boolean }) => {
  if (!data || !data.features) return null;

  const sortedFeatures = [...data.features].sort(
    (a, b) => Math.abs(b.impact) - Math.abs(a.impact),
  );

  return (
    <div className="mt-4 pt-4 border-t border-slate-700/50">
      <div className="flex justify-between items-center text-[10px] uppercase font-bold tracking-wider text-slate-500 mb-3">
        <span>Base League xG</span>
        <span className="text-slate-300">{data.base_value.toFixed(2)}</span>
      </div>
      <div className="flex flex-col gap-2">
        {sortedFeatures.map((f, idx) => {
          const isPositiveImpact = f.impact > 0;
          const colorClass = isFor
            ? isPositiveImpact
              ? "text-emerald-400"
              : "text-rose-400"
            : isPositiveImpact
              ? "text-rose-400"
              : "text-emerald-400";
          return (
            <div
              key={idx}
              className="flex justify-between items-center text-xs"
            >
              <span className="text-slate-400 capitalize">
                {f.name.replace("_", " ")}
              </span>
              <span className={`font-mono font-medium ${colorClass}`}>
                {isPositiveImpact ? "+" : ""}
                {f.impact.toFixed(2)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default function TacticalSimulator() {
  const [tactics, setTactics] = useState<TacticalState>({
    defensiveHeight: 45,
    pressureIntensity: 12,
    possession: 50,
    width: 50,
  });

  const [metrics, setMetrics] = useState<Metrics>({
    xgFor: 0.0,
    xgAgainst: 0.0,
  });

  const [isLoading, setIsLoading] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationFrameId = useRef<number>(0);
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchPredictions = useCallback(
    async (currentTactics: TacticalState) => {
      setIsLoading(true);
      try {
        const response = await fetch(
          `${API_BASE_URL}/simulator/predict`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              defensiveHeight: currentTactics.defensiveHeight,
              pressureIntensity: currentTactics.pressureIntensity,
              possession: currentTactics.possession,
              width: currentTactics.width,
            }),
          },
        );

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        setMetrics({
          xgFor: data.xg_for,
          xgAgainst: data.xg_against,
          shapFor: data.shap_for,
          shapAgainst: data.shap_against,
        });
      } catch (error) {
        console.error("Error fetching ML predictions:", error);
      } finally {
        setIsLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(() => {
      fetchPredictions(tactics);
    }, 200);

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [tactics, fetchPredictions]);

  const drawPitch = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    if (
      canvas.width !== rect.width * dpr ||
      canvas.height !== rect.height * dpr
    ) {
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
    }

    ctx.resetTransform();
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);

    const padding = 20;
    const availW = w - padding * 2;
    const availH = h - padding * 2;

    let pW = availW;
    let pH = pW * (PITCH_W / PITCH_L);

    if (pH > availH) {
      pH = availH;
      pW = pH * (PITCH_L / PITCH_W);
    }

    const ox = (w - pW) / 2;
    const oy = (h - pH) / 2;

    ctx.fillStyle = "#0f172a";
    ctx.fillRect(ox, oy, pW, pH);

    const stripes = 10;
    const stripeW = pW / stripes;
    for (let i = 0; i < stripes; i++) {
      ctx.fillStyle =
        i % 2 === 0 ? "rgba(255,255,255,0.02)" : "rgba(255,255,255,0.04)";
      ctx.fillRect(ox + i * stripeW, oy, stripeW, pH);
    }

    ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(ox, oy, pW, pH);

    ctx.beginPath();
    ctx.moveTo(ox + pW / 2, oy);
    ctx.lineTo(ox + pW / 2, oy + pH);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(ox + pW / 2, oy + pH / 2, pH * 0.15, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(ox + pW / 2, oy + pH / 2, 2, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,255,255,0.6)";
    ctx.fill();

    const pAlpha = ((tactics.pressureIntensity - 5) / (20 - 5)) * 0.5;
    const grad = ctx.createLinearGradient(ox + pW / 2, 0, ox + pW, 0);
    grad.addColorStop(0, "rgba(239, 68, 68, 0)");
    grad.addColorStop(1, `rgba(239, 68, 68, ${pAlpha})`);
    ctx.fillStyle = grad;
    ctx.fillRect(ox + pW / 2, oy, pW / 2, pH);

    const wFactor = 0.3 + ((tactics.width - 30) / (80 - 30)) * 0.65;
    const vSpread = pH * wFactor;
    const vStart = oy + (pH - vSpread) / 2;
    const vEnd = vStart + vSpread;

    ctx.setLineDash([6, 6]);
    ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(ox, vStart);
    ctx.lineTo(ox + pW, vStart);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(ox, vEnd);
    ctx.lineTo(ox + pW, vEnd);
    ctx.stroke();
    ctx.setLineDash([]);

    const defLineX = ox + (tactics.defensiveHeight / PITCH_L) * pW;
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(defLineX, vStart);
    ctx.lineTo(defLineX, vEnd);
    ctx.shadowBlur = 10;
    ctx.shadowColor = "#38bdf8";
    ctx.stroke();
    ctx.shadowBlur = 0;

    const badgeText = `${tactics.defensiveHeight}m`;
    ctx.font = "600 11px system-ui, -apple-system, sans-serif";
    const textW = ctx.measureText(badgeText).width;
    const badgeW = textW + 16;
    const badgeH = 20;

    ctx.fillStyle = "#0f172a";
    ctx.fillRect(defLineX - badgeW / 2, vStart - 26, badgeW, badgeH);
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1;
    ctx.strokeRect(defLineX - badgeW / 2, vStart - 26, badgeW, badgeH);

    ctx.fillStyle = "#38bdf8";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(badgeText, defLineX, vStart - 16);

    ctx.fillStyle = "rgba(56, 189, 248, 0.08)";
    ctx.fillRect(
      Math.max(ox, defLineX - pW * 0.25),
      vStart,
      Math.min(defLineX - ox, pW * 0.25),
      vSpread,
    );
  }, [tactics]);

  useEffect(() => {
    const renderLoop = () => {
      drawPitch();
      animationFrameId.current = requestAnimationFrame(renderLoop);
    };
    renderLoop();

    return () => {
      if (animationFrameId.current)
        cancelAnimationFrame(animationFrameId.current);
    };
  }, [drawPitch]);

  const handleSliderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>, key: keyof TacticalState) => {
      setTactics((prev) => ({
        ...prev,
        [key]: parseFloat(e.target.value),
      }));
    },
    [],
  );

  const getRatioColor = () => {
    const diff = metrics.xgFor - metrics.xgAgainst;
    if (diff > 0.4) return "text-emerald-400";
    if (diff > -0.2) return "text-sky-400";
    return "text-amber-400";
  };

  return (
    <div className="flex-grow w-full p-4 sm:p-8 flex flex-col">
      <div className="max-w-6xl mx-auto w-full flex-grow flex flex-col">
        <header className="mb-8 border-b border-slate-800 pb-6">
          <span className="inline-flex items-center gap-1.5 text-xs font-medium text-amber-400 mb-2">
            <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
            Tactical simulator
          </span>
          <div className="flex items-center gap-3">
            <h1 className="font-display text-2xl font-semibold tracking-tight text-slate-50">
              Predict match outcomes from tactical inputs
            </h1>
            {isLoading && (
              <span className="text-xs font-medium text-sky-400 animate-pulse bg-sky-400/10 px-2 py-1 rounded-md">
                Running inference…
              </span>
            )}
          </div>
          <p className="text-sm text-slate-400 mt-2 max-w-xl">
            XGBoost model interface — adjust the tactical variables below to
            predict Expected Goals based on real match data.
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 flex-grow">
          <div className="lg:col-span-4 flex flex-col gap-7 bg-slate-900/60 p-6 rounded-2xl border border-slate-800">
            <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2 border-b border-slate-800 pb-3">
              <svg
                className="w-4 h-4 text-slate-500"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"
                />
              </svg>
              Model features
            </h2>

            <ControlSlider
              label="Defensive Line Height"
              val={tactics.defensiveHeight}
              min={30}
              max={70}
              statKey="defensiveHeight"
              desc="Average X-coordinate of defensive actions."
              onChange={handleSliderChange}
            />
            <ControlSlider
              label="Pressure Intensity (PPDA)"
              val={tactics.pressureIntensity}
              min={5}
              max={20}
              statKey="pressureIntensity"
              desc="Derived from Passes Allowed Per Defensive Action. Lower is more intense."
              onChange={handleSliderChange}
            />
            <ControlSlider
              label="Target Possession"
              val={tactics.possession}
              min={30}
              max={70}
              statKey="possession"
              onChange={handleSliderChange}
            />
            <ControlSlider
              label="Offensive Width"
              val={tactics.width}
              min={30}
              max={80}
              statKey="width"
              onChange={handleSliderChange}
              desc="Expected field distribution (y-axis spread)."
            />

            <button
              onClick={() =>
                setTactics({
                  defensiveHeight: 45,
                  pressureIntensity: 12,
                  possession: 50,
                  width: 50,
                })
              }
              className="mt-2 w-full py-2.5 bg-slate-700/50 hover:bg-slate-700 text-slate-200 rounded-lg transition-all font-medium text-sm border border-slate-600 focus:outline-none focus:ring-2 focus:ring-slate-500 active:scale-95"
            >
              Reset to Baseline
            </button>
          </div>

          <div className="lg:col-span-8 flex flex-col gap-6">
            <div className="grid grid-cols-2 gap-5">
              <div className="bg-slate-900/60 p-5 rounded-2xl border border-slate-800">
                <p className="text-xs font-medium text-slate-500 mb-1">
                  Predicted xG
                </p>
                <div className="flex items-baseline gap-2">
                  <span
                    className={`font-display text-4xl font-semibold tracking-tight transition-colors duration-300 ${getRatioColor()}`}
                  >
                    {metrics.xgFor.toFixed(2)}
                  </span>
                  <span className="text-sm font-medium text-slate-500">
                    expected goals
                  </span>
                </div>
                <ShapExplanation data={metrics.shapFor} isFor={true} />
              </div>

              <div className="bg-slate-900/60 p-5 rounded-2xl border border-slate-800">
                <p className="text-xs font-medium text-slate-500 mb-1">
                  Conceded xG
                </p>
                <div className="flex items-baseline gap-2">
                  <span className="font-display text-4xl font-semibold tracking-tight text-rose-400 transition-all duration-300">
                    {metrics.xgAgainst.toFixed(2)}
                  </span>
                  <span className="text-sm font-medium text-slate-500">
                    risk exposure
                  </span>
                </div>
                <ShapExplanation data={metrics.shapAgainst} isFor={false} />
              </div>
            </div>

            <div
              ref={containerRef}
              className="flex-grow w-full bg-slate-950 rounded-2xl border border-slate-700/80 shadow-2xl overflow-hidden relative min-h-[400px]"
              style={{ boxShadow: "inset 0 0 40px rgba(0,0,0,0.5)" }}
            >
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full block"
              />
            </div>

            <div className="flex justify-between items-center text-xs font-medium text-slate-500 uppercase tracking-widest px-2">
              <span>Defending Half</span>
              <span className="flex items-center gap-1">
                Attacking Direction
                <svg
                  className="w-3 h-3"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M14 5l7 7m0 0l-7 7m7-7H3"
                  />
                </svg>
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
