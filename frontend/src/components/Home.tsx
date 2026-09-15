import type { JSX } from "react/jsx-runtime";

import type { Screen } from "../App";

interface HomeProps {
  onNavigate: (screen: Screen) => void;
}

function XgVisual() {
  return (
    <div className="flex items-end gap-2">
      <div className="w-4 h-12 rounded-sm bg-slate-800 relative overflow-hidden">
        <div className="absolute bottom-0 left-0 right-0 h-[68%] bg-amber-400/70" />
      </div>
      <div className="w-4 h-12 rounded-sm bg-slate-800 relative overflow-hidden">
        <div className="absolute bottom-0 left-0 right-0 h-[34%] bg-amber-400/30" />
      </div>
      <div className="w-4 h-12 rounded-sm bg-slate-800 relative overflow-hidden">
        <div className="absolute bottom-0 left-0 right-0 h-[50%] bg-amber-400/50" />
      </div>
    </div>
  );
}

function NetworkVisual() {
  return (
    <svg viewBox="0 0 64 48" className="w-16 h-12">
      <line
        x1="10"
        y1="12"
        x2="32"
        y2="30"
        stroke="currentColor"
        strokeOpacity="0.5"
        strokeWidth="1.5"
      />
      <line
        x1="32"
        y1="30"
        x2="54"
        y2="10"
        stroke="currentColor"
        strokeOpacity="0.35"
        strokeWidth="1.5"
      />
      <line
        x1="10"
        y1="12"
        x2="54"
        y2="10"
        stroke="currentColor"
        strokeOpacity="0.2"
        strokeWidth="1.5"
      />
      <line
        x1="32"
        y1="30"
        x2="20"
        y2="40"
        stroke="currentColor"
        strokeOpacity="0.35"
        strokeWidth="1.5"
      />
      <circle cx="10" cy="12" r="4" fill="currentColor" fillOpacity="0.85" />
      <circle cx="54" cy="10" r="3" fill="currentColor" fillOpacity="0.6" />
      <circle cx="32" cy="30" r="5.5" fill="currentColor" />
      <circle cx="20" cy="40" r="3" fill="currentColor" fillOpacity="0.6" />
    </svg>
  );
}

function HeatmapVisual() {
  const dots = [
    { x: 30, y: 22, r: 11, o: 0.5 },
    { x: 22, y: 30, r: 7, o: 0.4 },
    { x: 40, y: 16, r: 8, o: 0.35 },
    { x: 34, y: 30, r: 5, o: 0.6 },
  ];
  return (
    <svg viewBox="0 0 64 48" className="w-16 h-12">
      {dots.map((d, i) => (
        <circle
          key={i}
          cx={d.x}
          cy={d.y}
          r={d.r}
          fill="currentColor"
          fillOpacity={d.o}
        />
      ))}
    </svg>
  );
}

function VisionVisual() {
  return (
    <svg viewBox="0 0 64 48" className="w-16 h-12">
      <rect
        x="1"
        y="1"
        width="62"
        height="46"
        rx="3"
        stroke="currentColor"
        strokeOpacity="0.25"
      />
      <rect
        x="14"
        y="10"
        width="16"
        height="24"
        rx="1.5"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      <rect
        x="36"
        y="18"
        width="14"
        height="20"
        rx="1.5"
        stroke="currentColor"
        strokeOpacity="0.5"
        strokeWidth="1.5"
      />
    </svg>
  );
}

const tools: {
  screen: Screen;
  title: string;
  description: string;
  accentText: string;
  accentBorder: string;
  accentBg: string;
  visual: JSX.Element;
  icon: JSX.Element;
}[] = [
  {
    screen: "simulator",
    title: "Tactical ML Simulator",
    description:
      "Adjust defensive height, pressure intensity, and width to predict Expected Goals with a trained XGBoost model, and see the SHAP breakdown behind every prediction.",
    accentText: "text-amber-400",
    accentBorder: "hover:border-amber-400/40",
    accentBg: "bg-amber-400/10",
    visual: <XgVisual />,
    icon: (
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.75}
        d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
      />
    ),
  },
  {
    screen: "passNetwork",
    title: "Pass Networks",
    description:
      "Geometric passing structure from real StatsBomb match data — player positioning and the strongest connections on the pitch.",
    accentText: "text-teal-400",
    accentBorder: "hover:border-teal-400/40",
    accentBg: "bg-teal-400/10",
    visual: <NetworkVisual />,
    icon: (
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.75}
        d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
      />
    ),
  },
  {
    screen: "heatmaps",
    title: "Player Heatmaps",
    description:
      "Touch heatmaps and match statistics for any starter or substitute — click a player or a spot on the pitch.",
    accentText: "text-rose-400",
    accentBorder: "hover:border-rose-400/40",
    accentBg: "bg-rose-400/10",
    visual: <HeatmapVisual />,
    icon: (
      <>
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.75}
          d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
        />
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.75}
          d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
        />
      </>
    ),
  },
  {
    screen: "vision",
    title: "Computer Vision",
    description:
      "Upload a clip and let YOLOv8 with K-means clustering track players automatically, grouped by jersey color.",
    accentText: "text-indigo-400",
    accentBorder: "hover:border-indigo-400/40",
    accentBg: "bg-indigo-400/10",
    visual: <VisionVisual />,
    icon: (
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.75}
        d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
      />
    ),
  },
];

export default function Home({ onNavigate }: HomeProps) {
  return (
    <div className="flex flex-col items-center flex-grow w-full px-4 sm:px-8 py-14 sm:py-20">
      <div className="max-w-4xl w-full">
        <div className="relative mb-16">
          <svg
            className="absolute -top-6 right-0 w-72 h-72 text-slate-800/60 pointer-events-none hidden lg:block"
            viewBox="0 0 200 200"
            fill="none"
          >
            <rect
              x="0.5"
              y="0.5"
              width="199"
              height="199"
              rx="4"
              stroke="currentColor"
            />
            <circle cx="100" cy="100" r="36" stroke="currentColor" />
            <circle cx="100" cy="100" r="1.5" fill="currentColor" />
            <line x1="100" y1="0" x2="100" y2="200" stroke="currentColor" />
          </svg>

          <h1 className="font-display text-4xl sm:text-5xl font-semibold tracking-tight text-slate-50 mb-5 leading-[1.08] max-w-2xl">
            Football analytics, from raw event data to tactical decisions.
          </h1>
          <p className="text-base text-slate-400 leading-relaxed max-w-lg mb-8">
            Four tools built on one match dataset: a trained model for outcome
            prediction, a geometric read of passing structure, a spatial view of
            individual players, and a vision pipeline that tracks a match
            straight from video.
          </p>

          <dl className="flex flex-wrap gap-x-10 gap-y-4 border-t border-slate-800 pt-6 max-w-lg">
            <div>
              <dt className="text-[11px] uppercase tracking-wide text-slate-500">
                Data source
              </dt>
              <dd className="font-mono text-xl text-slate-100 mt-0.5">
                StatsBomb
              </dd>
            </div>
            <div>
              <dt className="text-[11px] uppercase tracking-wide text-slate-500">
                Models
              </dt>
              <dd className="font-mono text-xl text-slate-100 mt-0.5">
                XGBoost · YOLOv8
              </dd>
            </div>
          </dl>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {tools.map((tool) => (
            <button
              key={tool.screen}
              onClick={() => onNavigate(tool.screen)}
              className={`group text-left bg-slate-900/60 border border-slate-800 ${tool.accentBorder} rounded-2xl p-6 transition-colors duration-200 flex flex-col h-full`}
            >
              <div className="flex items-start justify-between gap-4 mb-5">
                <div
                  className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 ${tool.accentBg} ${tool.accentText}`}
                >
                  <svg
                    className="w-4.5 h-4.5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    {tool.icon}
                  </svg>
                </div>
                <div className={`${tool.accentText} opacity-80`}>
                  {tool.visual}
                </div>
              </div>
              <h3 className="font-display text-lg font-semibold text-slate-100 mb-2">
                {tool.title}
              </h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                {tool.description}
              </p>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
