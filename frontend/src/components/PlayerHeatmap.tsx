import React, { useState, useEffect, useRef, useCallback } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

interface PlayerStats {
  goals: number;
  assists: number;
  passes_attempted: number;
  passes_completed: number;
  dribbles_attempted: number;
  dribbles_completed: number;
  shots: number;
}

interface PlayerData {
  id: number;
  name: string;
  short_name: string;
  jersey_number: number;
  is_starter: boolean;
  minutes_played: number;
  position: string;
  events_xy: number[][];
  stats: PlayerStats;
  avg_x: number;
  avg_y: number;
}

interface HeatmapResponse {
  match_id: number;
  team: string;
  starters: PlayerData[];
  bench: PlayerData[];
}

interface Competition {
  competition_id: number;
  season_id: number;
  display_name: string;
}

interface Match {
  match_id: number;
  home_team: string;
  away_team: string;
  display_name: string;
}

const PITCH_L = 120;
const PITCH_W = 80;

export default function PlayerHeatmap() {
  const [competitions, setCompetitions] = useState<Competition[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);
  const [selectedCompConfig, setSelectedCompConfig] = useState<string>("");
  const [selectedMatch, setSelectedMatch] = useState<number | "">("");
  const [selectedTeam, setSelectedTeam] = useState<string>("");

  const [heatmapData, setHeatmapData] = useState<HeatmapResponse | null>(null);
  const [selectedPlayer, setSelectedPlayer] = useState<PlayerData | null>(null);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isCompsLoading, setIsCompsLoading] = useState<boolean>(true);
  const [isMatchesLoading, setIsMatchesLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const renderCache = useRef<
    { id: number; cx: number; cy: number; r: number }[]
  >([]);

  useEffect(() => {
    const fetchComps = async () => {
      try {
        const res = await fetch(
          `${API_BASE_URL}/matches/statsbomb/competitions`,
        );
        if (!res.ok) throw new Error("Error cargando competiciones");
        const data = await res.json();
        setCompetitions(data);

        const wc2022 = data.find(
          (c: Competition) => c.competition_id === 43 && c.season_id === 106,
        );
        if (wc2022) setSelectedCompConfig("43-106");
      } catch (err: any) {
        setError("Falló la conexión con la API de competiciones.");
      } finally {
        setIsCompsLoading(false);
      }
    };
    fetchComps();
  }, []);

  useEffect(() => {
    if (!selectedCompConfig) return;
    const [compId, seasonId] = selectedCompConfig.split("-");
    const fetchMatches = async () => {
      setIsMatchesLoading(true);
      setSelectedMatch("");
      setSelectedTeam("");
      try {
        const res = await fetch(
          `${API_BASE_URL}/matches/statsbomb/matches/${compId}/${seasonId}`,
        );
        if (!res.ok) throw new Error("Error cargando partidos");
        const data = await res.json();
        setMatches(data);

        if (compId === "43" && seasonId === "106") {
          const finalMatch = data.find((m: Match) => m.match_id === 3869685);
          if (finalMatch) {
            setSelectedMatch(3869685);
            setSelectedTeam("Argentina");
          }
        }
      } catch (err: any) {
        setError("Falló la carga de partidos.");
      } finally {
        setIsMatchesLoading(false);
      }
    };

    fetchMatches();
  }, [selectedCompConfig]);

  const fetchHeatmapData = useCallback(async () => {
    if (!selectedMatch || !selectedTeam) {
      setError("Selecciona partido y equipo.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setSelectedPlayer(null);
    try {
      const response = await fetch(
        `${API_BASE_URL}/matches/${selectedMatch}/heatmaps/${selectedTeam}`,
      );
      if (!response.ok) throw new Error("Datos no encontrados.");
      const data: HeatmapResponse = await response.json();
      setHeatmapData(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, [selectedMatch, selectedTeam]);

  useEffect(() => {
    if (selectedMatch === 3869685 && selectedTeam === "Argentina") {
      fetchHeatmapData();
    }
  }, [selectedMatch, selectedTeam, fetchHeatmapData]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!heatmapData || !canvasRef.current || !containerRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    let clickedPlayerId: number | null = null;
    const hitTolerance = 15;

    for (let i = renderCache.current.length - 1; i >= 0; i--) {
      const p = renderCache.current[i];
      const dist = Math.sqrt(
        Math.pow(clickX - p.cx, 2) + Math.pow(clickY - p.cy, 2),
      );
      if (dist <= p.r + hitTolerance) {
        clickedPlayerId = p.id;
        break;
      }
    }

    if (clickedPlayerId) {
      const player =
        heatmapData.starters.find((p) => p.id === clickedPlayerId) ||
        heatmapData.bench.find((p) => p.id === clickedPlayerId);

      if (
        player &&
        (player.minutes_played > 0 || player.events_xy.length > 0)
      ) {
        setSelectedPlayer(player);
      }
    } else {
      setSelectedPlayer(null);
    }
  };

  const drawHeatmap = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
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

    ctx.strokeStyle = "rgba(255,255,255,0.25)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(ox, oy, pW, pH);
    ctx.beginPath();
    ctx.moveTo(ox + pW / 2, oy);
    ctx.lineTo(ox + pW / 2, oy + pH);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(ox + pW / 2, oy + pH / 2, pH * 0.15, 0, Math.PI * 2);
    ctx.stroke();

    const mapX = (x: number) => ox + (x / PITCH_L) * pW;
    const mapY = (y: number) => oy + (y / PITCH_W) * pH;

    if (!heatmapData) return;

    if (selectedPlayer && selectedPlayer.events_xy.length > 0) {
      const heatCanvas = document.createElement("canvas");
      heatCanvas.width = canvas.width;
      heatCanvas.height = canvas.height;
      const heatCtx = heatCanvas.getContext("2d", { willReadFrequently: true });

      if (heatCtx) {
        heatCtx.scale(dpr, dpr);

        const radius = 25;
        const blur = 25;

        heatCtx.shadowBlur = blur;
        heatCtx.shadowColor = "black";
        heatCtx.fillStyle = "black";
        heatCtx.globalAlpha = 0.5;

        selectedPlayer.events_xy.forEach(([ex, ey]) => {
          const nx = mapX(ex);
          const ny = mapY(ey);

          const radGrad = heatCtx.createRadialGradient(
            nx,
            ny,
            0,
            nx,
            ny,
            radius,
          );
          radGrad.addColorStop(0, "rgba(0,0,0,1)");
          radGrad.addColorStop(1, "rgba(0,0,0,0)");

          heatCtx.fillStyle = radGrad;
          heatCtx.beginPath();
          heatCtx.arc(nx, ny, radius, 0, Math.PI * 2);
          heatCtx.fill();
        });

        heatCtx.globalAlpha = 1.0;

        const imageData = heatCtx.getImageData(
          0,
          0,
          heatCanvas.width,
          heatCanvas.height,
        );
        const data = imageData.data;
        const colorGradient = createColorGradient();

        for (let i = 0; i < data.length; i += 4) {
          const alpha = data[i + 3];

          if (alpha > 0) {
            const offset = alpha * 4;

            data[i] = colorGradient[offset];
            data[i + 1] = colorGradient[offset + 1];
            data[i + 2] = colorGradient[offset + 2];
            data[i + 3] = Math.pow(alpha / 255, 1.2) * 255 * 0.8;
          }
        }

        heatCtx.putImageData(imageData, 0, 0);
        ctx.drawImage(heatCanvas, 0, 0, w, h);
      }
    }

    renderCache.current = [];

    const renderPlayerOnPitch = (
      player: PlayerData,
      isSelected: boolean,
      isSub: boolean = false,
    ) => {
      const nx = mapX(player.avg_x);
      const ny = mapY(player.avg_y);

      const radius = isSelected ? 16 : 14;

      renderCache.current.push({ id: player.id, cx: nx, cy: ny, r: radius });

      ctx.beginPath();
      ctx.arc(nx, ny, radius, 0, 2 * Math.PI);

      if (isSelected) {
        ctx.fillStyle = "#e11d48";
      } else {
        ctx.fillStyle = isSub ? "#475569" : "#1e293b";
      }

      ctx.fill();
      ctx.lineWidth = isSelected ? 2 : 1.5;
      ctx.strokeStyle = isSelected ? "#fff" : isSub ? "#94a3b8" : "#cbd5e1";
      ctx.stroke();

      ctx.globalAlpha = 1.0;
      ctx.globalCompositeOperation = "source-over";
      ctx.shadowBlur = 0;

      ctx.font = "bold 13px system-ui, -apple-system, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = isSelected ? "#ffffff" : "#f1f5f9";

      ctx.fillText(
        player.jersey_number ? player.jersey_number.toString() : "-",
        nx,
        ny + 1,
      );

      const displayName = player.short_name;

      ctx.font = isSelected
        ? "bold 11px system-ui, -apple-system, sans-serif"
        : "600 10px system-ui, -apple-system, sans-serif";

      const textWidth = ctx.measureText(displayName).width;
      const badgeY = ny + radius + 8;

      ctx.fillStyle = isSelected
        ? "rgba(225, 29, 72, 0.95)"
        : "rgba(15, 23, 42, 0.85)";

      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(nx - textWidth / 2 - 4, badgeY - 7, textWidth + 8, 14, 4);
      } else {
        ctx.fillRect(nx - textWidth / 2 - 4, badgeY - 7, textWidth + 8, 14);
      }
      ctx.fill();

      ctx.fillStyle = isSelected ? "#ffffff" : "#cbd5e1";
      ctx.fillText(displayName, nx, badgeY);
    };

    heatmapData.starters.forEach((player) => {
      if (selectedPlayer?.id !== player.id) {
        renderPlayerOnPitch(player, false, false);
      }
    });

    if (selectedPlayer && selectedPlayer.is_starter) {
      renderPlayerOnPitch(selectedPlayer, true, false);
    }

    if (
      selectedPlayer &&
      !selectedPlayer.is_starter &&
      selectedPlayer.avg_x !== 10
    ) {
      renderPlayerOnPitch(selectedPlayer, true, true);
    }
  }, [heatmapData, selectedPlayer]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    drawHeatmap();

    const resizeObserver = new ResizeObserver(() => {
      drawHeatmap();
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
    };
  }, [drawHeatmap]);

  const currentMatchObj = matches.find(
    (m) => m.match_id === Number(selectedMatch),
  );

  const sortedBench = heatmapData
    ? [...heatmapData.bench].sort((a, b) => {
        const aPlayed = a.minutes_played > 0 || a.events_xy.length > 0;
        const bPlayed = b.minutes_played > 0 || b.events_xy.length > 0;

        if (aPlayed && !bPlayed) return -1;
        if (!aPlayed && bPlayed) return 1;
        return b.minutes_played - a.minutes_played;
      })
    : [];

  return (
    <div className="flex flex-col flex-grow w-full p-4 sm:p-8">
      <div className="max-w-7xl mx-auto w-full flex-grow flex flex-col">
        <header className="mb-6 border-b border-slate-800 pb-6">
          <span className="inline-flex items-center gap-1.5 text-xs font-medium text-rose-400 mb-2">
            <span className="h-1.5 w-1.5 rounded-full bg-rose-400" />
            Player heatmaps
          </span>
          <h1 className="font-display text-2xl font-semibold tracking-tight text-slate-50">
            Individual spatial analysis
          </h1>
          <p className="text-sm text-slate-400 mt-2 max-w-xl">
            Touch locations and per-player statistics for any starter or
            substitute in the match sheet.
          </p>
        </header>

        <div className="flex flex-col md:flex-row gap-4 mb-6 bg-slate-900/60 p-4 rounded-xl border border-slate-800">
          <div className="flex flex-col gap-1 flex-1">
            <label className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
              1. Competition
            </label>
            <select
              value={selectedCompConfig}
              onChange={(e) => setSelectedCompConfig(e.target.value)}
              disabled={isCompsLoading}
              className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-rose-400/60 disabled:opacity-50"
            >
              <option value="">Select a competition...</option>
              {competitions.map((c) => (
                <option
                  key={`${c.competition_id}-${c.season_id}`}
                  value={`${c.competition_id}-${c.season_id}`}
                >
                  {c.display_name}
                </option>
              ))}
            </select>
          </div>
          <div className="flex flex-col gap-1 flex-1">
            <label className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
              2. Match
            </label>
            <select
              value={selectedMatch}
              onChange={(e) =>
                setSelectedMatch(e.target.value ? Number(e.target.value) : "")
              }
              disabled={!selectedCompConfig || isMatchesLoading}
              className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-rose-400/60 disabled:opacity-50"
            >
              <option value="">
                {isMatchesLoading ? "Loading matches..." : "Select a match..."}
              </option>
              {matches.map((m) => (
                <option key={m.match_id} value={m.match_id}>
                  {m.display_name}
                </option>
              ))}
            </select>
          </div>
          <div className="flex flex-col gap-1 flex-1">
            <label className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
              3. Team
            </label>
            <select
              value={selectedTeam}
              onChange={(e) => setSelectedTeam(e.target.value)}
              disabled={!selectedMatch}
              className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-rose-400/60 disabled:opacity-50"
            >
              <option value="">Select team...</option>
              {currentMatchObj && (
                <>
                  <option value={currentMatchObj.home_team}>
                    {currentMatchObj.home_team}
                  </option>
                  <option value={currentMatchObj.away_team}>
                    {currentMatchObj.away_team}
                  </option>
                </>
              )}
            </select>
          </div>
          <div className="flex flex-col gap-1 justify-end">
            <button
              onClick={fetchHeatmapData}
              disabled={isLoading || !selectedMatch || !selectedTeam}
              className="bg-amber-400 hover:bg-amber-300 text-slate-950 font-semibold px-6 py-2 rounded-lg transition-colors disabled:opacity-40 h-[38px]"
            >
              {isLoading ? "Loading…" : "Generate"}
            </button>
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-900/30 border border-red-500/50 text-red-400 rounded-lg">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 flex-grow">
          <div className="lg:col-span-3 flex flex-col gap-4">
            <div
              ref={containerRef}
              className="w-full bg-slate-950 rounded-2xl border border-slate-700/80 shadow-2xl overflow-hidden relative min-h-[450px]"
              style={{ boxShadow: "inset 0 0 40px rgba(0,0,0,0.5)" }}
            >
              <canvas
                ref={canvasRef}
                onClick={handleCanvasClick}
                className="absolute inset-0 w-full h-full block cursor-pointer"
              />

              {!selectedPlayer && heatmapData && (
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none bg-slate-950/40">
                  <span className="bg-slate-900/90 text-slate-300 px-4 py-2 rounded-full border border-slate-700 backdrop-blur-sm shadow-xl font-medium text-sm flex items-center gap-2">
                    <svg
                      className="w-4 h-4 text-sky-400"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122"
                      />
                    </svg>
                    Select a player from the list or click on the pitch to view
                    heatmap
                  </span>
                </div>
              )}
            </div>

            {selectedPlayer && (
              <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5 flex flex-wrap gap-6 items-center shadow-lg animate-in fade-in slide-in-from-bottom-4 duration-300">
                <div className="flex-1 min-w-[200px]">
                  <h3 className="text-xl font-bold text-white flex items-center gap-3">
                    <span className="bg-rose-500/20 text-rose-400 px-2 py-0.5 rounded text-sm">
                      {selectedPlayer.jersey_number
                        ? selectedPlayer.jersey_number
                        : "?"}
                    </span>
                    {selectedPlayer.name}
                  </h3>
                  <p className="text-slate-400 text-sm mt-1">
                    {selectedPlayer.position} •{" "}
                    {selectedPlayer.minutes_played > 0
                      ? `${selectedPlayer.minutes_played}' played`
                      : "Sub (No recorded minutes)"}
                  </p>
                </div>

                <div className="flex gap-6 divide-x divide-slate-700">
                  <div className="flex flex-col pl-6 first:pl-0">
                    <span className="text-xs text-slate-400 uppercase tracking-wider">
                      Goals/Assists
                    </span>
                    <span className="text-2xl font-bold text-emerald-400">
                      {selectedPlayer.stats.goals} /{" "}
                      {selectedPlayer.stats.assists}
                    </span>
                  </div>
                  <div className="flex flex-col pl-6">
                    <span className="text-xs text-slate-400 uppercase tracking-wider">
                      Passes
                    </span>
                    <span className="text-2xl font-bold text-sky-400">
                      {selectedPlayer.stats.passes_completed}/
                      {selectedPlayer.stats.passes_attempted}
                      <span className="text-sm font-normal text-slate-500 ml-1">
                        (
                        {selectedPlayer.stats.passes_attempted > 0
                          ? Math.round(
                              (selectedPlayer.stats.passes_completed /
                                selectedPlayer.stats.passes_attempted) *
                                100,
                            )
                          : 0}
                        %)
                      </span>
                    </span>
                  </div>
                  <div className="flex flex-col pl-6">
                    <span className="text-xs text-slate-400 uppercase tracking-wider">
                      Dribbles
                    </span>
                    <span className="text-2xl font-bold text-amber-400">
                      {selectedPlayer.stats.dribbles_completed}/
                      {selectedPlayer.stats.dribbles_attempted}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="lg:col-span-1 flex flex-col gap-4 overflow-y-auto pr-2 max-h-[600px] scrollbar-thin scrollbar-thumb-slate-700">
            {heatmapData && (
              <>
                <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 border-b border-slate-800 pb-2">
                  Starting XI
                </h3>
                <div className="flex flex-col gap-2">
                  {heatmapData.starters.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => setSelectedPlayer(p)}
                      className={`text-left px-4 py-3 rounded-lg border transition-all flex items-center justify-between group
                                    ${
                                      selectedPlayer?.id === p.id
                                        ? "bg-rose-500/20 border-rose-500/50 text-white"
                                        : "bg-slate-900 border-slate-800 text-slate-300 hover:bg-slate-800 hover:border-slate-600"
                                    }`}
                    >
                      <span className="font-medium truncate pr-2">
                        {p.short_name}
                      </span>
                      <span
                        className={`text-xs font-mono px-2 py-1 rounded bg-slate-950 ${selectedPlayer?.id === p.id ? "text-rose-400" : "text-slate-500 group-hover:text-slate-400"}`}
                      >
                        {p.jersey_number}
                      </span>
                    </button>
                  ))}
                </div>

                <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 border-b border-slate-800 pb-2 mt-4">
                  Bench
                </h3>
                <div className="flex flex-col gap-2">
                  {sortedBench.map((p) => {
                    const played =
                      p.minutes_played > 0 || p.events_xy.length > 0;

                    return (
                      <button
                        key={p.id}
                        onClick={() => setSelectedPlayer(p)}
                        disabled={!played}
                        className={`text-left px-4 py-2 rounded-lg border transition-all flex flex-col
                                        ${
                                          !played
                                            ? "opacity-40 cursor-not-allowed bg-slate-900/50 border-slate-800/50"
                                            : selectedPlayer?.id === p.id
                                              ? "bg-rose-500/20 border-rose-500/50 text-white"
                                              : "bg-slate-900 border-slate-800 text-slate-300 hover:bg-slate-800"
                                        }`}
                      >
                        <div className="flex items-center justify-between w-full">
                          <span className="font-medium truncate pr-2 text-sm">
                            {p.short_name}
                          </span>
                          {played && (
                            <span className="text-[10px] text-emerald-500">
                              {p.minutes_played > 0
                                ? `${p.minutes_played}'`
                                : "Sub"}
                            </span>
                          )}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function createColorGradient() {
  const canvas = document.createElement("canvas");
  canvas.width = 1;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");
  if (!ctx) return new Uint8ClampedArray(256 * 4);

  const grad = ctx.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0, "rgba(0, 0, 255, 0)");
  grad.addColorStop(0.3, "rgba(0, 255, 255, 0.4)");
  grad.addColorStop(0.5, "rgba(0, 255, 0, 0.6)");
  grad.addColorStop(0.8, "rgba(255, 255, 0, 0.9)");
  grad.addColorStop(1, "rgba(255, 0, 0, 1)");

  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 1, 256);

  return ctx.getImageData(0, 0, 1, 256).data;
}
