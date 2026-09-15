import { useState, useEffect, useRef, useCallback } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

interface Node {
  id: string;
  label: string;
  x: number;
  y: number;
  value: number;
}

interface Link {
  source: string;
  target: string;
  value: number;
}

interface NetworkData {
  match_id: number;
  team: string;
  nodes: Node[];
  links: Link[];
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

export default function PassNetwork() {
  const [competitions, setCompetitions] = useState<Competition[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);

  const [selectedCompConfig, setSelectedCompConfig] = useState<string>("");
  const [selectedMatch, setSelectedMatch] = useState<number | "">("");
  const [selectedTeam, setSelectedTeam] = useState<string>("");

  const [networkData, setNetworkData] = useState<NetworkData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isCompsLoading, setIsCompsLoading] = useState<boolean>(true);
  const [isMatchesLoading, setIsMatchesLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

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
        if (wc2022) {
          setSelectedCompConfig("43-106");
        }
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

  const fetchNetwork = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `${API_BASE_URL}/matches/${selectedMatch}/pass-network/${selectedTeam}`,
      );
      if (!response.ok) {
        throw new Error(`Error ${response.status}: Datos no encontrados.`);
      }
      const data: NetworkData = await response.json();
      setNetworkData(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message);
      setNetworkData(null);
    } finally {
      setIsLoading(false);
    }
  }, [selectedMatch, selectedTeam]);

  useEffect(() => {
    if (selectedMatch === 3869685 && selectedTeam === "Argentina") {
      fetchNetwork();
    }
  }, [selectedMatch, selectedTeam, fetchNetwork]);

  const drawNetwork = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
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

    ctx.beginPath();
    ctx.arc(ox + pW / 2, oy + pH / 2, 2, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,255,255,0.6)";
    ctx.fill();

    if (!networkData) return;

    const mapX = (x: number) => ox + (x / PITCH_L) * pW;
    const mapY = (y: number) => oy + (y / PITCH_W) * pH;

    const maxLinkValue = Math.max(...networkData.links.map((l) => l.value), 1);

    networkData.links.forEach((link) => {
      const sourceNode = networkData.nodes.find((n) => n.id === link.source);
      const targetNode = networkData.nodes.find((n) => n.id === link.target);

      if (sourceNode && targetNode) {
        const x1 = mapX(sourceNode.x);
        const y1 = mapY(sourceNode.y);
        const x2 = mapX(targetNode.x);
        const y2 = mapY(targetNode.y);

        const linkThickness = (link.value / maxLinkValue) * 6 + 1;

        const alpha = Math.max(0.1, (link.value / maxLinkValue) * 0.8);

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.lineWidth = linkThickness;
        ctx.strokeStyle = `rgba(56, 189, 248, ${alpha})`;
        ctx.stroke();
      }
    });

    const maxNodeValue = Math.max(...networkData.nodes.map((n) => n.value), 1);

    networkData.nodes.forEach((node) => {
      const nx = mapX(node.x);
      const ny = mapY(node.y);

      const radius = (node.value / maxNodeValue) * 15 + 8;

      ctx.shadowColor = `rgba(0,0,0,0.5)`;
      ctx.shadowBlur = 4;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 2;

      ctx.beginPath();
      ctx.arc(nx, ny, radius, 0, 2 * Math.PI);
      ctx.fillStyle = "#e2e8f0";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#38bdf8";
      ctx.stroke();

      ctx.shadowBlur = 0;
      ctx.shadowColor = "transparent";

      ctx.font = "bold 11px system-ui, -apple-system, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      const textWidth = ctx.measureText(node.label).width;
      ctx.fillStyle = "rgba(15,23,42,0.7)";
      ctx.roundRect(
        nx - textWidth / 2 - 4,
        ny + radius + 2,
        textWidth + 8,
        16,
        4,
      );
      ctx.fill();

      ctx.fillStyle = "#f8fafc";
      ctx.fillText(node.label, nx, ny + radius + 10);
    });

    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.textAlign = "center";
    ctx.font = "12px sans-serif";
    ctx.fillText("Attacking Direction ➔", w / 2, h - 10);
  }, [networkData]);

  useEffect(() => {
    drawNetwork();
    const handleResize = () => drawNetwork();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [drawNetwork]);

  const currentMatchObj = matches.find(
    (m) => m.match_id === Number(selectedMatch),
  );

  return (
    <div className="flex-grow w-full p-4 sm:p-8 flex flex-col">
      <div className="max-w-6xl mx-auto w-full flex-grow flex flex-col">
        <header className="mb-6 border-b border-slate-800 pb-6">
          <span className="inline-flex items-center gap-1.5 text-xs font-medium text-teal-400 mb-2">
            <span className="h-1.5 w-1.5 rounded-full bg-teal-400" />
            Pass networks
          </span>
          <h1 className="font-display text-2xl font-semibold tracking-tight text-slate-50">
            Passing geometry from StatsBomb event data
          </h1>
          <p className="text-sm text-slate-400 mt-2 max-w-xl">
            Player positioning is the average location of their touches;
            connections are weighted by how often two players combined.
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
              className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-amber-400/60 disabled:opacity-50"
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
              className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-amber-400/60 disabled:opacity-50"
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
              className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-amber-400/60 disabled:opacity-50"
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
              onClick={fetchNetwork}
              disabled={isLoading}
              className="bg-amber-400 hover:bg-amber-300 text-slate-950 font-semibold px-6 py-2 rounded-lg transition-colors disabled:opacity-40 h-[38px]"
            >
              {isLoading ? "Loading…" : "Analyze"}
            </button>
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-rose-900/30 border border-rose-500/50 text-rose-400 rounded-lg">
            {error}
          </div>
        )}

        <div
          ref={containerRef}
          className="flex-grow w-full bg-slate-950 rounded-2xl border border-slate-700/80 shadow-2xl overflow-hidden relative min-h-[500px]"
          style={{ boxShadow: "inset 0 0 40px rgba(0,0,0,0.5)" }}
        >
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full block"
          />
        </div>
      </div>
    </div>
  );
}
