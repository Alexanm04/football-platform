import React, { useState, useRef } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";
const API_BASE = `${API_BASE_URL}/vision`;

export default function ComputerVision() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [resultVideoUrl, setResultVideoUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.type.startsWith("video/")) {
        setError("Please upload a valid video file (.mp4, .avi, .mov)");
        return;
      }
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));

      // Limpiamos la URL anterior de memoria si existía
      if (resultVideoUrl) {
        URL.revokeObjectURL(resultVideoUrl);
      }

      setResultVideoUrl(null);
      setError(null);
      setProgress(0);
    }
  };

  const clearSelection = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (resultVideoUrl) URL.revokeObjectURL(resultVideoUrl);
    setPreviewUrl(null);
    setSelectedFile(null);
    setResultVideoUrl(null);
    setError(null);
    setProgress(0);
  };

  const handleUpload = () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setError(null);
    setResultVideoUrl(null);
    setProgress(0);

    const formData = new FormData();
    formData.append("file", selectedFile);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/process-video`, true);

    // 1. Escuchamos el progreso REAL de subida del vídeo al servidor
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        const percentComplete = Math.round((event.loaded / event.total) * 100);
        // Vamos a usar del 0% al 90% para la subida y espera.
        // Como no podemos saber el progreso interno de la IA desde aquí,
        // al menos veremos la barra llenarse mientras sube el archivo.
        setProgress(Math.min(percentComplete * 0.9, 90));
      }
    };

    xhr.responseType = "blob";

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        const blob = xhr.response;
        const videoUrl = URL.createObjectURL(blob);
        setProgress(100);
        setResultVideoUrl(videoUrl);
        setIsProcessing(false);
      } else {
        // Si hay error (500, 400), intentamos leer el JSON que devuelve FastAPI
        const reader = new FileReader();
        reader.onload = () => {
          try {
            const errorData = JSON.parse(reader.result as string);
            setError(errorData.detail || "Error interno en el servidor");
          } catch {
            setError(`Error del servidor: código ${xhr.status}`);
          }
          setIsProcessing(false);
          setProgress(0);
        };
        reader.readAsText(xhr.response);
      }
    };

    xhr.onerror = () => {
      // Si entra aquí, casi al 100% es un problema de CORS o el backend se ha apagado.
      setError(
        "Error de Red. Revisa la consola del navegador (F12) y el CORS en FastAPI.",
      );
      setIsProcessing(false);
      setProgress(0);
    };

    xhr.send(formData);
  };

  return (
    <div className="flex flex-col items-center flex-grow w-full p-4 sm:p-8 overflow-y-auto">
      <div className="max-w-4xl w-full flex flex-col gap-6">
        <header className="mb-2 border-b border-slate-800 pb-6">
          <span className="inline-flex items-center gap-1.5 text-xs font-medium text-indigo-400 mb-2">
            <span className="h-1.5 w-1.5 rounded-full bg-indigo-400" />
            Computer vision
          </span>
          <h1 className="font-display text-2xl font-semibold tracking-tight text-slate-50 mb-2">
            Track players and the ball from raw video
          </h1>
          <p className="text-slate-400 text-sm max-w-2xl">
            Upload a football clip. The pipeline tracks players and the ball,
            classifies teams by jersey appearance, and produces tactical
            analytics from the footage.
          </p>
        </header>

        {error && (
          <div className="p-4 bg-red-900/30 border border-red-500/50 text-red-400 rounded-xl text-center">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="flex flex-col gap-4 bg-slate-900/60 p-6 rounded-2xl border border-slate-800">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <span className="bg-indigo-400/10 text-indigo-400 p-2 rounded-lg">
                <svg
                  className="w-5 h-5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                  />
                </svg>
              </span>
              1. Upload Original Clip
            </h2>

            {!previewUrl ? (
              <div
                className="border-2 border-dashed border-slate-700 rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer hover:border-indigo-400/60 hover:bg-slate-900 transition-colors h-64"
                onClick={() => fileInputRef.current?.click()}
              >
                <svg
                  className="w-12 h-12 text-slate-500 mb-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                  />
                </svg>
                <p className="text-slate-300 font-medium mb-1">
                  Click to select video
                </p>
                <p className="text-slate-500 text-sm">MP4, AVI, or MOV</p>
              </div>
            ) : (
              <div className="relative rounded-xl overflow-hidden bg-black border border-slate-700 aspect-video flex items-center justify-center">
                <video
                  src={previewUrl}
                  controls
                  className="max-h-full max-w-full"
                />
                <button
                  onClick={clearSelection}
                  className="absolute top-2 right-2 bg-slate-900/80 hover:bg-red-500/80 text-white p-2 rounded-lg transition-colors backdrop-blur"
                  title="Remove video"
                >
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>
            )}

            <input
              type="file"
              accept="video/*"
              className="hidden"
              ref={fileInputRef}
              onChange={handleFileChange}
            />

            <button
              onClick={handleUpload}
              disabled={!selectedFile || isProcessing}
              className={`mt-auto py-3 rounded-xl font-semibold transition-colors flex justify-center items-center gap-2
                    ${
                      !selectedFile
                        ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                        : isProcessing
                          ? "bg-indigo-500/50 text-white cursor-wait"
                          : "bg-indigo-500 hover:bg-indigo-400 text-white"
                    }`}
            >
              {isProcessing ? (
                <>
                  <svg
                    className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  Processing with AI...
                </>
              ) : (
                "Analyze Video"
              )}
            </button>
          </div>

          <div className="flex flex-col gap-4 bg-slate-900/60 p-6 rounded-2xl border border-slate-800">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <span className="bg-indigo-400/10 text-indigo-400 p-2 rounded-lg">
                <svg
                  className="w-5 h-5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                  />
                </svg>
              </span>
              2. Tracked Result
            </h2>

            <div className="relative rounded-xl overflow-hidden bg-slate-900 border border-slate-700 aspect-video flex items-center justify-center h-64 md:h-auto">
              {isProcessing ? (
                <div className="flex flex-col items-center text-slate-400 w-full px-8">
                  <svg
                    className="w-12 h-12 mb-4 text-indigo-400 animate-pulse"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1}
                      d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"
                    />
                  </svg>
                  <p className="mb-3">Analyzing frames...</p>
                  <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className="h-full bg-indigo-400 transition-all duration-500 ease-out"
                      style={{ width: `${Math.min(progress, 100)}%` }}
                    />
                  </div>
                  <p className="text-xs mt-2 text-slate-500 tabular-nums">
                    {progress.toFixed(0)}%
                  </p>
                </div>
              ) : resultVideoUrl ? (
                <video
                  src={resultVideoUrl}
                  controls
                  autoPlay
                  className="max-h-full max-w-full"
                />
              ) : (
                <div className="text-slate-600 flex flex-col items-center">
                  <svg
                    className="w-12 h-12 mb-2 opacity-50"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1}
                      d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                    />
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1}
                      d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                    />
                  </svg>
                  <p>Awaiting video upload</p>
                </div>
              )}
            </div>

            {resultVideoUrl && (
              <a
                href={resultVideoUrl}
                download={"tracked_football_clip.mp4"}
                className="mt-auto w-full py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-xl font-bold transition-all text-center border border-slate-600"
              >
                Download Processed Video
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
