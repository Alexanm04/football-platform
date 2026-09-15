from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: int = Field(..., description="Coordenada X de la esquina superior izquierda")
    y1: int = Field(..., description="Coordenada Y de la esquina superior izquierda")
    x2: int = Field(..., description="Coordenada X de la esquina inferior derecha")
    y2: int = Field(..., description="Coordeneada Y de la esquina inferior derecha")

class Detection(BaseModel):
    track_id: int = Field(..., description="ID único del objeto trackeado a lo largo del vídeo")
    class_id: int = Field(..., description="ID de la clase YOLO (ej: 0=Persona, 32=Balón)")
    class_name: str = Field(..., description="Nombre de la clase (ej: 'Player', 'Ball)")
    box: BoundingBox = Field(..., description="Coordenadas de la caja delimitadora")
    team_id: int | None = Field(None, description="ID del equipo asignado por K-Means (0, 1 o 2 para árbitros)")
    team_color: tuple[int, int, int] | None = Field(None, description="Color RGB asignado al equipo para visualización")

class FrameData(BaseModel):
    frame_index: int = Field(..., description="Número del fotograma en el vídeo")
    timestamp_sec: float = Field(..., description="Marca de tiempo en segundos")
    detections: list[Detection] = Field(default_factory=list, description="Lista de objetos detectados en este frame")

class VisionProcessResponse(BaseModel):
    success: bool
    message: str
    video_url: str | None = Field(None, description="Ruta o URL para descargar el vídeo renderizado")
    total_frames_processed: int
    teams_detected: int = Field(default=3, description="Número de clusters encontrados (equipos + árbitros)")