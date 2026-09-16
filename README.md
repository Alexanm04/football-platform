# ⚽ Football Platform

## ¿Qué es Football Platform?

Football Platform es un proyecto que reúne análisis de partidos,
visualización de datos, un simulador táctico y procesamiento de vídeo de fútbol
en una misma aplicación.

He construido una arquitectura con un backend FastAPI, un frontend React,
persistencia PostgreSQL, integración con StatsBomb, inferencia con modelos
XGBoost y un pipeline de Computer Vision basado en YOLO, BoT-SORT y OpenCV.

El objetivo del proyecto es combinar estas tecnologías para ofrecer diferentes herramientas de análisis de fútbol, desde el estudio de datos de partidos y aspectos tácticos hasta el análisis de vídeo mediante Computer Vision.

El repositorio incluye el código de la aplicación, los scripts necesarios para su funcionamiento y los modelos utilizados en los distintos módulos.

## 🚀 Funcionalidades

### 📊 Análisis de partidos

El backend integra `statsbombpy` para consultar competiciones, temporadas,
partidos y eventos de StatsBomb. También permite crear partidos y almacenar
eventos JSON validados en PostgreSQL.

Desde la interfaz se pueden seleccionar competiciones y partidos para generar
redes de pases y mapas de calor.

### 🔗 Red de pases

El endpoint:

```text
GET /api/v1/matches/{match_id}/pass-network/{team_name}
```

obtiene eventos de StatsBomb, filtra los pases completados del equipo elegido
y excluye los eventos posteriores a la primera sustitución. Calcula posiciones
medias y número de pases por jugador, y conserva únicamente las conexiones con
al menos tres pases.

La respuesta contiene nodos (`id`, etiqueta, posición y volumen de pases) y
enlaces (`source`, `target` y número de pases). El frontend los representa
sobre un campo dibujado en un `canvas`.

### 🗺️ Mapas de calor

El endpoint:

```text
GET /api/v1/matches/{match_id}/heatmaps/{team_name}
```

devuelve jugadores titulares y suplentes, eventos espaciales, minutos,
estadísticas y posiciones medias. El frontend dibuja las posiciones y la
distribución espacial sobre un campo, y permite seleccionar jugadores con
actividad o eventos disponibles.

No es un sistema de tracking de vídeo: estos mapas proceden de eventos de
StatsBomb.

### 🤖 Simulador táctico con Machine Learning

El endpoint:

```text
POST /api/v1/simulator/predict
```

recibe:

- `defensiveHeight`: altura defensiva, entre 0 y 105;
- `pressureIntensity`: intensidad de presión, usada como PPDA;
- `possession`: posesión, entre 0 y 100;
- `width`: amplitud del equipo, entre 0 y 68.

Utiliza los modelos XGBoost previamente entrenados y guardados en `tactical_models.pkl` para devolver
`xg_for` y `xg_against`. También devuelve valores SHAP para explicar el
impacto de las cuatro variables en cada predicción.

El frontend ofrece sliders para modificar estos valores y actualiza la
predicción con un pequeño debounce. El entrenamiento se realiza fuera del
endpoint mediante `train_tactical_model.py` y el endpoint únicamente carga e
infiere con el bundle ya generado.

### 🎥 Análisis de vídeo con Computer Vision

El endpoint:

```text
POST /api/v1/vision/process-video
```

recibe un vídeo MP4, AVI o MOV, valida su contenido y devuelve un MP4
renderizado con las personas detectadas, sus IDs y etiquetas de equipo o rol
cuando pueden determinarse.

### 👥 Tracking de jugadores

El pipeline utiliza detección YOLO y `model.track(...)` de Ultralytics con
BoT-SORT. Mantiene observaciones por frame, agrupa observaciones por track y
aplica una capa de canonicalización para reducir cambios de identidad entre
fragmentos de vídeo y cortes de plano.

### 🟢 Clasificación de equipos

`TeamClassifier` construye descriptores visuales de la zona del torso usando
OpenCV (RGB, HSV, LAB, histogramas y densidad de bordes). Los perfiles se
normalizan de forma robusta y se separan en dos grupos mediante KMeans.

### 🧤 Identificación de porteros y árbitro

Después de asignar los tracks a los dos equipos, `VisionService` aplica
heurísticas de apariencia y continuidad temporal para seleccionar candidatos a
portero y árbitro. Estas identidades se utilizan al generar las etiquetas del
vídeo.

La clasificación no depende de una base de datos de jugadores ni de nombres
individuales.

## 🔄 ¿Cómo funciona?

```text
Navegador
    |
    v
Nginx / React
    |
    +--> /api/v1 --> FastAPI / Uvicorn
                         |
                         +--> StatsBomb
                         |
                         +--> PostgreSQL
                         |
                         +--> XGBoost + SHAP
                         |
                         +--> YOLO + BoT-SORT + OpenCV + FFmpeg
```

El frontend utiliza `VITE_API_BASE_URL` para comunicarse con el backend. Los
routers delegan en servicios, repositorios, modelos locales o el proveedor
StatsBomb según el caso.

## 🏗️ Arquitectura

### Backend

`src/main.py` crea la aplicación FastAPI, registra CORS, rate limiting,
request IDs, documentación personalizada y los routers bajo `/api/v1`.

Los módulos principales son:

- `football`: jugadores, partidos y persistencia básica;
- `analytics`: eventos, StatsBomb, redes de pases y heatmaps;
- `ml_engine`: inferencia táctica y utilidades de features;
- `computer_vision`: upload, tracking, clasificación y renderizado de vídeo;
- `core`: configuración, base de datos, seguridad, excepciones y límites.

### Frontend

`frontend/src/App.tsx` gestiona la navegación entre Home, simulador táctico,
red de pases, heatmaps y visión por computador. Las pantallas realizan
peticiones HTTP directamente al backend y dibujan los resultados con React,
SVG o `canvas`.

### Servicios externos y persistencia

- StatsBomb proporciona competiciones, partidos y eventos para analytics.
- PostgreSQL almacena jugadores, partidos y eventos.
- Los modelos locales proporcionan la inferencia táctica y de visión.
- FFmpeg convierte el vídeo temporal generado por OpenCV a MP4.

## 🎥 Pipeline de Computer Vision

El flujo actual de `VisionService` es:

1. El endpoint recibe el vídeo.
2. Se valida el nombre, extensión, MIME, magic bytes, tamaño, duración y
   resolución.
3. OpenCV abre el vídeo y lee los frames.
4. YOLO detecta las personas en cada frame.
5. BoT-SORT mantiene los tracks entre frames.
6. Se extraen observaciones geométricas y de apariencia.
7. Se eliminan observaciones duplicadas y se limitan los tracks por frame.
8. Se detectan cambios de plano y se reinicia la memoria de identidad cuando
   corresponde.
9. Se canonicalizan IDs y se acumulan observaciones por trayectoria.
10. En una primera fase se crean perfiles visuales por track.
11. `TeamClassifier` genera features cromáticas y KMeans agrupa los dos equipos.
12. Se incorporan los tracks excluidos temporalmente y se calculan confianzas.
13. Se aplican heurísticas para porteros y árbitro.
14. Se reabre el vídeo en una segunda pasada.
15. Se dibujan elipses, colores, IDs y etiquetas sobre cada frame.
16. OpenCV escribe un AVI temporal y FFmpeg lo convierte al MP4 final.

```text
vídeo
  -> validación
  -> OpenCV
  -> YOLO
  -> BoT-SORT
  -> observaciones e IDs canónicos
  -> perfiles de apariencia
  -> KMeans / equipos
  -> portero y árbitro
  -> segunda pasada
  -> renderizado
  -> FFmpeg
  -> MP4
```

## 🖥️ Frontend

### Home

Presenta las áreas disponibles y permite navegar por la aplicación.

### Tactical Simulator

Permite modificar altura defensiva, intensidad de presión, posesión y
amplitud. Muestra xG a favor, xG en contra y el desglose SHAP de la
predicción.

### Pass Networks

Carga competiciones y partidos desde StatsBomb, permite seleccionar un equipo
y dibuja nodos y conexiones de pase sobre un campo.

### Player Heatmaps

Carga la información de un partido y equipo, dibuja eventos y posiciones, y
permite consultar los datos de jugadores con actividad.

### Computer Vision

Permite seleccionar un vídeo, subirlo al endpoint de visión y reproducir el
MP4 devuelto por el backend.

El frontend no contiene una pantalla de gestión CRUD de jugadores, aunque esas
operaciones sí están disponibles mediante la API.

## 🔌 API

| Área      | Método y endpoint                                                    | Descripción                   |
| --------- | -------------------------------------------------------------------- | ----------------------------- |
| Jugadores | `POST /api/v1/players/`                                              | Crea un jugador               |
| Jugadores | `GET /api/v1/players/{player_id}`                                    | Obtiene un jugador            |
| Partidos  | `POST /api/v1/matches/`                                              | Crea un partido               |
| Eventos   | `POST /api/v1/matches/{match_id}/events/upload`                      | Valida e importa eventos JSON |
| Eventos   | `GET /api/v1/matches/{match_id}/events`                              | Consulta eventos almacenados  |
| StatsBomb | `GET /api/v1/matches/statsbomb/competitions`                         | Lista competiciones           |
| StatsBomb | `GET /api/v1/matches/statsbomb/matches/{competition_id}/{season_id}` | Lista partidos                |
| Analytics | `GET /api/v1/matches/{match_id}/pass-network/{team_name}`            | Genera una red de pases       |
| Analytics | `GET /api/v1/matches/{match_id}/heatmaps/{team_name}`                | Genera datos de heatmap       |
| ML        | `POST /api/v1/simulator/predict`                                     | Predice xG y devuelve SHAP    |
| Vision    | `POST /api/v1/vision/process-video`                                  | Procesa y devuelve un vídeo   |

También existe `GET /health` como comprobación básica del servicio.

## 🗄️ Datos y base de datos

La persistencia utiliza SQLAlchemy asíncrono, `asyncpg` y PostgreSQL.

Las entidades actuales son:

- `players`: nombre, fecha de nacimiento y pie dominante;
- `matches`: fecha e identificador;
- `events`: minuto, segundo, tipo, coordenadas, partido relacionado y detalles
  originales en `JSONB`.

Los repositorios encapsulan las consultas de jugadores y eventos. Alembic se
configura en `alembic.ini` y las revisiones se encuentran en `migrations/`.

La base de datos se utiliza para persistir jugadores, partidos y eventos
importados. Las consultas de StatsBomb para redes de pases y heatmaps se
realizan mediante el proveedor externo.

## 🔐 Seguridad y robustez

El código actual incluye:

- rate limiting con `slowapi` en los endpoints;
- request IDs y manejo centralizado de excepciones inesperadas;
- validación de nombre, extensión, MIME y magic bytes de uploads;
- límite de tamaño de vídeo de 25 MB por defecto;
- límite de duración de 45 segundos por defecto;
- límite de resolución de 3840×2160;
- límite de 5 MB y 20.000 eventos por upload JSON;
- semáforo para limitar vídeos concurrentes;
- timeout configurable de procesamiento de vídeo;
- `hide_parameters=True` en el engine SQLAlchemy;
- exclusión de `.env`, caches, vídeos, datasets y resultados generados en
  `.gitignore` y `.dockerignore`.

Estas medidas mejoran la validación y la robustez operativa, pero no
constituyen un modelo completo de seguridad.

## 🛠️ Tecnologías utilizadas

### Frontend

- React;
- TypeScript;
- Vite;
- Tailwind CSS;
- SVG y Canvas para visualizaciones.

### Backend

- Python;
- FastAPI;
- Uvicorn;
- Pydantic y Pydantic Settings;
- `python-multipart`;
- `slowapi`.

### Base de datos

- PostgreSQL;
- SQLAlchemy async;
- `asyncpg`;
- Alembic.

### Machine Learning y datos

- NumPy;
- pandas;
- XGBoost;
- SHAP;
- scikit-learn;
- StatsBomb API mediante `statsbombpy`.

### Computer Vision

- OpenCV;
- PyTorch;
- Ultralytics YOLO;
- BoT-SORT;
- KMeans.

### Sistema e infraestructura

- FFmpeg;
- Docker;
- Docker Compose;
- Nginx;
- CUDA cuando está disponible;
- Node.js/npm para el frontend;

## 📦 Modelos utilizados

### Modelos utilizados en runtime

| Archivo                                                         | Uso                                                   |
| --------------------------------------------------------------- | ----------------------------------------------------- |
| `yolo-person-ball-v1.pt`                                        | Pesos YOLO cargados por `VisionService`               |
| `tactical_models.pkl`                                           | Modelos XGBoost y orden de features para el simulador |
| `src/modules/computer_vision/application/botsort_football.yaml` | Configuración BoT-SORT activa                         |

Las rutas de los modelos pueden cambiarse mediante `FOOTBALL_MODEL_PATH`,
`TACTICAL_MODELS_PATH` y `FOOTBALL_TRACKER_PATH`.

### Procedencia del modelo YOLO

El modelo local `yolo-person-ball-v1.pt` fue fine-tuned a partir del modelo
documentado `martinjolif/yolo-football-player-detection`. El dataset
relacionado es `martinjolif/football-player-detection`, con la transformación
de clases documentada en `NOTICE.md`.

La procedencia del modelo, el dataset, Roboflow y Ultralytics se describe en
[NOTICE.md](NOTICE.md). Los materiales derivados pueden estar sujetos a
condiciones adicionales; esta documentación no constituye asesoramiento
jurídico.

## ⚙️ Configuración

Desde la raíz del repositorio:

```bash
cp .env.example .env
```

Después hay que sustituir el placeholder de `DATABASE_URL` por una conexión
válida y ajustar las rutas si los artefactos no se encuentran en sus
ubicaciones predeterminadas.

| Variable                        | Propósito                           | Valor de plantilla                                                |
| ------------------------------- | ----------------------------------- | ----------------------------------------------------------------- |
| `DATABASE_URL`                  | Conexión PostgreSQL                 | Placeholder local                                                 |
| `FOOTBALL_MODEL_PATH`           | Pesos YOLO                          | `./yolo-person-ball-v1.pt`                                        |
| `TACTICAL_MODELS_PATH`          | Bundle táctico                      | `./tactical_models.pkl`                                           |
| `FOOTBALL_TRACKER_PATH`         | Configuración BoT-SORT              | `./src/modules/computer_vision/application/botsort_football.yaml` |
| `LOG_LEVEL`                     | Nivel de logging                    | `INFO`                                                            |
| `SQLALCHEMY_ECHO`               | Logging SQL                         | `false`                                                           |
| `MAX_VIDEO_UPLOAD_MB`           | Tamaño máximo de vídeo              | `25`                                                              |
| `MAX_VIDEO_DURATION_SECONDS`    | Duración máxima                     | `45`                                                              |
| `MAX_CONCURRENT_VIDEOS`         | Vídeos simultáneos                  | `1`                                                               |
| `VIDEO_PROCESS_TIMEOUT_SECONDS` | Espera máxima del endpoint de vídeo | `240`                                                             |
| `MAX_EVENT_UPLOAD_BYTES`        | Tamaño máximo JSON                  | `5242880`                                                         |
| `MAX_EVENTS_PER_UPLOAD`         | Eventos máximos por archivo         | `20000`                                                           |
| `VITE_API_BASE_URL`             | URL base del frontend               | `http://localhost:8000/api/v1`                                    |

## 🐳 Ejecución con Docker

El proyecto puede ejecutarse mediante Docker Compose utilizando tres servicios:

- `frontend`: aplicación React/Vite compilada y servida mediante Nginx;
- `backend`: API FastAPI ejecutada con Python 3.12, Uvicorn y las dependencias de Machine Learning y Computer Vision;
- `db`: base de datos PostgreSQL con almacenamiento persistente mediante un volumen Docker.

La arquitectura de ejecución es:

```text
Navegador
    |
    v
Nginx / React
    |
    +--> /api/v1 --> FastAPI / Uvicorn
                         |
                         +--> PostgreSQL
                         |
                         +--> StatsBomb
                         |
                         +--> XGBoost + SHAP
                         |
                         +--> YOLO + BoT-SORT + OpenCV + FFmpeg
```

### Requisitos

Para ejecutar el proyecto con Docker se necesita:

- Docker Engine;
- Docker Compose.

La configuración Docker actual está orientada a CPU. PyTorch utiliza CUDA cuando detecta una GPU compatible, pero la aplicación también puede ejecutarse en CPU.

### Configuración

Copia el archivo de variables de entorno de ejemplo:

```bash
cp .env.example .env
```

Configura las variables necesarias para PostgreSQL en `.env`:

```env
POSTGRES_USER=football_user
POSTGRES_PASSWORD=<tu_contraseña>
POSTGRES_DB=football_db
```

El archivo `.env` es exclusivamente local y no debe subirse al repositorio.

Las rutas de los modelos también pueden configurarse mediante variables de entorno. En Docker se utilizan las siguientes rutas internas:

```env
FOOTBALL_MODEL_PATH=/app/yolo-person-ball-v1.pt
FOOTBALL_TRACKER_PATH=/app/src/modules/computer_vision/application/botsort_football.yaml
TACTICAL_MODELS_PATH=/app/tactical_models.pkl
```

El frontend se compila utilizando `/api/v1` como base de la API. Nginx redirige internamente las peticiones hacia el servicio backend.

### Construcción de las imágenes

Desde la raíz del repositorio:

```bash
docker compose build
```

La imagen del backend incluye:

- Python 3.12;
- FFmpeg;
- OpenCV y sus dependencias del sistema;
- PyTorch;
- Ultralytics;
- scikit-learn;
- XGBoost;
- SHAP;
- `lap` para el tracking utilizado por BoT-SORT;
- el resto de dependencias definidas en `pyproject.toml`.

La imagen del frontend utiliza una construcción multietapa: Node.js se utiliza durante el build y Nginx sirve posteriormente los archivos estáticos generados.

### Arranque

Para iniciar los servicios en segundo plano:

```bash
docker compose up -d
```

Comprobar el estado:

```bash
docker compose ps
```

Los servicios principales son:

```text
frontend
backend
db
```

PostgreSQL incluye un healthcheck y el backend espera a que la base de datos esté disponible antes de iniciarse.

### Migraciones

Las migraciones de Alembic se ejecutan dentro del contenedor backend:

```bash
docker compose exec backend alembic upgrade head
```

Para consultar la revisión actual:

```bash
docker compose exec backend alembic current
```

### Acceso a la aplicación

Una vez iniciados los servicios:

```text
Frontend:
http://localhost

Backend:
http://localhost:8000

Swagger:
http://localhost:8000/docs
```

El endpoint de comprobación básica del backend es:

```text
GET /health
```

### Comunicación entre servicios

Dentro de la red Docker, el backend se conecta a PostgreSQL mediante el nombre del servicio `db`:

```text
postgresql+asyncpg://<usuario>:<contraseña>@db:5432/<base_de_datos>
```

El navegador no utiliza el nombre interno `backend`. Las peticiones del frontend a `/api/v1` pasan por Nginx y son redirigidas internamente al servicio FastAPI.

### Computer Vision en Docker

El contenedor backend incluye las dependencias necesarias para ejecutar el pipeline de Computer Vision:

- PyTorch;
- Ultralytics;
- OpenCV;
- BoT-SORT;
- KMeans;
- FFmpeg.

Los artefactos necesarios para el runtime son:

```text
yolo-person-ball-v1.pt
tactical_models.pkl
src/modules/computer_vision/application/botsort_football.yaml
```

El proceso utiliza CPU cuando CUDA no está disponible.

Los directorios temporales utilizados por el pipeline se encuentran dentro del contenedor:

```text
/tmp/football_platform/uploads
/tmp/football_platform/outputs
```

El endpoint de vídeo mantiene las restricciones de tamaño, duración, resolución, concurrencia y timeout definidas por la configuración de la aplicación.

El timeout de procesamiento puede configurarse mediante:

```env
VIDEO_PROCESS_TIMEOUT_SECONDS=240
```

El procesamiento de vídeo puede tardar varios minutos en CPU dependiendo de la duración y complejidad del vídeo.

### Detener los servicios

Para detener los contenedores:

```bash
docker compose down
```

El volumen de PostgreSQL no se elimina al ejecutar este comando, por lo que los datos persistentes se conservan.

Para eliminar también el volumen de PostgreSQL:

```bash
docker compose down -v
```

Esta última operación elimina los datos persistidos de la base de datos.

### Logs

Para consultar los logs del backend:

```bash
docker compose logs -f backend
```

Para consultar los logs de todos los servicios:

```bash
docker compose logs -f
```

### Estructura Docker

Los archivos principales relacionados con la contenerización son:

```text
Dockerfile
docker-compose.yml
frontend/
  Dockerfile
  nginx.conf
```

El backend utiliza una imagen independiente, el frontend se compila mediante una imagen multietapa y PostgreSQL se ejecuta en un servicio separado.

Los archivos `.env`, datasets, vídeos, resultados temporales, `node_modules`, entornos virtuales y artefactos experimentales se excluyen mediante `.dockerignore`.

## 📥 Instalación

### Backend

Se recomienda Python 3.12 y un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

Configura PostgreSQL, completa `DATABASE_URL` y ejecuta las migraciones:

```bash
alembic upgrade head
```

Los modelos necesarios para ejecutar la aplicación deben estar disponibles en las rutas configuradas.

Para Computer Vision también se necesita FFmpeg instalado y disponible en
`PATH`.

### Frontend

```bash
cd frontend
npm install
npm run build
```

El archivo `frontend/package.json` define además los comandos `dev`, `lint` y
`preview`.

## ▶️ Ejecución

Desde la raíz del repositorio, con PostgreSQL disponible:

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

En otra terminal:

```bash
cd frontend
npm run dev
```

El frontend utilizará `VITE_API_BASE_URL` para localizar el backend.

## 📁 Estructura del proyecto

```text
src/
  core/
    config.py
    database.py
    rate_limit.py
    security.py
  modules/
    football/
    analytics/
    ml_engine/
    computer_vision/
  main.py
frontend/
  src/
    components/
migrations/
  env.py
  versions/
yolo-person-ball-v1.pt
tactical_models.pkl
LICENSE
NOTICE.md
pyproject.toml
```

Los datasets, vídeos, caches, dependencias instaladas y salidas generadas no
forman parte de la estructura runtime documentada.

## ⚠️ Limitaciones actuales

- El procesamiento de vídeo es costoso y puede superar el timeout si se
  ejecuta en CPU o con vídeos complejos.
- CUDA se utiliza cuando PyTorch la detecta, pero no es obligatoria; el
  rendimiento en CPU puede ser considerablemente menor.
- El endpoint de visión depende de los pesos YOLO, BoT-SORT, OpenCV y FFmpeg.
- El simulador necesita un `tactical_models.pkl` compatible con el código.
- Analytics depende de la disponibilidad y estructura de datos de StatsBomb.
- La persistencia depende de PostgreSQL y de las migraciones aplicadas.
- El límite de vídeo es 25 MB y 45 segundos por defecto.

## 🧪 Estado del proyecto

### ✅ Funcionalidades activas

- API FastAPI y frontend React/Vite;
- gestión mínima de jugadores en backend (alta y consulta);
- creación de partidos y persistencia de eventos;
- consultas StatsBomb;
- redes de pases y heatmaps;
- simulador táctico XGBoost con explicaciones SHAP;
- tracking y renderizado de personas en vídeo;
- clasificación de equipos mediante KMeans;
- heurísticas de porteros y árbitro;
- migraciones Alembic para las tablas actuales.

## 📄 Licencia

El código propio del proyecto se distribuye bajo
[GNU Affero General Public License v3.0](LICENSE).

Las licencias y atribuciones de terceros no quedan sustituidas por la licencia
del proyecto. Consulta [NOTICE.md](NOTICE.md) antes de redistribuir modelos,
datasets o material derivado.
