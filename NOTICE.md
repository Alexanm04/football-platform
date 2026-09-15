# NOTICE

Este proyecto contiene o utiliza componentes de terceros sujetos a sus
propias licencias. La licencia AGPL-3.0 del proyecto no sustituye ni cambia
las licencias aplicables a esos componentes.

## Ultralytics YOLO

Ultralytics YOLO se utiliza para la detección y el procesamiento de vídeo.

- Licencia declarada: AGPL-3.0
- Sitio oficial: https://www.ultralytics.com/license
- Documentación oficial: https://docs.ultralytics.com/
- Licencia del repositorio: https://github.com/ultralytics/ultralytics/blob/main/LICENSE

## Modelo base

El modelo base utilizado para el fine-tuning local es:

- `martinjolif/yolo-football-player-detection`
- URL: https://huggingface.co/martinjolif/yolo-football-player-detection
- Licencia declarada por el model card: AGPL-3.0
- Modelo base declarado: `Ultralytics/YOLO11m`

## Modelo local

`yolo-person-ball-v1.pt` es un modelo fine-tuned localmente a partir de
`martinjolif/yolo-football-player-detection` y fue entrenado en este proyecto.

La procedencia y las condiciones de redistribución del modelo deben
interpretarse conjuntamente con las licencias de Ultralytics, del modelo
base y del dataset utilizado.

## Dataset

El dataset utilizado es:

- `martinjolif/football-player-detection`
- URL: https://huggingface.co/datasets/martinjolif/football-player-detection
- Licencia declarada: CC BY 4.0
- Licencia: https://creativecommons.org/licenses/by/4.0/

El dataset referencia el proyecto original de Roboflow Universe:

- Football project:
  https://universe.roboflow.com/football-project-pifbc/football-players-detection-3zvbc-yyhdl

El dataset fue modificado mediante `merge_classes.py`:

- `player`
- `goalkeeper`
- `referee`

se agrupan como `person`, y se conserva la clase `ball`.

La publicación de modelos, datasets o material derivado puede estar sujeta a
derechos y condiciones adicionales. Este aviso no constituye asesoramiento
jurídico ni garantiza que toda redistribución esté libre de riesgo.
