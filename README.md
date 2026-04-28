# EEG Motor Imagery Classification — BCI Competition IV-2b

**Trabajo de Grado — Maestría en Ingeniería**  
Institución Universitaria de Envigado, 2026  
Autor: Juan Carlos Guerrero Sierra  
Asesor: Hernán Darío Villota Bolaños

---

## Descripción

Pipeline reproducible para la clasificación binaria de imaginación motora (mano izquierda vs. derecha) a partir de señales EEG, utilizando espectrogramas ERSP como entrada a redes neuronales convolucionales (CNN) ligeras.

**Dataset:** BCI Competition IV – Dataset 2b  
**Arquitecturas evaluadas:** EEGNet, ShallowConvNet, SpectNet  
**Entregable académico:** Artículo científico sometido a revista indexada

---

## Estructura del repositorio

```
bci-iv2b-ersp-cnn/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/          ← coloca aquí los archivos GDF (B01T.gdf ... B09E.gdf)
│   └── processed/    ← épocas y espectrogramas generados automáticamente
├── notebooks/
│   └── 01_explorar_dataset.ipynb   ← visualización y verificación del dataset
├── src/
│   ├── config.py          ← todos los hiperparámetros centralizados
│   ├── preprocessing.py   ← carga, filtrado, ICA, epoching
│   ├── ersp.py            ← generación de espectrogramas ERSP
│   ├── dataset.py         ← PyTorch Dataset para imágenes ERSP
│   ├── models/
│   │   ├── eegnet.py
│   │   ├── shallowconvnet.py
│   │   └── spectnet.py
│   ├── train.py           ← entrenamiento y validación
│   └── evaluate.py        ← métricas, matriz de confusión, Grad-CAM
├── results/
│   ├── figures/           ← gráficos generados
│   └── metrics/           ← CSVs con resultados
└── .gitignore
```

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/bci-iv2b-ersp-cnn.git
cd bci-iv2b-ersp-cnn

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Uso rápido

### Paso 1 — Verificar el dataset
```bash
# Coloca los 18 archivos GDF en data/raw/ y ejecuta:
python src/preprocessing.py --verify
```

### Paso 2 — Exploración visual (notebook)
```bash
jupyter notebook notebooks/01_explorar_dataset.ipynb
```

### Paso 3 — Preprocesamiento completo
```bash
python src/preprocessing.py --subject all
```

### Paso 4 — Generar espectrogramas ERSP
```bash
python src/ersp.py
```

### Paso 5 — Entrenar modelos
```bash
python src/train.py --model eegnet
python src/train.py --model shallowconvnet
python src/train.py --model spectnet
```

### Paso 6 — Evaluación comparativa
```bash
python src/evaluate.py
```

---

## Dataset

El BCI-IV-2b no se incluye en este repositorio. Descárgalo desde:  
https://www.bbci.de/competition/iv/download/

Coloca los 18 archivos en `data/raw/`:
- Entrenamiento: `B01T.gdf` ... `B09T.gdf`
- Evaluación: `B01E.gdf` ... `B09E.gdf`

---

## Parámetros del pipeline

Todos los parámetros están centralizados en `src/config.py`. Los principales:

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Filtro pasa banda | 8–30 Hz | Bandas mu y beta (actividad motora) |
| Ventana de análisis | -0.5 a 4.0 s | Incluye línea base pre-estímulo |
| Transformada | STFT (Hann) | Balance resolución tiempo-frecuencia |
| Longitud de ventana STFT | 256 muestras (1.024 s) | Resolución frecuencial de ~1 Hz |
| Solapamiento STFT | 75% (64 muestras de paso) | Resolución temporal suficiente |
| Rango frecuencial | 8–30 Hz | 22 bins de frecuencia |
| Normalización | ERSP divisiva (dB) | Relativa a línea base pre-estímulo |
| Dimensión imagen | 22 × 128 px | Por canal y por ensayo |
| División train/test | Sesiones 1-3 / 4-5 | Protocolo estándar BCI-IV-2b |

---

## Resultados esperados

*(Esta sección se completará con los resultados experimentales)*

---

## Cita

Si utilizas este código, por favor cita:

```
Guerrero Sierra, J.C. (2026). Clasificación de imaginación motora EEG 
mediante CNN y espectrogramas ERSP sobre BCI-IV-2b. Trabajo de Grado, 
Maestría en Ingeniería, Institución Universitaria de Envigado.
```

---

## Licencia

MIT License — ver archivo `LICENSE`
