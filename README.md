# eeg-mdd-pipeline

Reconstrucción desde cero del dataset y del preprocesamiento para el estudio
**"Detección de depresión (MDD vs. HC) con EEG"**, a partir de los datos crudos.

> Regla de oro: los datos crudos nunca se modifican, ningún sujeto pasa sin una
> etiqueta verificada, y todo lo que se ajusta con datos se ajusta por sujeto o
> dentro del pliegue de entrenamiento.

## Instalación

```bash
conda env create -f environment.yml
conda activate eeg-mdd
pip install -e .
```

## Estructura

```
configs/
  config.yaml            # rutas, parámetros y event_id (llenar tras la auditoría)
  manual_id_map.csv      # cruces de ID manuales, cada uno con justificación
data/
  raw/                   # datos crudos (SOLO LECTURA, no se versionan)
  interim/               # tablas de inventario, auditoría y cruce
  processed/             # épocas por sujeto en .fif
reports/                 # registros y figuras
scripts/                 # pasos del pipeline, en orden
src/eeg_mdd/             # código reutilizable
tests/                   # pruebas con EEG sintético
```

## Flujo (Fase 0)

| Paso | Comando | Salida | Qué revisar a mano |
|---|---|---|---|
| 1. Inventario | `python scripts/01_inventory.py` | `data/interim/inventory.csv` | Formatos, carpetas, dónde está la etiqueta |
| 2. Auditoría | `python scripts/02_audit.py` | `data/interim/audit.csv` | Canales, frecuencia, eventos por código, canales planos |
| 3. Cruce de IDs | `python scripts/03_match_ids.py` | `data/interim/subjects.csv` | Todo caso en `only_eeg`, `only_clinical` o `duplicated` |
| 4. Preprocesamiento | `python scripts/04_preprocess.py` | `data/processed/*-epo.fif` y `reports/preprocessing_log.csv` | Ensayos que quedan por sujeto y condición |
| 5. Comparación | `python scripts/05_compare_old.py --old ruta/al/viejo.parquet` | `reports/old_vs_new.csv` | Cada diferencia es un posible error de la versión vieja |

Antes del paso 4 hay que completar en `configs/config.yaml`:
- `dataset.subject_id_regex`: cómo extraer el ID del nombre del archivo.
- `dataset.clinical`: archivo, columna de ID y columna de grupo.
- `preprocessing.event_id`: códigos de evento de cada condición (sale de la auditoría).
- `preprocessing.notch_freqs`: **verificar 50 Hz** (MODMA se registró en China).

## Pruebas

```bash
pytest -q
```

Las pruebas generan EEG sintético de 128 canales, así que no necesitan los datos reales.

## Pendientes conocidos

- Si los datos vienen en `.mat`, hay que escribir el lector en `src/eeg_mdd/io.py` (`_read_mat`) según la estructura real del archivo.
- La ICA se ajusta y se guarda por sujeto, pero los componentes a eliminar se eligen a mano y se anotan en `configs/ica_exclusions.csv`.
