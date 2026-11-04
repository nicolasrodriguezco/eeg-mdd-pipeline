# Revisión técnica: detección de depresión con EEG (MODMA), rehecha desde cero

> Objetivo: convertir el proyecto *"Análisis de biomarcadores EEG para el diagnóstico de depresión"* en un trabajo publicable en arXiv (y, si es posible, en un workshop o conferencia), con resultados honestos y reproducibles.

---

## 0. Diagnóstico de la versión actual

| # | Problema | Gravedad | Consecuencia |
|---|---|---|---|
| 1 | Separación entrenamiento/prueba **por ensayo, no por sujeto** | 🔴 Crítica | Fuga de información: el modelo puede reconocer personas, no depresión. El AUC de 0,86 probablemente está inflado. |
| 2 | `gender` es la variable más importante del Random Forest | 🔴 Crítica | Posible confusor: con unos 53 sujetos, el modelo puede estar aprendiendo el sexo. |
| 3 | Se usa la señal cruda (25.728 columnas) aunque el título habla de "biomarcadores" | 🟠 Alta | No hay biomarcadores interpretables, hay demasiadas dimensiones para pocos sujetos y el resultado no se puede comparar con la literatura. |
| 4 | Una sola partición de prueba, sin validación cruzada ni intervalos de confianza | 🟠 Alta | No se sabe si el resultado es estable o fruto del azar. |
| 5 | Filtro de red eléctrica a 60/120 Hz | 🟠 Alta | MODMA se registró en China, donde la red es de **50 Hz**. Hay que verificarlo en el artículo del dataset. |
| 6 | Sin revisión de literatura ni referencias | 🟠 Alta | No se puede ubicar el aporte frente a lo publicado. |
| 7 | La TCN figura como "modelo elegido" pero no está implementada | 🟡 Media | El texto promete algo que no entrega. |
| 8 | Inconsistencias (53 vs. 58 sujetos) y texto en español | 🟡 Media | Resta credibilidad y alcance. |
| 9 | **28 sujetos no coincidían con los datos clínicos** y se "arreglaron" normalizando códigos | 🔴 Crítica | Si algún cruce quedó mal, hay sujetos con la etiqueta MDD/HC equivocada, y ningún modelo lo corrige después. |
| 10 | El dataset se construyó con un script generado con IA (GPT-4) sin pruebas | 🟠 Alta | No hay garantía de que cada paso haga lo que se cree. Hay que rehacerlo desde los datos crudos. |

---

## Fase 0. Reconstruir el dataset desde los datos crudos (4,8 GB)

> Todo el estudio depende de esta fase. Un error aquí contamina todo lo que viene después.

**0.1 Permisos**
- [ ] Revisar el acuerdo de uso con el que la entidad entregó los datos: ¿permite publicar resultados?, ¿qué hay que citar?, ¿prohíbe compartir datos derivados?
- [ ] Confirmar si es MODMA original o una versión procesada por la entidad.

**0.2 Inventario sin modificar nada** (`scripts/01_inventory.py`)
- [ ] Copiar los crudos a `data/raw/` y dejarlos en **solo lectura**.
- [ ] Listar archivos, extensiones, tamaños y la estructura de carpetas.
- [ ] Ubicar dónde está la etiqueta MDD/HC y cómo se llama cada sujeto en cada archivo.

**0.3 Auditoría por sujeto** (`scripts/02_audit.py`)
- [ ] Tabla con una fila por sujeto: frecuencia de muestreo, número de canales, duración, eventos por código, canales planos y amplitud máxima.
- [ ] Revisar visualmente la señal de algunos sujetos de cada grupo (`raw.plot()`).
- [ ] Documentar qué significa cada código de evento y llenar `event_id` en `configs/config.yaml`.

**0.4 Cruce de identificadores** (`scripts/03_match_ids.py`)
- [ ] Cruzar EEG con datos clínicos **sin arreglos automáticos**. Todo caso dudoso va a `configs/manual_id_map.csv` con una justificación escrita.
- [ ] Ningún sujeto sin etiqueta, duplicado o con dos etiquetas pasa a la siguiente fase.

**0.5 Preprocesamiento nuevo** (`scripts/04_preprocess.py`)
- [ ] MNE-Python, guardando épocas en `.fif` por sujeto (no una tabla gigante de 25.000 columnas).
- [ ] Un registro por sujeto de cuántos ensayos quedan por condición.
- [ ] Pruebas automáticas (`pytest`) para cada paso crítico.

**0.6 Comparación con la versión vieja** (`scripts/05_compare_old.py`)
- [ ] Comparar sujetos, etiquetas y ensayos por condición entre el dataset viejo y el nuevo.
- [ ] Cada diferencia se investiga y se documenta: es un error encontrado en la versión anterior.

**0.7 Uso de IA**
- [ ] Puedes usar IA para escribir código, pero cada paso se entiende, se prueba y se valida. Si publican, declaren su uso: muchas revistas y conferencias lo piden.

---

## 1. Pregunta de investigación (reformulada)

- **Pregunta principal:** ¿se puede distinguir a personas con trastorno depresivo mayor (MDD) de controles sanos (HC) **en sujetos nunca vistos** por el modelo, usando EEG?
- **Hipótesis secundarias:**
  - Las características espectrales y de ERP (potenciales evocados) generalizan mejor entre sujetos que la señal cruda.
  - La condición emocional del estímulo (miedo, tristeza, alegría) cambia la capacidad de discriminación.
  - El desempeño reportado en trabajos previos con separación por ensayo cae al evaluar por sujeto.
- **Aporte esperado:** una evaluación **honesta e independiente del sujeto** sobre MODMA, con línea base reproducible. Que los resultados de otros estén inflados es en sí un aporte publicable.

---

## 2. Datos

- [ ] Confirmar qué parte de MODMA se usa: la tarea de EEG de 128 canales con rostros emocionales (*dot-probe*) y/o el registro en reposo.
- [ ] Tabla de sujetos: **n por grupo** (MDD/HC), edad, sexo, escolaridad, puntajes clínicos (PHQ-9 y otros). Resolver la discrepancia 53 vs. 58.
- [ ] Revisar licencia y forma de citar el dataset (artículo original de MODMA).
- [ ] Registrar ensayos por sujeto tras el rechazo de artefactos. Excluir sujetos con muy pocos ensayos válidos (definir un umbral).
- [ ] Revisar posibles confusores: diferencias de edad o sexo entre grupos, medicación y horario de registro.

---

## 3. Preprocesamiento (MNE-Python)

1. Cargar los datos con montaje EGI de 128 canales y verificar la frecuencia de muestreo.
2. Filtro pasa-banda 0,5–40 Hz (FIR de fase cero) y **notch a 50 Hz** (verificar).
3. Detectar e **interpolar canales malos**.
4. **ICA por sujeto** para quitar parpadeos y movimientos oculares (con ICLabel o una revisión manual documentada).
5. Re-referenciar al promedio.
6. Segmentar de −200 a 600 ms respecto al estímulo, con **corrección de línea base**.
7. Rechazo de épocas con `autoreject` o un umbral justificado (±100 µV).
8. Guardar la salida en formato estándar (BIDS o `.fif`) y dejar un **reporte por sujeto** (`mne.Report`).

> Regla de oro: todo lo que se ajuste con datos (ICA, escalado, PCA, selección de variables) se ajusta **dentro de cada sujeto** o **dentro del pliegue de entrenamiento**, nunca con todos los datos.

---

## 4. Características (biomarcadores)

| Familia | Ejemplos | Por qué |
|---|---|---|
| **Espectrales** | Potencia absoluta y relativa por banda (delta, theta, alfa, beta, gamma) con el método de Welch, por canal o región | Lo más usado y comparable en la literatura de EEG y depresión |
| **Asimetría** | Asimetría alfa frontal (mapear F3/F4 al montaje EGI) | Biomarcador clásico propuesto para depresión |
| **ERP** | Amplitud y latencia de P300, LPP y N170 por condición emocional | Aprovecha que la tarea tiene rostros emocionales |
| **Conectividad** | Coherencia, PLV, wPLI entre regiones | Hipótesis de conectividad alterada en MDD |
| **Complejidad** | Entropía muestral, dimensión fractal de Higuchi, Lempel-Ziv | Se reportan diferencias en la complejidad de la señal |
| **Riemanniana** | Matrices de covarianza en espacio tangente (pyRiemann) | Línea base muy fuerte para EEG con pocos sujetos |

- **Reducir dimensiones con sentido fisiológico:** agrupar los 128 canales en unas 8–10 regiones (frontal, central, parietal, occipital, temporal; izquierda y derecha).
- **No usar `gender` como predictor.** Usarlo solo para análisis de sesgo (sección 7).

---

## 5. Modelos

**Líneas base (obligatorias):**
- Regresión logística con regularización L1/L2 sobre características.
- SVM (lineal y RBF).
- Random Forest o XGBoost con profundidad limitada.
- Espacio tangente riemanniano + regresión logística.

**Deep learning (comparación):**
- EEGNet y ShallowConvNet (librería *braindecode*).
- TCN, ya que estaba en el plan original.
- Con unos 53 sujetos, el deep learning puede sobreajustar. Usar aumentación de datos, *early stopping* y reportarlo igual aunque no gane.

**De ensayo a sujeto:**
- Los modelos predicen por ensayo, pero **el diagnóstico es por persona**. Agregar las probabilidades de todos los ensayos de un sujeto (promedio o mediana) y reportar las métricas **por sujeto**.

---

## 6. Validación (la parte más importante)

- [ ] **Leave-one-subject-out (LOSO)** o **GroupKFold estratificado por sujeto**, repetido con varias semillas.
- [ ] **Validación cruzada anidada:** los hiperparámetros se eligen en un bucle interno, solo con los sujetos de entrenamiento.
- [ ] Todo el *pipeline* (escalado, PCA, selección) dentro de `sklearn.pipeline.Pipeline` para evitar fugas.
- [ ] **Test de permutación** a nivel sujeto (barajar etiquetas por sujeto, unas 1.000 veces) para saber si el resultado supera al azar.
- [ ] **Experimento de control:** repetir con separación por ensayo para **cuantificar la inflación**. Es una figura clave del paper.

---

## 7. Sesgos y confusores

- [ ] ¿El EEG predice el sexo o la edad? Si sí, controlar (estratificar, emparejar o regresar el confusor).
- [ ] Reportar el desempeño por sexo.
- [ ] Comparar el modelo con uno que use **solo variables demográficas**: si rinde parecido, el EEG no aporta.
- [ ] Revisar si hay diferencias sistemáticas de calidad de señal entre grupos (número de ensayos rechazados, canales malos).

---

## 8. Métricas y estadística

- **Por sujeto:** exactitud balanceada, AUC-ROC, sensibilidad, especificidad, F1, y calibración (*Brier score*).
- **Intervalos de confianza al 95 %** por *bootstrap* sobre sujetos.
- **Nivel de azar real** para el n de la muestra, calculado con el test de permutación.
- **Comparar modelos** con pruebas pareadas sobre los mismos pliegues, no por diferencias puntuales.

---

## 9. Interpretabilidad

- Importancia por permutación **calculada dentro de la validación cruzada**.
- SHAP para los modelos de características.
- **Topomapas** de las características más discriminativas (por banda y región).
- Para el deep learning: mapas de saliencia, con cautela.
- Discutir si los hallazgos coinciden con la literatura (por ejemplo, alfa frontal o ERP emocionales).

---

## 10. Reproducibilidad

```
repo/
├── README.md               # cómo correr todo de principio a fin
├── environment.yml         # o pyproject.toml con versiones fijas
├── configs/                # hiperparámetros y rutas en YAML
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   └── evaluation.py
├── notebooks/              # solo exploración y figuras
├── reports/                # mne.Report por sujeto y figuras finales
└── book/                   # el Jupyter Book como documentación
```

- Semillas fijas y registro de experimentos (MLflow o Weights & Biases).
- Una *data card* (qué datos, qué sujetos excluidos y por qué) y una *model card* (uso previsto y limitaciones).
- Advertencia explícita: **no es una herramienta diagnóstica clínica.**

---

## 11. Literatura mínima a revisar

- El artículo original del dataset **MODMA** y los trabajos que lo usan para clasificar depresión. Anotar en cada uno **si separan por sujeto o por ensayo**.
- Saeb et al. (2017), sobre validación por registro vs. por sujeto en ML clínico.
- Varoquaux (2018), sobre las barras de error grandes con muestras pequeñas.
- Combrisson y Jerbi (2015), sobre el nivel de azar en clasificación con neurodatos.
- Revisiones de biomarcadores EEG en depresión (alfa frontal, ERP emocionales).

---

## 12. El artículo (inglés, ~8 páginas)

1. **Abstract**
2. **Introduction:** el problema clínico y el vacío (evaluaciones infladas)
3. **Related work:** tabla de estudios con MODMA y su tipo de validación
4. **Data and preprocessing**
5. **Features and models**
6. **Evaluation protocol:** LOSO, validación anidada, permutaciones
7. **Results:** tabla principal por sujeto con IC 95 %, figura de inflación (por ensayo vs. por sujeto), topomapas
8. **Discussion and limitations:** n pequeño, un solo sitio, sin validación externa
9. **Conclusion** y enlace al código

**Dónde publicar:** primero en arXiv, luego en un workshop de ML en salud o una conferencia latinoamericana de IA.

---

## 13. Cronograma propuesto (antes de la convocatoria del DAAD)

| Mes | Entregable |
|---|---|
| Nov. 2026 | **Fase 0:** permisos, inventario, auditoría y cruce de identificadores |
| Dic. 2026 | Revisión de literatura y tabla de estudios con MODMA, y comparación del dataset nuevo con el viejo |
| Ene. 2027 | Preprocesamiento nuevo y reporte por sujeto |
| Feb. 2027 | Características y líneas base con LOSO |
| Mar. 2027 | Modelos riemannianos y deep learning, más el análisis de confusores |
| Abr. 2027 | Estadística (IC, permutaciones) e interpretabilidad |
| May. 2027 | Borrador del artículo en inglés |
| Jun. 2027 | Revisión con el profesor del seminario y correcciones |
| **Jul. 2027** | **Publicación en arXiv**, antes de postular al DAAD en septiembre |

---

## 14. Checklist final

- [ ] Cada sujeto verificado a mano con su etiqueta MDD/HC correcta
- [ ] Ningún sujeto aparece en entrenamiento y prueba a la vez
- [ ] Todo ajuste ocurre dentro del pliegue de entrenamiento
- [ ] Métricas por sujeto, con IC 95 % y test de permutación
- [ ] Comparación contra una línea base solo demográfica
- [ ] `gender` fuera de las características
- [ ] Notch correcto (50 Hz, verificado)
- [ ] Código reproducible con un solo comando
- [ ] Literatura citada y comparada
- [ ] Limitaciones escritas con honestidad
