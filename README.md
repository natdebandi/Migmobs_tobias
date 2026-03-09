# Migmobs — LatamDiD

Replicación en Python del análisis de diferencias en diferencias (DiD) sobre el efecto de los **Acuerdos de Libre Movimiento de Personas (FMR)** en los flujos de migración y movilidad en América Latina.

El análisis replica la lógica del script Stata `104.04_LatamDiD_03_analysis.do` (Tobias Grohmann), usando `statsmodels` (GLM Poisson) como estimador PPML.

---

## Notebooks

| Archivo | Descripción |
|---|---|
| `01_FMR_scenarios.ipynb` | Preparación del dataset panel `did_latam_data`. Combina fuentes de migración (Gaskin & Abel 2025), movilidad (GTMD2) y códigos UN. Codifica los FMR como dummies corredor × año. |
| `01_LATAM_data_verification.ipynb` | Verificación y visualización de la serie temporal de visitantes internacionales a Argentina (GTMD2). |
| `02_build_latam_dataset.ipynb` | Construcción alternativa del dataset panel con fuentes adicionales (turismo ARG, UNWTO). |
| `03_vflow_imputation_diagnostic.ipynb` | Diagnóstico de datos faltantes en `vflow` para los 10 países de Sudamérica. Evalúa estrategias de imputación por corredor: rescalado con fuente alternativa, interpolación log-lineal, o exclusión. |
| `04_analysis_FMR_all_v5.ipynb` | **Análisis principal.** DiD PPML con `FMR_all` como tratamiento (todos los acuerdos de libre movimiento). Estima el Efecto de Tratamiento Promedio sobre los Tratados (PTE) para `mflow` y `vflow`, con bootstrap de 500 iteraciones clusterizado por corredor. |
| `04_analysis_FMR_mercosur_v5.ipynb` | Mismo análisis restringiendo el tratamiento a acuerdos **Mercosur** (`mercosur1` / `mercosur2`). |

## Script auxiliar

| Archivo | Descripción |
|---|---|
| `diagnostic_vflow_imputation.py` | Versión en script Python del diagnóstico de imputación (`03_vflow_imputation_diagnostic.ipynb`). |

---

## Output

| Archivo | Descripción |
|---|---|
| `Output/did_latam_data.csv` | Dataset panel principal (corredores × años). |
| `Output/did_latam_data_v2.csv` | Versión actualizada del dataset panel. |
| `Output/bootstrap_mflow_FMR_all_500.csv` | PTEs bootstrap — flujos de migración, FMR global (500 iteraciones). |
| `Output/bootstrap_vflow_FMR_all_500.csv` | PTEs bootstrap — flujos de visitantes, FMR global (500 iteraciones). |
| `Output/bootstrap_mflow_FMR_Mercosur_500.csv` | PTEs bootstrap — flujos de migración, FMR Mercosur (500 iteraciones). |
| `Output/bootstrap_vflow_FMR_Mercosur_500.csv` | PTEs bootstrap — flujos de visitantes, FMR Mercosur (500 iteraciones). |
| `Output/Fig01_*.png` / `Fig02_*.png` | Figuras de descripción del tratamiento y flujos. |
| `Output/diagnostic_vflow_imputation.csv` | Diagnóstico de faltantes en `vflow` por corredor. |

---

## Estrategia de identificación

El efecto del FMR se estima sobre la muestra **no tratada** (`TREATMENT == 0`), usando tres efectos fijos de alta dimensión:

| Efecto fijo | Controla por |
|---|---|
| Origen × Año | Factores del país emisor que varían en el tiempo (PBI, política migratoria, conflictos) |
| Destino × Año | Factores del país receptor (mercado laboral, política de admisión) |
| Corredor | Características fijas del par origen-destino (distancia, idioma, historia compartida) |

El contrafáctico para los corredores tratados se construye aplicando los efectos fijos estimados en la muestra de control al dataset completo. El **PTE (Proportional Treatment Effect)** sigue la metodología de Ninon Moreau-Kastler:

```
PTE = Σ(ITEs | tratados) / Σ(ŷ₀ | tratados)
```

---

## Dependencias

```
pandas, numpy, matplotlib, statsmodels, pyfixest, scipy
```
