"""
diagnostic_vflow_imputation.py
==============================
Evalúa los datos faltantes en gtmd2_vflow_int para los 10 países de SA
y recomienda una estrategia de imputación corredor por corredor.

Fuentes alternativas consideradas:
  - UNWTO (unwto_tflow_112/111/122/121) — columnas del propio GTMD2
  - turismo_anio_ARG.csv — visitantes a ARG por origen (INDEC/Mintur, desde 1990)
  - tourism_UN_ARG_Latam_clean.csv — visitantes a ARG por país latinoamericano (UN, desde 1995/2016)

Técnicas evaluadas:
  A) Usar dato alternativo directamente (si la escala es comparable)
  B) Rescalar dato alternativo (ratio en años con ambas series)
  C) Interpolación log-lineal (si hay datos en ambos extremos del gap)
  D) Sin imputación posible (sin ninguna fuente alternativa ni datos adyacentes)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

# ── Rutas ─────────────────────────────────────────────────────────────────────
GTMD2_PATH       = "C:/Data/GTMD2/GTMD2_Data_MIGMOBS_share.csv"
ARG_TOUR_PATH    = "Data/turismo_anio_ARG.csv"
ARG_UN_PATH      = "Data/tourism_UN_ARG_Latam_clean.csv"
OUTPUT_PATH      = "Output/diagnostic_vflow_imputation.csv"

# ── Parámetros ─────────────────────────────────────────────────────────────────
SA10 = ["ARG", "BOL", "BRA", "CHL", "COL", "ECU", "PRY", "PER", "URY", "VEN"]
YEAR_MIN, YEAR_MAX = 1995, 2022
ALL_YEARS = list(range(YEAR_MIN, YEAR_MAX + 1))
N_YEARS   = len(ALL_YEARS)

GTMD2_COLS = [
    "iso3code_i", "iso3code_j", "year",
    "country_i", "country_j",
    "gtmd2_vflow_int",
    "unwto_tflow_112", "unwto_tflow_111",
    "unwto_tflow_122", "unwto_tflow_121",
]
UNWTO_COLS = ["unwto_tflow_112", "unwto_tflow_111", "unwto_tflow_122", "unwto_tflow_121"]

# Orígenes para ARG cubiertos por turismo_anio_ARG (INDEC)
ARG_TOUR_MAP = {"Bolivia": "BOL", "Brasil": "BRA", "Chile": "CHL",
                "Paraguay": "PRY", "Uruguay": "URY"}

MIN_OVERLAP_RESCALE = 3   # mínimo de años con ambas series para calcular ratio

# ──────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ──────────────────────────────────────────────────────────────────────────────
print("Cargando GTMD2 (puede tardar 1-2 min)...")
gtmd2_raw = pd.read_csv(GTMD2_PATH, usecols=GTMD2_COLS, low_memory=False)
gtmd2 = gtmd2_raw[
    gtmd2_raw["iso3code_i"].isin(SA10) &
    gtmd2_raw["iso3code_j"].isin(SA10) &
    (gtmd2_raw["iso3code_i"] != gtmd2_raw["iso3code_j"]) &
    gtmd2_raw["year"].between(YEAR_MIN, YEAR_MAX)
].copy()
del gtmd2_raw
print(f"  GTMD2 filtrado: {len(gtmd2):,} filas, {gtmd2.groupby(['iso3code_i','iso3code_j']).ngroups} corredores")

# Fuentes alternativas para j=ARG
print("Cargando fuentes adicionales para ARG...")
arg_tour_raw = pd.read_csv(ARG_TOUR_PATH)
arg_tour = (
    arg_tour_raw[arg_tour_raw["pais_agrupado"].isin(ARG_TOUR_MAP)]
    .assign(iso3code_i=lambda x: x["pais_agrupado"].map(ARG_TOUR_MAP),
            iso3code_j="ARG",
            alt_value=lambda x: x["Viajes"])
    .rename(columns={"anio": "year"})
    [["iso3code_i", "iso3code_j", "year", "alt_value"]]
    .query(f"{YEAR_MIN} <= year <= {YEAR_MAX}")
)

arg_un = pd.read_csv(ARG_UN_PATH)
arg_un = arg_un.rename(columns={"un_latam_trips": "alt_value"})
arg_un["iso3code_j"] = "ARG"
arg_un = arg_un[arg_un["year"].between(YEAR_MIN, YEAR_MAX)].copy()

# Combinar fuentes ARG: turismo_anio tiene precedencia; UN cubre lo que falta
arg_alt = pd.concat([
    arg_tour.assign(alt_source="turismo_anio_ARG"),
    arg_un[~arg_un["iso3code_i"].isin(list(ARG_TOUR_MAP.values()))]
           .assign(alt_source="tourism_UN_ARG_Latam"),
], ignore_index=True)
# Lookup rápido: (iso3code_i, year) → (alt_value, alt_source)
arg_alt_lookup = arg_alt.set_index(["iso3code_i", "year"])

# ──────────────────────────────────────────────────────────────────────────────
# 2. FUNCIONES DE EVALUACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def best_unwto(row):
    """Devuelve el primer valor UNWTO no-NaN disponible para una fila, o NaN."""
    for col in UNWTO_COLS:
        if pd.notna(row[col]):
            return row[col], col
    return np.nan, None


def evaluate_corridor(sub, iso_i, iso_j):
    """
    sub: DataFrame con las filas del corredor (i, j), todos los años disponibles.
    Devuelve un dict con el diagnóstico y la recomendación.
    """
    sub = sub.set_index("year").reindex(ALL_YEARS)

    gtmd2_vals = sub["gtmd2_vflow_int"]
    n_gtmd2    = gtmd2_vals.notna().sum()
    missing_yrs = gtmd2_vals[gtmd2_vals.isna()].index.tolist()

    if not missing_yrs:
        return {
            "iso3code_i": iso_i, "iso3code_j": iso_j,
            "n_missing": 0, "missing_years": "",
            "tecnica": "—  sin faltantes",
            "fuente_alt": "", "n_overlap_rescale": "",
            "n_interpolable": "", "notas": "",
        }

    # ── UNWTO disponible en los años faltantes ────────────────────────────────
    unwto_available = {}
    for yr in missing_yrs:
        if yr in sub.index:
            row = sub.loc[yr]
            val, col = best_unwto(row)
            unwto_available[yr] = (val, col)
        else:
            unwto_available[yr] = (np.nan, None)

    n_unwto_in_missing = sum(1 for v, _ in unwto_available.values() if pd.notna(v))

    # Overlap: años donde AMBOS gtmd2 y best UNWTO existen (para calcular ratio)
    overlap_yrs = []
    for yr in ALL_YEARS:
        if yr not in sub.index:
            continue
        row = sub.loc[yr]
        if pd.notna(row.get("gtmd2_vflow_int")):
            val, _ = best_unwto(row)
            if pd.notna(val):
                overlap_yrs.append(yr)
    n_overlap = len(overlap_yrs)

    # ── Fuente alternativa especial: ARG ──────────────────────────────────────
    if iso_j == "ARG":
        alt_in_missing = []
        for yr in missing_yrs:
            key = (iso_i, yr)
            if key in arg_alt_lookup.index:
                row_alt = arg_alt_lookup.loc[key]
                # puede haber duplicados si ambas fuentes cubren el año
                val = row_alt["alt_value"].iloc[0] if isinstance(row_alt, pd.DataFrame) else row_alt["alt_value"]
                src = row_alt["alt_source"].iloc[0] if isinstance(row_alt, pd.DataFrame) else row_alt["alt_source"]
                if pd.notna(val):
                    alt_in_missing.append((yr, val, src))
        n_alt_arg = len(alt_in_missing)
        alt_sources_arg = list({s for _, _, s in alt_in_missing})
    else:
        n_alt_arg = 0
        alt_sources_arg = []

    # ── Interpolabilidad ──────────────────────────────────────────────────────
    # Un gap es interpolable si hay al menos un dato ANTES y uno DESPUÉS del gap
    years_with_data = gtmd2_vals.dropna().index.tolist()
    n_interpolable = 0
    if years_with_data:
        first_data, last_data = min(years_with_data), max(years_with_data)
        n_interpolable = sum(
            1 for yr in missing_yrs
            if first_data < yr < last_data
        )

    # ── Decisión de técnica ───────────────────────────────────────────────────
    notas = []

    # Para j=ARG con fuente directa disponible
    if iso_j == "ARG" and n_alt_arg > 0:
        # ¿Hay overlap suficiente con gtmd2 para rescalar?
        # Calcular overlap con la fuente ARG (no UNWTO)
        overlap_arg = []
        for yr in ALL_YEARS:
            key = (iso_i, yr)
            if key in arg_alt_lookup.index and pd.notna(gtmd2_vals.get(yr)):
                row_alt = arg_alt_lookup.loc[key]
                val = row_alt["alt_value"].iloc[0] if isinstance(row_alt, pd.DataFrame) else row_alt["alt_value"]
                if pd.notna(val):
                    overlap_arg.append(yr)
        n_overlap_arg = len(overlap_arg)

        if n_overlap_arg >= MIN_OVERLAP_RESCALE:
            tecnica = "B) Rescalar fuente ARG (ratio corredor-específico)"
            fuente  = " + ".join(alt_sources_arg)
            notas.append(f"{n_overlap_arg} años de overlap para ratio")
        else:
            tecnica = "A) Usar fuente ARG directo (overlap insuficiente para rescalar)"
            fuente  = " + ".join(alt_sources_arg)
            notas.append(f"solo {n_overlap_arg} años de overlap — ratio inestable")

        if n_alt_arg < len(missing_yrs):
            notas.append(f"fuente ARG cubre {n_alt_arg}/{len(missing_yrs)} años faltantes")
            if n_interpolable > 0:
                notas.append(f"años restantes: interpolación ({n_interpolable} interpolables)")

    # Para otros países: UNWTO
    elif n_unwto_in_missing > 0:
        if n_overlap >= MIN_OVERLAP_RESCALE:
            tecnica = "B) Rescalar UNWTO (ratio corredor-específico)"
            fuente  = f"UNWTO — {n_overlap} años overlap"
            notas.append(f"mejor UNWTO disponible: {[c for _, c in unwto_available.values() if c][:3]}")
        else:
            tecnica = "A) Usar UNWTO directo (overlap insuficiente para rescalar)"
            fuente  = "UNWTO (ratio inestable)"
            notas.append(f"solo {n_overlap} años overlap gtmd2∩UNWTO")

        if n_unwto_in_missing < len(missing_yrs):
            gap_sin_unwto = len(missing_yrs) - n_unwto_in_missing
            notas.append(f"UNWTO falta en {gap_sin_unwto} años de los {len(missing_yrs)} faltantes")
            if n_interpolable > 0:
                notas.append(f"años restantes: interpolación ({n_interpolable} interpolables)")

    # Sin fuente alternativa: interpolación o nada
    elif n_interpolable > 0:
        tecnica = "C) Interpolación log-lineal"
        fuente  = "— (solo datos adyacentes)"
        notas.append(f"{n_interpolable}/{len(missing_yrs)} años faltantes son interpolables")
        n_no_interp = len(missing_yrs) - n_interpolable
        if n_no_interp > 0:
            notas.append(f"{n_no_interp} años en extremos — sin imputación")

    else:
        tecnica = "D) Sin imputación posible"
        fuente  = "—"
        notas.append("sin UNWTO, sin fuente alternativa, sin datos adyacentes")

    return {
        "iso3code_i":       iso_i,
        "iso3code_j":       iso_j,
        "n_gtmd2":          int(n_gtmd2),
        "n_missing":        len(missing_yrs),
        "missing_years":    str(missing_yrs),
        "tecnica":          tecnica,
        "fuente_alt":       fuente,
        "n_overlap_rescale": n_overlap if iso_j != "ARG" else (n_overlap_arg if n_alt_arg > 0 else n_overlap),
        "n_unwto_missing":  n_unwto_in_missing,
        "n_interpolable":   n_interpolable,
        "notas":            " | ".join(notas),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. EJECUTAR DIAGNÓSTICO
# ──────────────────────────────────────────────────────────────────────────────
print("\nEvaluando corredores...")
results = []
for (iso_i, iso_j), sub in gtmd2.groupby(["iso3code_i", "iso3code_j"]):
    results.append(evaluate_corridor(sub.copy(), iso_i, iso_j))

# Corredores que están completamente ausentes del GTMD2 (nunca aparecen)
existing_pairs = set(gtmd2.groupby(["iso3code_i", "iso3code_j"]).groups.keys())
for iso_i in SA10:
    for iso_j in SA10:
        if iso_i == iso_j:
            continue
        if (iso_i, iso_j) not in existing_pairs:
            results.append({
                "iso3code_i": iso_i, "iso3code_j": iso_j,
                "n_gtmd2": 0, "n_missing": N_YEARS,
                "missing_years": str(ALL_YEARS),
                "tecnica": "D) Sin imputación posible — CORREDOR AUSENTE EN GTMD2",
                "fuente_alt": "—", "n_overlap_rescale": 0,
                "n_unwto_missing": 0, "n_interpolable": 0,
                "notas": "el corredor no existe en el GTMD2 — no hay UNWTO tampoco",
            })

diag = pd.DataFrame(results).sort_values(["iso3code_j", "iso3code_i"]).reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────────────
# 4. RESUMEN POR PAÍS DESTINO
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESUMEN POR PAÍS DESTINO")
print("=" * 70)

for iso_j in SA10:
    sub_j = diag[diag["iso3code_j"] == iso_j]
    con_faltantes = sub_j[sub_j["n_missing"] > 0]

    print(f"\n[Destino: {iso_j}] {'sin faltantes' if len(con_faltantes) == 0 else f'{len(con_faltantes)} corredores con faltantes'}")
    if len(con_faltantes) == 0:
        print("  Todos los corredores tienen cobertura completa en gtmd2_vflow_int.")
        continue

    for _, row in con_faltantes.iterrows():
        print(f"  {row['iso3code_i']} -> {iso_j}  |  faltantes: {row['n_missing']}/{N_YEARS} anios"
              f"  |  {row['tecnica']}")
        if row["notas"]:
            print(f"      -> {row['notas']}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. TABLA GLOBAL Y EXPORT
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLA GLOBAL — Corredores con datos faltantes")
print("=" * 70)

con_faltantes_all = diag[diag["n_missing"] > 0].copy()
print(f"Total corredores: {len(diag)}  |  Con faltantes: {len(con_faltantes_all)}\n")

cols_display = ["iso3code_i", "iso3code_j", "n_gtmd2", "n_missing",
                "tecnica", "fuente_alt", "n_overlap_rescale",
                "n_unwto_missing", "n_interpolable", "notas"]
print(con_faltantes_all[cols_display].to_string(index=False))

# Conteo por técnica
print("\n" + "-" * 50)
print("Distribucion de tecnicas recomendadas:")
print(diag["tecnica"].value_counts().to_string())

# Exportar
diag.to_csv(OUTPUT_PATH, index=False)
print(f"\nResultados exportados: {OUTPUT_PATH}")
