"""Catalog loader for RUNE — supports CSV (default) and YAML (extended).

Loading priority
----------------
1. ``scopes.yaml`` in *catalog_dir*  →  full YAML load; merges ``chains.yaml``
   from the same directory if present.
2. ``scopes.csv`` / ``chains.csv`` in *catalog_dir*  →  CSV load; merges
   ``chains.yaml`` if present.
3. Bundled defaults (``rune_bench/catalog/defaults/``)  →  same resolution as
   above, using the CSV files shipped with the package.

CSV behaviour
-------------
* ``chains.csv`` is preferred over ``scopes.csv`` because it is the enriched
  superset: it adds the ``Ollama Model (2026)`` column and carries role-specific
  questions for chain-mode scopes (e.g. Legal/Ops).
* ``scopes.csv`` is accepted as a fallback when ``chains.csv`` is absent; in
  that case models default to ``qwen3:14b-instruct``.
* All scopes loaded from CSV alone default to ``mode="atomic"``.

YAML behaviour
--------------
* ``chains.yaml`` overlays chain topology on top of a CSV-loaded catalog:
  matching scopes are upgraded to ``mode="chain"`` and gain a ``ChainSpec``.
* ``scopes.yaml`` replaces the CSV entirely (full YAML catalog, no CSV read).
* PyYAML (``pyyaml``) is required only when a YAML file is actually loaded;
  a clear ``ImportError`` is raised if it is missing.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .models import AgentSpec, Catalog, ChainSpec, ChainStep, QuestionSpec, ScopeSpec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULTS_DIR = Path(__file__).parent / "defaults"
_DEFAULT_MODEL = "qwen3:14b-instruct"

# CSV column names (handles both scopes.csv and chains.csv)
_COL_SCOPE = "Scope"
_COL_RANK = "Rank"
_COL_AGENT = "Agent Name"
_COL_RATING = "Rating"
_COL_CAPABILITY = "Agentic Capability"
_COL_Q1 = "Q1 (Technical/Standard)"
_COL_Q1_ACTION = "Q1 Agentic Action"
_COL_Q2 = "Q2 (Investigation/Deep)"
_COL_Q2_ACTION = "Q2 Agentic Action"
_COL_Q3 = "Q3 (Optimization/Logic)"
_COL_Q3_ACTION = "Q3 Agentic Action"
_COL_GITHUB = "GitHub/Docs Link"
_COL_ECOSYSTEM = "Ecosystem"
_COL_MODEL = "Ollama Model (2026)"  # present in chains.csv only


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows, skipping blank rows."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if any(v.strip() for v in row.values())]


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return default


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value.strip()))
    except (ValueError, AttributeError):
        return default


def _build_scope_from_rows(
    scope_name: str,
    rows: list[dict[str, str]],
    model: str,
) -> ScopeSpec:
    agents: list[AgentSpec] = []
    for row in rows:
        agent = AgentSpec(
            name=row.get(_COL_AGENT, "").strip(),
            rank=_safe_int(row.get(_COL_RANK, "0")),
            rating=_safe_float(row.get(_COL_RATING, "0")),
            capability=row.get(_COL_CAPABILITY, "").strip(),
            questions=[
                QuestionSpec(
                    text=row.get(_COL_Q1, "").strip(),
                    action=row.get(_COL_Q1_ACTION, "").strip(),
                ),
                QuestionSpec(
                    text=row.get(_COL_Q2, "").strip(),
                    action=row.get(_COL_Q2_ACTION, "").strip(),
                ),
                QuestionSpec(
                    text=row.get(_COL_Q3, "").strip(),
                    action=row.get(_COL_Q3_ACTION, "").strip(),
                ),
            ],
            github=row.get(_COL_GITHUB, "").strip(),
            ecosystem=row.get(_COL_ECOSYSTEM, "").strip(),
        )
        agents.append(agent)

    agents.sort(key=lambda a: a.rank)
    return ScopeSpec(name=scope_name, model=model, mode="atomic", agents=agents)


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_from_csv(csv_path: Path | None = None) -> Catalog:
    """Load the agent catalog from a CSV file.

    Accepts either ``chains.csv`` (preferred — includes model column) or
    ``scopes.csv`` (fallback — no model column, uses default model).
    When *csv_path* is ``None`` the bundled ``chains.csv`` is used, falling
    back to the bundled ``scopes.csv``.
    """
    if csv_path is None:
        chains_default = _DEFAULTS_DIR / "chains.csv"
        scopes_default = _DEFAULTS_DIR / "scopes.csv"
        csv_path = chains_default if chains_default.exists() else scopes_default

    rows = _read_csv(csv_path)
    has_model_col = _COL_MODEL in (rows[0].keys() if rows else [])

    # Group rows by scope, collecting per-scope model from first occurrence
    scope_rows: dict[str, list[dict[str, str]]] = {}
    scope_models: dict[str, str] = {}

    for row in rows:
        scope_name = row.get(_COL_SCOPE, "").strip()
        if not scope_name:
            continue
        scope_rows.setdefault(scope_name, []).append(row)
        if has_model_col and scope_name not in scope_models:
            model = row.get(_COL_MODEL, "").strip()
            if model:
                scope_models[scope_name] = model

    scopes = [
        _build_scope_from_rows(
            scope_name=name,
            rows=agent_rows,
            model=scope_models.get(name, _DEFAULT_MODEL),
        )
        for name, agent_rows in scope_rows.items()
    ]

    return Catalog(scopes=scopes)


# ---------------------------------------------------------------------------
# YAML loaders
# ---------------------------------------------------------------------------

def _require_yaml() -> object:
    """Import and return the yaml module, raising a clear error if missing."""
    try:
        import yaml  # type: ignore[import-untyped]
        return yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load YAML catalog files. "
            "Install it with: pip install pyyaml"
        ) from exc


def merge_chains(catalog: Catalog, chains_yaml_path: Path) -> Catalog:
    """Overlay chain topology from a YAML file onto a CSV-loaded Catalog.

    For every chain defined in the YAML the matching scope is upgraded to
    ``mode="chain"`` and a ``ChainSpec`` is attached.  Scopes not mentioned
    in the YAML remain ``mode="atomic"``.

    Raises ``ImportError`` if PyYAML is not installed.
    """
    yaml = _require_yaml()

    with open(chains_yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)  # type: ignore[attr-defined]

    chain_specs: dict[str, ChainSpec] = {}
    for chain_data in data.get("chains", []):
        steps = [
            ChainStep(
                id=step["id"],
                agent=step["agent"],
                role=step.get("role", ""),
                question=step.get("question", ""),
                input_from=step.get("input_from") or None,
            )
            for step in chain_data.get("steps", [])
        ]
        spec = ChainSpec(
            scope=chain_data["scope"],
            name=chain_data.get("name", chain_data["scope"]),
            trigger=chain_data.get("trigger", ""),
            steps=steps,
        )
        chain_specs[spec.scope] = spec

    upgraded: list[ScopeSpec] = []
    for scope in catalog.scopes:
        if scope.name in chain_specs:
            upgraded.append(ScopeSpec(
                name=scope.name,
                model=scope.model,
                mode="chain",
                agents=scope.agents,
                chain=chain_specs[scope.name],
            ))
        else:
            upgraded.append(scope)

    return Catalog(scopes=upgraded)


def load_from_yaml(
    scopes_yaml_path: Path,
    chains_yaml_path: Path | None = None,
) -> Catalog:
    """Load the full catalog from a YAML file (no CSV required).

    If *chains_yaml_path* is provided and exists, chain topology is merged
    automatically.

    Raises ``ImportError`` if PyYAML is not installed.
    """
    yaml = _require_yaml()

    with open(scopes_yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)  # type: ignore[attr-defined]

    scopes: list[ScopeSpec] = []
    for scope_data in data.get("scopes", []):
        agents = [
            AgentSpec(
                name=a["name"],
                rank=a.get("rank", 0),
                rating=float(a.get("rating", 0.0)),
                capability=a.get("capability", ""),
                questions=[
                    QuestionSpec(
                        text=q["text"],
                        action=q.get("action", ""),
                    )
                    for q in a.get("questions", [])
                ],
                github=a.get("github", ""),
                ecosystem=a.get("ecosystem", ""),
            )
            for a in scope_data.get("agents", [])
        ]
        scopes.append(ScopeSpec(
            name=scope_data["name"],
            model=scope_data.get("model", _DEFAULT_MODEL),
            mode=scope_data.get("mode", "atomic"),
            agents=agents,
        ))

    catalog = Catalog(scopes=scopes)
    if chains_yaml_path and chains_yaml_path.exists():
        catalog = merge_chains(catalog, chains_yaml_path)
    return catalog


# ---------------------------------------------------------------------------
# Auto-detecting entry point
# ---------------------------------------------------------------------------

def load(catalog_dir: Path | None = None) -> Catalog:
    """Auto-detect and load the RUNE benchmark catalog.

    Search order (highest to lowest priority)::

        1. <catalog_dir>/scopes.yaml  →  load_from_yaml (+ chains.yaml if present)
        2. <catalog_dir>/scopes.csv   →  load_from_csv  (+ chains.yaml if present)
        3. Bundled defaults           →  same resolution as above

    When *catalog_dir* is ``None`` only the bundled defaults are searched.
    """
    search_dirs: list[Path] = []
    if catalog_dir is not None:
        search_dirs.append(Path(catalog_dir))
    search_dirs.append(_DEFAULTS_DIR)

    for d in search_dirs:
        if not d.is_dir():
            continue

        scopes_yaml = d / "scopes.yaml"
        chains_yaml = d / "chains.yaml"
        scopes_csv = d / "scopes.csv"
        chains_csv = d / "chains.csv"

        if scopes_yaml.exists():
            return load_from_yaml(
                scopes_yaml,
                chains_yaml if chains_yaml.exists() else None,
            )

        # Prefer chains.csv (enriched) over scopes.csv
        primary_csv = chains_csv if chains_csv.exists() else (scopes_csv if scopes_csv.exists() else None)
        if primary_csv is not None:
            catalog = load_from_csv(primary_csv)
            if chains_yaml.exists():
                catalog = merge_chains(catalog, chains_yaml)
            return catalog

    raise FileNotFoundError(
        "No catalog files found. Expected scopes.csv, chains.csv, or scopes.yaml in "
        + (str(catalog_dir) if catalog_dir else f"the bundled defaults ({_DEFAULTS_DIR})")
    )
