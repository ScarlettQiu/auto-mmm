"""PrismMMM Dataset Discovery.

Profiles a raw dataset and writes:
  - metadata.json   ← rich knowledge file used by all agents
  - config.json     ← updated with correct column names and settings

Usage:
    python discover.py --source csv --path ./data.csv
    python discover.py --source bigquery --query "SELECT * FROM project.dataset.table"
    python discover.py --source gsheet --sheet_id 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ── Source loaders ────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(Path(path).expanduser())


def load_bigquery(query: str, project: str | None = None) -> pd.DataFrame:
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project=project)
        return client.query(query).to_dataframe()
    except ImportError:
        print("ERROR: google-cloud-bigquery not installed. Run: pip install google-cloud-bigquery")
        sys.exit(1)


def load_gsheet(sheet_id: str, sheet_name: str = "Sheet1") -> pd.DataFrame:
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        creds = Credentials.from_service_account_file("service_account.json", scopes=scopes)
        gc = gspread.authorize(creds)
        ws = gc.open_by_key(sheet_id).worksheet(sheet_name)
        return pd.DataFrame(ws.get_all_records())
    except ImportError:
        print("ERROR: gspread not installed. Run: pip install gspread google-auth")
        sys.exit(1)


def load_source(args: argparse.Namespace) -> pd.DataFrame:
    if args.source == "csv":
        print(f"Loading CSV: {args.path}")
        return load_csv(args.path)
    elif args.source == "bigquery":
        print(f"Loading BigQuery: {args.query}")
        return load_bigquery(args.query, args.project)
    elif args.source == "gsheet":
        print(f"Loading Google Sheet: {args.sheet_id}")
        return load_gsheet(args.sheet_id, args.sheet_name)
    else:
        print(f"ERROR: Unknown source '{args.source}'. Use csv, bigquery, or gsheet.")
        sys.exit(1)


# ── Column detection ──────────────────────────────────────────────────────────

DATE_PATTERNS = ["date", "week", "month", "day", "period", "time", "dt"]
KPI_PATTERNS  = ["revenue", "sales", "gmv", "orders", "conversions",
                 "purchases", "income", "transactions", "kpi"]
SPEND_PATTERNS = ["spend", "cost", "budget", "investment", "media",
                  "google", "meta", "facebook", "tiktok", "tv", "radio",
                  "digital", "sem", "paid", "affiliate", "display", "email"]
CONTROL_PATTERNS = ["discount", "promo", "holiday", "season", "price",
                    "nps", "cpi", "temperature", "event", "competitor"]


def _score(col: str, patterns: list[str]) -> int:
    col_l = col.lower()
    return sum(p in col_l for p in patterns)


def detect_date_column(df: pd.DataFrame) -> str | None:
    # First try to parse each column as datetime
    for col in df.columns:
        if _score(col, DATE_PATTERNS) > 0:
            try:
                pd.to_datetime(df[col].dropna().head(5))
                return col
            except Exception:
                continue
    # Fallback: any column that parses as date
    for col in df.columns:
        try:
            sample = df[col].dropna().head(10)
            parsed = pd.to_datetime(sample, infer_datetime_format=True)
            if parsed.notna().all():
                return col
        except Exception:
            continue
    return None


def detect_date_format(series: pd.Series) -> str:
    sample = str(series.dropna().iloc[0])
    formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
        "%b %Y", "%B %Y", "%Y-%m", "%d-%m-%Y",
    ]
    for fmt in formats:
        try:
            pd.to_datetime(sample, format=fmt)
            return fmt
        except Exception:
            continue
    return "%Y-%m-%d"


def detect_date_frequency(dates: pd.Series) -> str:
    dates = pd.to_datetime(dates).sort_values().dropna()
    if len(dates) < 2:
        return "unknown"
    diffs = dates.diff().dropna().dt.days
    median_diff = diffs.median()
    if median_diff <= 1:   return "daily"
    if median_diff <= 8:   return "weekly"
    if median_diff <= 16:  return "biweekly"
    if median_diff <= 35:  return "monthly"
    if median_diff <= 100: return "quarterly"
    return "annual"


def detect_kpi(df: pd.DataFrame, date_col: str) -> str | None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scored = sorted(numeric_cols, key=lambda c: _score(c, KPI_PATTERNS), reverse=True)
    for col in scored:
        if col == date_col:
            continue
        if _score(col, KPI_PATTERNS) > 0:
            return col
    # Fallback: highest-variance numeric column
    if numeric_cols:
        return max(numeric_cols, key=lambda c: df[c].std() if c != date_col else 0)
    return None


def detect_channels(df: pd.DataFrame, date_col: str, kpi_col: str) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {date_col, kpi_col}
    channels = []
    for col in numeric_cols:
        if col in exclude:
            continue
        if _score(col, SPEND_PATTERNS) > 0:
            channels.append(col)
    return channels


def detect_controls(df: pd.DataFrame, date_col: str, kpi_col: str,
                    channels: list[str]) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {date_col, kpi_col} | set(channels)
    controls = []
    for col in numeric_cols:
        if col in exclude:
            continue
        if _score(col, CONTROL_PATTERNS) > 0:
            controls.append(col)
    return controls


# ── Anomaly detection ─────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame, kpi_col: str,
                     date_col: str | None) -> list[dict]:
    anomalies = []
    kpi = df[kpi_col].dropna()
    mean, std = kpi.mean(), kpi.std()
    z_scores = ((kpi - mean) / (std + 1e-8)).abs()

    for idx in z_scores[z_scores > 3].index:
        val = float(df.loc[idx, kpi_col])
        date = str(df.loc[idx, date_col]) if date_col and date_col in df.columns else str(idx)
        anomalies.append({
            "date": date,
            "kpi_value": round(val, 2),
            "z_score": round(float(z_scores[idx]), 2),
            "note": f"KPI = {val:,.0f} vs mean {mean:,.0f} (z={z_scores[idx]:.1f}σ) — investigate"
        })
    return anomalies


# ── Correlation analysis ──────────────────────────────────────────────────────

def channel_correlations(df: pd.DataFrame, kpi_col: str,
                         channels: list[str]) -> dict:
    result = {}
    for ch in channels:
        if ch in df.columns and kpi_col in df.columns:
            try:
                corr = float(df[ch].corr(df[kpi_col]))
                result[ch] = round(corr, 3)
            except Exception:
                result[ch] = None
    return dict(sorted(result.items(), key=lambda x: abs(x[1] or 0), reverse=True))


# ── Multi-entity detection ────────────────────────────────────────────────────

ENTITY_PATTERNS = ["brand", "region", "country", "market", "segment",
                   "channel_name", "geo", "area", "territory", "store"]

def detect_entity_columns(df: pd.DataFrame, date_col: str, kpi_col: str,
                          channels: list[str]) -> list[str]:
    """Detect brand/region/country grouping columns (string/categorical only)."""
    exclude = {date_col, kpi_col} | set(channels)
    entity_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        # Only string/object columns with low cardinality, or columns matching entity names
        is_string = df[col].dtype == object
        matches_pattern = _score(col, ENTITY_PATTERNS) > 0
        if is_string and df[col].nunique() < max(20, len(df) * 0.5):
            entity_cols.append(col)
        elif matches_pattern:
            entity_cols.append(col)
    return entity_cols


# ── Main discovery ────────────────────────────────────────────────────────────

def discover(df: pd.DataFrame, source_info: dict) -> tuple[dict, dict]:
    print("\nProfiling dataset...")

    # Column detection
    date_col   = detect_date_column(df)
    date_fmt   = detect_date_format(df[date_col]) if date_col else "%Y-%m-%d"
    kpi_col    = detect_kpi(df, date_col or "")
    channels   = detect_channels(df, date_col or "", kpi_col or "")
    controls   = detect_controls(df, date_col or "", kpi_col or "", channels)
    entities   = detect_entity_columns(df, date_col or "", kpi_col or "", channels)

    # Date stats
    dates      = pd.to_datetime(df[date_col]) if date_col else pd.Series([])
    freq       = detect_date_frequency(df[date_col]) if date_col else "unknown"
    n_rows     = len(df)
    n_periods  = df[date_col].nunique() if date_col else n_rows

    # KPI stats
    kpi_stats = {}
    if kpi_col and kpi_col in df.columns:
        kpi = df[kpi_col].dropna()
        kpi_stats = {
            "mean":   round(float(kpi.mean()), 2),
            "median": round(float(kpi.median()), 2),
            "std":    round(float(kpi.std()), 2),
            "min":    round(float(kpi.min()), 2),
            "max":    round(float(kpi.max()), 2),
            "zeros":  int((kpi == 0).sum()),
            "nulls":  int(kpi.isna().sum()),
        }

    # Channel stats
    channel_stats = {}
    for ch in channels:
        col = df[ch].dropna()
        channel_stats[ch] = {
            "mean":       round(float(col.mean()), 2),
            "max":        round(float(col.max()), 2),
            "zero_pct":   round(float((col == 0).mean() * 100), 1),
            "null_pct":   round(float(df[ch].isna().mean() * 100), 1),
            "corr_to_kpi": channel_correlations(df, kpi_col, [ch]).get(ch),
        }

    # Anomalies
    anomalies = detect_anomalies(df, kpi_col, date_col) if kpi_col else []

    # Multi-entity info
    entity_info = {}
    for col in entities:
        entity_info[col] = {
            "unique_values": int(df[col].nunique()),
            "sample": df[col].dropna().unique()[:5].tolist(),
        }

    # Readiness assessment
    warnings = []
    if n_periods < 30:
        warnings.append(f"Only {n_periods} time periods — MMM needs 50+ for reliable estimates")
    if len(channels) == 0:
        warnings.append("No media channels detected — check column names match spend patterns")
    if len(channels) > n_periods:
        warnings.append(f"{len(channels)} channels > {n_periods} periods — underdetermined system")
    if anomalies:
        warnings.append(f"{len(anomalies)} KPI anomaly(ies) detected — review before modelling")
    for ch, stats in channel_stats.items():
        if stats["zero_pct"] > 50:
            warnings.append(f"{ch}: {stats['zero_pct']}% zeros — sparse channel, may not be identifiable")

    # ── metadata.json ──────────────────────────────────────────────────────────
    metadata = {
        "_generated": datetime.now().isoformat(),
        "_source": source_info,
        "dataset": {
            "n_rows":       n_rows,
            "n_periods":    n_periods,
            "date_range":   [str(dates.min().date()), str(dates.max().date())] if len(dates) else [],
            "date_frequency": freq,
            "entities":     entity_info,
        },
        "columns": {
            "date":     date_col,
            "kpi":      kpi_col,
            "channels": channels,
            "controls": controls,
            "all":      list(df.columns),
        },
        "kpi_stats":     kpi_stats,
        "channel_stats": channel_stats,
        "anomalies":     anomalies,
        "warnings":      warnings,
        "channel_notes": {ch: "" for ch in channels},  # fill in manually
    }

    # ── config.json ────────────────────────────────────────────────────────────
    holdout = 4 if freq == "weekly" else 2
    config = {
        **source_info,
        "kpi_column":       kpi_col,
        "date_column":      date_col,
        "date_format":      date_fmt,
        "media_channels":   channels,
        "control_variables": controls,
        "adstock_max_lag":  2 if freq == "weekly" else 1,
        "hill_slope":       1.5,
        "hill_ec":          0.5,
        "n_bootstrap":      200,
        "pymc_samples":     200,
        "pymc_tune":        100,
        "holdout_periods":  holdout,
        "results_dir":      "./results",
        "rounds_dir":       "./rounds",
    }

    return metadata, config


def print_summary(metadata: dict) -> None:
    d = metadata["dataset"]
    c = metadata["columns"]
    print(f"\n{'='*60}")
    print(f"DATASET PROFILE")
    print(f"{'='*60}")
    print(f"  Rows:        {d['n_rows']:,}")
    print(f"  Periods:     {d['n_periods']} ({d['date_frequency']})")
    print(f"  Date range:  {' → '.join(d['date_range']) if d['date_range'] else 'n/a'}")
    print(f"  KPI:         {c['kpi']}")
    print(f"  Channels:    {len(c['channels'])} → {', '.join(c['channels'])}")
    print(f"  Controls:    {len(c['controls'])} → {', '.join(c['controls']) or 'none detected'}")
    if d["entities"]:
        for col, info in d["entities"].items():
            print(f"  Entity '{col}': {info['unique_values']} unique values — consider filtering to one brand/region")
    if metadata["anomalies"]:
        print(f"\n  ANOMALIES ({len(metadata['anomalies'])}):")
        for a in metadata["anomalies"]:
            print(f"    {a['date']}: {a['note']}")
    if metadata["warnings"]:
        print(f"\n  WARNINGS:")
        for w in metadata["warnings"]:
            print(f"    ⚠  {w}")
    print(f"{'='*60}")


# ── Notion knowledge fetch ────────────────────────────────────────────────────

NOTION_FIELD_DB   = "3382169d-cf08-8134-855b-c04d754c733e"
NOTION_BIZ_DB     = "3382169d-cf08-8132-bbbe-cc73c0257ef2"
NOTION_ISSUES_DB  = "3382169d-cf08-8130-907d-d6bc83e98598"


def _notion_query(database_id: str, token: str) -> list[dict]:
    import urllib.request, urllib.error
    req = urllib.request.Request(
        f"https://api.notion.com/v1/databases/{database_id}/query",
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        },
        data=b"{}",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["results"]


def _text(prop: dict) -> str:
    """Extract plain text from a Notion rich_text or title property."""
    for key in ("rich_text", "title"):
        if key in prop:
            return "".join(t.get("plain_text", "") for t in prop[key])
    return ""


def fetch_notion_knowledge(token: str) -> dict:
    """Pull all three Notion databases and return a knowledge dict."""
    print("Fetching knowledge layer from Notion...")

    # Field Definitions
    fields = {}
    for page in _notion_query(NOTION_FIELD_DB, token):
        p = page["properties"]
        field_name = _text(p.get("field", {}))
        if not field_name:
            continue
        fields[field_name] = {
            "label":            _text(p.get("label", {})),
            "type":             p.get("type",  {}).get("select",  {}).get("name", ""),
            "unit":             _text(p.get("unit", {})),
            "description":      _text(p.get("description", {})),
            "expected_roi_min": p.get("expected_roi_min", {}).get("number"),
            "expected_roi_max": p.get("expected_roi_max", {}).get("number"),
            "notes":            _text(p.get("notes", {})),
        }

    # Business Context
    biz = {}
    for page in _notion_query(NOTION_BIZ_DB, token):
        p = page["properties"]
        key = _text(p.get("key", {}))
        if key:
            biz[key] = {
                "value": _text(p.get("value", {})),
                "notes": _text(p.get("notes", {})),
            }

    # Known Issues
    issues = []
    for page in _notion_query(NOTION_ISSUES_DB, token):
        p = page["properties"]
        issue = _text(p.get("issue", {}))
        if issue:
            issues.append({
                "issue":      issue,
                "affects":    _text(p.get("affects", {})),
                "date_range": _text(p.get("date_range", {})),
                "severity":   p.get("severity", {}).get("select", {}).get("name", ""),
                "action":     _text(p.get("action", {})),
            })

    print(f"  Loaded {len(fields)} field definitions, {len(biz)} business context rows, {len(issues)} known issues")
    return {"fields": fields, "business_context": biz, "known_issues": issues}


def merge_notion_into_metadata(metadata: dict, notion: dict) -> dict:
    """Enrich metadata with Notion knowledge layer."""
    # Override channel_notes with Notion descriptions
    for col, info in notion["fields"].items():
        if col in metadata.get("columns", {}).get("channels", []):
            metadata.setdefault("channel_notes", {})[col] = info.get("description", "")
        # Inject expected ROI ranges for Critic/Analyst
        metadata.setdefault("expected_roi", {})[col] = {
            "min": info.get("expected_roi_min"),
            "max": info.get("expected_roi_max"),
            "label": info.get("label", col),
            "unit":  info.get("unit", ""),
        }

    # Merge business context
    metadata["business_context"] = {k: v["value"] for k, v in notion["business_context"].items()}

    # Merge known issues (union with auto-detected anomalies)
    existing_issues = {a["date"] for a in metadata.get("anomalies", [])}
    for issue in notion["known_issues"]:
        if issue["issue"] not in existing_issues:
            metadata.setdefault("known_issues", []).append(issue)

    metadata["_notion_synced"] = datetime.now().isoformat()
    return metadata


def main():
    parser = argparse.ArgumentParser(description="PrismMMM dataset discovery")
    parser.add_argument("--source", choices=["csv", "bigquery", "gsheet"], default="csv")
    parser.add_argument("--path",       help="CSV file path")
    parser.add_argument("--query",      help="BigQuery SQL query")
    parser.add_argument("--project",    help="GCP project ID (BigQuery)")
    parser.add_argument("--sheet_id",   help="Google Sheet ID")
    parser.add_argument("--sheet_name", default="Sheet1", help="Sheet tab name")
    parser.add_argument("--out-config",   default="config.json",   help="Output config path")
    parser.add_argument("--out-metadata", default="metadata.json", help="Output metadata path")
    parser.add_argument("--no-overwrite-config", action="store_true",
                        help="Skip writing config.json if it already exists")
    parser.add_argument("--notion-token", default=None,
                        help="Notion integration token — fetches knowledge layer from Notion")
    args = parser.parse_args()

    # Auto-load token from .env if not passed explicitly
    if not args.notion_token:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("NOTION_TOKEN=") and not line.startswith("#"):
                    args.notion_token = line.split("=", 1)[1].strip()
                    break

    df = load_source(args)

    source_info: dict = {"source": args.source}
    if args.source == "csv":
        source_info["data_path"] = args.path
    elif args.source == "bigquery":
        source_info["bigquery_query"] = args.query
        source_info["bigquery_project"] = args.project
    elif args.source == "gsheet":
        source_info["gsheet_id"] = args.sheet_id
        source_info["gsheet_sheet"] = args.sheet_name

    metadata, config = discover(df, source_info)

    # Enrich with Notion knowledge layer if token provided
    if args.notion_token:
        notion = fetch_notion_knowledge(args.notion_token)
        metadata = merge_notion_into_metadata(metadata, notion)

    print_summary(metadata)

    # Write metadata
    Path(args.out_metadata).write_text(json.dumps(metadata, indent=2, default=str))
    print(f"\nWrote {args.out_metadata}")

    # Write config (unless --no-overwrite-config)
    cfg_path = Path(args.out_config)
    if args.no_overwrite_config and cfg_path.exists():
        print(f"Skipped {args.out_config} (already exists, --no-overwrite-config set)")
    else:
        cfg_path.write_text(json.dumps(config, indent=2))
        print(f"Wrote {args.out_config}")

    print("\nNext step: review config.json, then run the agent loop:")
    print("  Read program.md and run the loop.")


if __name__ == "__main__":
    main()
