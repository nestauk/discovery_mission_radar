"""Overton Analysis Module.

Provides output writing for Overton results, separating concerns from the
getter/data source. Mirrors the folder structure and artefacts expected
by the pipeline: UK and International outputs side-by-side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json
import pandas as pd
from discovery_utils.utils import charts
import altair as alt
from .base import BaseAnalysisModule

try:  # Optional typing/import safety
    from discovery_utils.getters.overton import OvertonGetter  # type: ignore
except Exception:  # pragma: no cover
    OvertonGetter = object  # type: ignore


class OvertonAnalysisModule(BaseAnalysisModule[OvertonGetter]):
    """Overton analysis/output writer, aligned with BaseAnalysisModule.

    Overton fetching, filtering and selection happen in the data source; this
    module focuses on formatting outputs (CSV/JSON) and simple charts. The
    BaseAnalysisModule abstract methods are implemented minimally to satisfy
    the interface, but the pipeline calls `write_outputs` directly.
    """

    def __init__(self, mission: str):
        super().__init__("overton", mission)

    def _create_default_getter(self) -> OvertonGetter:  # type: ignore[override]
        try:
            from discovery_utils.getters.overton import OvertonGetter as _OG  # type: ignore
            return _OG()
        except Exception:
            return OvertonGetter()  # type: ignore

    def _process_topic_data(self, topic_data: Dict[str, Any], getter: OvertonGetter) -> Dict[str, pd.DataFrame]:  # type: ignore[override]
        # Not used in current flow; return empty placeholders.
        return {
            'ts_yearly': pd.DataFrame(),
            'ts_quarterly': pd.DataFrame(),
        }

    def _generate_custom_stats(self, analysis_results: Dict[str, pd.DataFrame], topic_data: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        return {}

    def _create_source_charts(self, analysis_results: Dict[str, pd.DataFrame], charts_dir: Path, category_name: str, scale_factor: int):  # type: ignore[override]
        return []

    def write_outputs(self, topic_cfg: Dict[str, Any], mission: str, uk_df: pd.DataFrame,
                      intl_df: Optional[pd.DataFrame], uk_facets: Dict[str, Any],
                      intl_facets: Dict[str, Any], uk_summary: Dict[str, Any],
                      intl_summary: Dict[str, Any]) -> None:
        category_name = (topic_cfg.get("search_recipe", {}) or {}).get("category_name", "topic")
        topic_slug = self._slugify(category_name)
        base = Path("outputs") / mission / topic_slug

        uk_dir = base / "overton_uk"
        uk_dir.mkdir(parents=True, exist_ok=True)
        self._write_outputs_for(uk_dir, uk_df, uk_facets, uk_summary, category_name)

        if intl_df is not None and not intl_df.empty:
            intl_dir = base / "overton_international"
            intl_dir.mkdir(parents=True, exist_ok=True)
            self._write_outputs_for(intl_dir, intl_df, intl_facets, intl_summary, category_name)

    def _slugify(self, text: str) -> str:
        import re
        t = (text or "").strip().lower()
        if not t:
            return "topic"
        slug = re.sub(r"[^a-z0-9]+", "_", t)
        slug = re.sub(r"_+", "_", slug).strip("_")
        return slug or "topic"

    def _write_outputs_for(self, outdir: Path, df: pd.DataFrame, facets: Dict[str, Any], summary: Dict[str, Any], category_name: str) -> None:
        outdir.mkdir(parents=True, exist_ok=True)

        # selected_documents.csv
        cols = [
            "id",
            "title",
            "abstract",
            "publication_year",
            "venue",
            "source_country",
            "source_type",
            "overton_policy_document_series",
            "overton_url",
            "similarity_score",
        ]
        if df is not None and not df.empty:
            available_cols = [c for c in cols if c in df.columns]
            if available_cols:
                df[available_cols].to_csv(outdir / "selected_documents.csv", index=False)
            else:
                pd.DataFrame(columns=[c for c in cols if c != "similarity_score"]).to_csv(outdir / "selected_documents.csv", index=False)
        else:
            pd.DataFrame(columns=[c for c in cols if c != "similarity_score"]).to_csv(outdir / "selected_documents.csv", index=False)

        # facets.json
        try:
            with (outdir / "facets.json").open("w", encoding="utf-8") as f:
                json.dump(facets or {}, f, ensure_ascii=False, indent=2)
        except Exception:
            with (outdir / "facets.json").open("w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)

        # summary.json
        with (outdir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary or {}, f, ensure_ascii=False, indent=2)

        # Time series and charts
        yearly_df, quarterly_df = self._write_timeseries(df, outdir)
        self._write_charts(yearly_df, quarterly_df, outdir, category_name)

    def _write_timeseries(self, df: pd.DataFrame, outdir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        yearly = pd.DataFrame(columns=["year", "count"])  # default empty
        quarterly = pd.DataFrame(columns=["quarter", "count"])  # default empty
        # Yearly
        if df is not None and not df.empty and "publication_year" in df.columns:
            yearly = (
                pd.DataFrame({"year": pd.to_numeric(df["publication_year"], errors="coerce")})
                .dropna()
                .astype({"year": "Int64"})
                .groupby("year").size().reset_index(name="count")
                .sort_values("year")
            )
            yearly.to_csv(outdir / "ts_yearly.csv", index=False)
        else:
            pd.DataFrame(columns=["year", "count"]).to_csv(outdir / "ts_yearly.csv", index=False)

        # Quarterly
        date_col = None
        if df is not None and not df.empty:
            for c in ["published_on", "publication_date", "published_date", "added_on", "date"]:
                if c in df.columns:
                    date_col = c
                    break
        if date_col is not None:
            dt = pd.to_datetime(df[date_col], errors="coerce")
            qdf = (
                pd.DataFrame({"year": dt.dt.year, "q": dt.dt.quarter})
                .dropna()
                .astype({"year": "Int64", "q": "Int64"})
            )
            qdf["quarter"] = qdf.apply(lambda r: f"{int(r['year'])}-Q{int(r['q'])}", axis=1)
            quarterly = (
                qdf.groupby(["quarter"]).size().reset_index(name="count").sort_values(["quarter"])
            )
            quarterly.to_csv(outdir / "ts_quarterly.csv", index=False)
        else:
            pd.DataFrame(columns=["quarter", "count"]).to_csv(outdir / "ts_quarterly.csv", index=False)
        return yearly, quarterly

    def _write_charts(self, yearly_df: pd.DataFrame, quarterly_df: pd.DataFrame, outdir: Path, category_name: str) -> None:
        # Match BaseAnalysisModule behaviour: temporarily set default theme
        current_theme = alt.themes.active
        alt.themes.enable('default')
        try:
            scale_factor = 2
            # Yearly docs chart
            if isinstance(yearly_df, pd.DataFrame) and not yearly_df.empty:
                fig = charts.ts_bar(yearly_df, variable="count", variable_title="Number of policy documents")
                fig = charts.configure_plots(fig, chart_title=f"Policy documents per year ({category_name})")
                fig.save(str(outdir / "ts_yearly.png"), scale_factor=scale_factor)
            # Quarterly docs chart
            if isinstance(quarterly_df, pd.DataFrame) and not quarterly_df.empty:
                fig = charts.ts_bar(quarterly_df, variable="count", variable_title="Number of policy documents", time_column="quarter")
                fig = charts.configure_plots(fig, chart_title=f"Policy documents per quarter ({category_name})")
                fig.save(str(outdir / "ts_quarterly.png"), scale_factor=scale_factor)
        except Exception:
            # Never fail pipeline due to chart errors
            pass
        finally:
            # Restore original theme
            alt.themes.enable(current_theme)


