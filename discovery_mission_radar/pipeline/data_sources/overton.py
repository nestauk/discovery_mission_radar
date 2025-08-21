"""Overton policy data source (Step 1).

Implements UK-focused boolean and semantic passes with optional targeted
source restrictions, plus an optional international strict boolean pass.

This step focuses on search logic, post-filtering, deduplication, facets,
and a minimal `summary.json` output sufficient for smoke testing. The full
output structure will be added in the next step.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import logging
import os

import pandas as pd

try:
    from discovery_utils.getters.overton import OvertonGetter, OvertonAPIError
except Exception:  # pragma: no cover - import guarded for environments without dependency
    OvertonGetter = None  # type: ignore
    OvertonAPIError = Exception  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class OvertonResult:
    """Container for Overton search results for a single topic.

    Attributes:
        uk_df (pd.DataFrame): UK results after post-filters and dedupe.
        intl_df (Optional[pd.DataFrame]): International results if enabled.
        uk_facets (Dict): Facets snapshot for UK query strategies.
        intl_facets (Dict): Facets snapshot for international (may be empty).
        uk_summary (Dict): Summary stats for UK results.
        intl_summary (Dict): Summary stats for international results.
    """

    uk_df: pd.DataFrame
    intl_df: Optional[pd.DataFrame]
    uk_facets: Dict[str, Any]
    intl_facets: Dict[str, Any]
    uk_summary: Dict[str, Any]
    intl_summary: Dict[str, Any]


class OvertonDataSource:
    """Overton policy search data source.

    Step-1 implementation provides:
    - Boolean query builder (AND-of-sets with optional OR-all fallback)
    - Semantic prompt builder
    - UK boolean and semantic passes with fixed filters
    - Optional targeted source pass per mission
    - Optional international strict boolean pass
    - Title/abstract post-filter for boolean strategies
    - Doc-series exclusions and Hansard exclusion
    - Dedupe by Overton `id`
    - Facets and minimal summary authoring
    - Minimal `save_outputs` writing only a summary.json for smoke tests

    The full output artefacts are added in the next step.
    """

    # Fallback threshold for switching to OR-all keywords if AND-across-sets yields very few
    BOOLEAN_FALLBACK_THRESHOLD = 20

    # UK semantic minimum similarity
    UK_MIN_SIMILARITY = 0.40

    # International (strict) semantic default: OFF in step-1
    INTL_SEMANTIC_ENABLED_DEFAULT = False
    INTL_MIN_SIMILARITY = 0.50

    DOC_SERIES_EXCLUDE = {"Transcript", "Press Release"}

    def __init__(
        self,
        core_sources: Optional[List[str]] = None,
        mission_sources: Optional[Dict[str, List[str]]] = None,
        international_enabled_for: Optional[Dict[str, bool]] = None,
    ) -> None:
        """Initialise the data source.

        Args:
            core_sources (Optional[List[str]]): Optional set of source slugs for a UK targeted pass.
            mission_sources (Optional[Dict[str, List[str]]]): Optional per-mission source slugs.
            international_enabled_for (Optional[Dict[str, bool]]): Per-mission flag for international strict pass.
        """
        self.core_sources = core_sources or []
        self.mission_sources = mission_sources or {}
        self.international_enabled_for = international_enabled_for or {}

    def fetch_topic(self, topic_cfg: Dict[str, Any], mission: str, window_months: int) -> OvertonResult:
        """Fetch Overton results for a topic.

    Args:
            topic_cfg (Dict[str, Any]): Topic configuration dictionary.
            mission (str): Mission key, e.g. "AHL" or "ASF".
            window_months (int): Backwards window in months (default 60) for date filter.

    Returns:
            OvertonResult: Result container with data frames, facets and summaries.
        """
        if not os.getenv("OVERTON_API_KEY"):
            logger.warning("OVERTON_API_KEY is not set. Skipping Overton for this run.")
            empty = pd.DataFrame()
            return OvertonResult(
                uk_df=empty,
                intl_df=None,
                uk_facets={},
                intl_facets={},
                uk_summary=self._summarise_df(empty, tag="uk"),
                intl_summary={},
            )

        if OvertonGetter is None:
            logger.warning("discovery_utils OvertonGetter not available. Skipping.")
            empty = pd.DataFrame()
            return OvertonResult(
                uk_df=empty,
                intl_df=None,
                uk_facets={},
                intl_facets={},
                uk_summary=self._summarise_df(empty, tag="uk"),
                intl_summary={},
            )

        getter = OvertonGetter()

        recipe = topic_cfg.get("search_recipe", {}) or {}
        keyword_sets = self._extract_keyword_sets(recipe)
        scope_statements = self._extract_scope_statements(recipe)

        boolean_query_and = self._build_boolean_query(keyword_sets)
        boolean_query_or = self._build_or_all_keywords(keyword_sets)
        semantic_prompt = self._build_semantic_prompt(scope_statements)

        published_after = self._compute_published_after(window_months)

        # --- UK passes ---
        uk_frames: List[pd.DataFrame] = []
        uk_facets: Dict[str, Any] = {}

        # Boolean (AND-of-sets) with fallback OR-all if very few
        primary_boolean_query = boolean_query_and or boolean_query_or
        uk_boolean_df = self._run_uk_boolean_pass(
            getter,
            primary_boolean_query,
            keyword_sets,
            published_after,
        )
        # Fallback to OR-all if low results and OR query exists
        if boolean_query_or and len(uk_boolean_df) < self.BOOLEAN_FALLBACK_THRESHOLD and boolean_query_or != primary_boolean_query:
            uk_boolean_or_df = self._run_uk_boolean_pass(
                getter,
                boolean_query_or,
                keyword_sets,
                published_after,
            )
            if len(uk_boolean_or_df) > len(uk_boolean_df):
                uk_boolean_df = uk_boolean_or_df
        if not uk_boolean_df.empty:
            uk_frames.append(uk_boolean_df)
        # Facets for boolean
        try:
            if primary_boolean_query:
                uk_facets["boolean"] = getter.get_facets(query=primary_boolean_query)
        except OvertonAPIError:
            uk_facets["boolean"] = {}

        # Targeted mission/core sources (boolean)
        targeted_sources = self._get_targeted_sources(mission)
        if targeted_sources and primary_boolean_query:
            targeted_frames: List[pd.DataFrame] = []
            for source_slug in targeted_sources:
                try:
                    part = getter.search_documents(
                        query=primary_boolean_query,
                        semantic_search=False,
                        source=source_slug,
                        source_country="UK",
                        source_type="government",
                        published_after=published_after,
                    )
                except Exception:
                    part = None
                if part is not None and not part.empty:
                    part = self._post_filter_boolean(part, keyword_sets)
                    part = self._exclude_series_and_hansard(part)
                    targeted_frames.append(part)
            if targeted_frames:
                uk_frames.append(pd.concat(targeted_frames, ignore_index=True))

        # Semantic
        if semantic_prompt:
            uk_semantic_df = self._run_uk_semantic_pass(
                getter,
                semantic_prompt,
                published_after,
            )
            if not uk_semantic_df.empty:
                uk_frames.append(uk_semantic_df)
            try:
                uk_facets["semantic"] = getter.get_facets(query=semantic_prompt)
            except OvertonAPIError:
                uk_facets["semantic"] = {}

        uk_df = self._dedupe_concat(uk_frames)
        uk_summary = self._summarise_df(uk_df, tag="uk")

        # --- International strict (boolean only by default) ---
        intl_df: Optional[pd.DataFrame] = None
        intl_facets: Dict[str, Any] = {}
        intl_summary: Dict[str, Any] = {}

        if self._is_international_enabled(mission) and primary_boolean_query:
            intl_frames: List[pd.DataFrame] = []
            for src_type in ["government", "igo"]:
                try:
                    part = getter.search_documents(
                        query=primary_boolean_query,
                        semantic_search=False,
                        source_type=src_type,
                        source_country="OECD members",  # mapped to region by getter
                        published_after=published_after,
                    )
                except Exception:
                    part = None
                if part is not None and not part.empty:
                    part = self._post_filter_boolean(part, keyword_sets)
                    part = self._exclude_series_and_hansard(part)
                    intl_frames.append(part)
            intl_df = self._dedupe_concat(intl_frames)
            # Cross-pass dedupe: remove any documents already included in UK pass
            try:
                if intl_df is not None and not intl_df.empty and uk_df is not None and not uk_df.empty:
                    if "id" in intl_df.columns and "id" in uk_df.columns:
                        uk_ids = set(uk_df["id"].tolist())
                        intl_df = intl_df[~intl_df["id"].isin(uk_ids)]
            except Exception:
                # Be defensive: never fail the run due to cross-dedupe issues
                pass
            intl_summary = self._summarise_df(intl_df, tag="international")
            # Facets (optional in step-1)
            try:
                intl_facets["boolean"] = getter.get_facets(query=primary_boolean_query)
            except OvertonAPIError:
                intl_facets["boolean"] = {}

        return OvertonResult(
            uk_df=uk_df,
            intl_df=intl_df,
            uk_facets=uk_facets,
            intl_facets=intl_facets,
            uk_summary=uk_summary,
            intl_summary=intl_summary,
        )

    def save_outputs(self, topic_cfg: Dict[str, Any], mission: str, result: OvertonResult) -> None:
        """Persist minimal outputs for step-1 smoke tests.

        This writes only `summary.json` files under the future target structure:
        outputs/{MISSION}/{topic-slug}/overton_uk/summary.json and
        outputs/{MISSION}/{topic-slug}/overton_international/summary.json (if present).

    Args:
            topic_cfg (Dict[str, Any]): Topic configuration dictionary.
            mission (str): Mission key.
            result (OvertonResult): Result container.
        """
        topic_slug = self._slugify((topic_cfg.get("search_recipe", {}) or {}).get("category_name", "topic"))
        base = Path("outputs") / mission / topic_slug
        uk_dir = base / "overton_uk"
        uk_dir.mkdir(parents=True, exist_ok=True)
        with (uk_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(result.uk_summary, f, ensure_ascii=False, indent=2)

        if result.intl_df is not None and not result.intl_df.empty and result.intl_summary:
            intl_dir = base / "overton_international"
            intl_dir.mkdir(parents=True, exist_ok=True)
            with (intl_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(result.intl_summary, f, ensure_ascii=False, indent=2)

    # ------------------------ Internals ------------------------
    def _extract_keyword_sets(self, recipe: Dict[str, Any]) -> List[List[str]]:
        sets_cfg = recipe.get("keyword_sets", []) or []
        keyword_sets: List[List[str]] = []
        for ks in sets_cfg:
            kws = ks.get("keywords", []) or []
            clean = [str(k).strip() for k in kws if k and str(k).strip()]
            keyword_sets.append(clean)
        return keyword_sets

    def _extract_scope_statements(self, recipe: Dict[str, Any]) -> List[str]:
        scopes = recipe.get("scope_statements", []) or []
        return [str(s).strip() for s in scopes if s and str(s).strip()]

    def _build_boolean_query(self, keyword_sets: List[List[str]]) -> Optional[str]:
    groups: List[str] = []
    for kws in keyword_sets:
        if not kws:
            continue
        terms = [f'"{k}"' if " " in k else k for k in kws]
        groups.append("( " + " OR ".join(terms) + " )")
    if not groups:
        return None
    return " AND ".join(groups)

    def _build_or_all_keywords(self, keyword_sets: List[List[str]]) -> Optional[str]:
        all_kws = [k for group in keyword_sets for k in group]
        if not all_kws:
            return None
        terms = [f'"{k}"' if " " in k else k for k in all_kws]
        return "( " + " OR ".join(terms) + " )"

    def _build_semantic_prompt(self, scope_statements: List[str]) -> Optional[str]:
        cleaned = [s for s in scope_statements if s]
    if not cleaned:
        return None
    return " ".join(cleaned)

    def _compute_published_after(self, window_months: int) -> str:
        days = int(round(window_months * 30.4167))
    return (date.today() - timedelta(days=days)).isoformat()

    def _post_filter_boolean(self, df: pd.DataFrame, keyword_sets: List[List[str]]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
        # Exclude transcripts/press releases
        df = self._exclude_series_and_hansard(df)
        # Title/abstract contains at least one keyword (case-insensitive)
        all_keywords = [k.strip('"').lower() for group in keyword_sets for k in group]
        if not all_keywords:
            return df

        def _contains_kw(text: Any) -> bool:
            t = str(text or "").lower()
            for kw in all_keywords:
                if kw and kw in t:
                    return True
            return False

        has_title = "title" in df.columns
        has_abs = "abstract" in df.columns
        if has_title or has_abs:
            mask = False
            if has_title:
                mask = df["title"].apply(_contains_kw)
            if has_abs:
                mask = mask | df["abstract"].apply(_contains_kw)
            df = df[mask]
    # Dedupe by id if present
    if "id" in df.columns:
            df = df.drop_duplicates(subset=["id"])  # keep first
        return df

    def _exclude_series_and_hansard(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if "overton_policy_document_series" in df.columns:
            df = df[~df["overton_policy_document_series"].isin(self.DOC_SERIES_EXCLUDE)]
        if "source" in df.columns:
            df = df[df["source"] != "hansard_uk"]
    return df

    def _run_uk_boolean_pass(
        self,
        getter: Any,
        query_text: Optional[str],
        keyword_sets: List[List[str]],
        published_after: str,
    ) -> pd.DataFrame:
        if not query_text:
            return pd.DataFrame()
        try:
            df = getter.search_documents(
                query=query_text,
                semantic_search=False,
                source_country="UK",
                source_type="government",
                published_after=published_after,
            )
        except Exception:
            return pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()
        df = self._post_filter_boolean(df, keyword_sets)
        return df

    def _run_uk_semantic_pass(
        self,
        getter: Any,
        prompt: Optional[str],
        published_after: str,
    ) -> pd.DataFrame:
        if not prompt:
            return pd.DataFrame()
        try:
            df = getter.search_documents(
                query=prompt,
                semantic_search=True,
                source_country="UK",
                source_type="government",
                published_after=published_after,
            )
        except Exception:
            return pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()
        # Exclude series and Hansard, apply similarity threshold if available
        df = self._exclude_series_and_hansard(df)
        if "similarity_score" in df.columns:
            df = df[df["similarity_score"] >= self.UK_MIN_SIMILARITY]
        if "id" in df.columns:
            df = df.drop_duplicates(subset=["id"])  # keep first
        return df

    def _dedupe_concat(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        if "id" in combined.columns:
            combined = combined.drop_duplicates(subset=["id"])  # keep first
        return combined

    def _summarise_df(self, df: pd.DataFrame, tag: str) -> Dict[str, Any]:
        if df is None or df.empty:
            return {"tag": tag, "num_results": 0, "source_type_counts": {}, "source_country_counts": {}, "year_span": {"min": None, "max": None}}
        st_counts = df["source_type"].value_counts(dropna=False).to_dict() if "source_type" in df.columns else {}
        sc_counts = df["source_country"].value_counts(dropna=False).to_dict() if "source_country" in df.columns else {}
        def _to_native(value):
            try:
                import numpy as np  # noqa: F401
                import math
                if hasattr(value, "item"):
                    value = value.item()
                if isinstance(value, float) and (math.isnan(value)):
                    return None
            except Exception:
                pass
            return value
        years_min = None
        years_max = None
        if "publication_year" in df.columns:
            years = pd.to_numeric(df["publication_year"], errors="coerce")
            if not years.dropna().empty:
                years_min = _to_native(years.min())
                years_max = _to_native(years.max())
        st_counts_native = {str(k): _to_native(v) for k, v in st_counts.items()}
        sc_counts_native = {str(k): _to_native(v) for k, v in sc_counts.items()}
        return {
            "tag": tag,
            "num_results": int(len(df)),
            "source_type_counts": st_counts_native,
            "source_country_counts": sc_counts_native,
            "year_span": {"min": years_min, "max": years_max},
        }

    def _get_targeted_sources(self, mission: str) -> List[str]:
        targeted = list(self.core_sources)
        targeted += self.mission_sources.get(mission, [])
        # Preserve order and uniqueness
        seen = set()
        uniq: List[str] = []
        for s in targeted:
            if s and s not in seen:
                seen.add(s)
                uniq.append(s)
        return uniq

    def _is_international_enabled(self, mission: str) -> bool:
        return bool(self.international_enabled_for.get(mission, False))

    def _slugify(self, text: str) -> str:
        import re
        t = (text or "").strip().lower()
        if not t:
            return "topic"
        slug = re.sub(r"[^a-z0-9]+", "-", t)
        slug = re.sub(r"-+", "-", slug).strip("-")
        return slug or "topic"


