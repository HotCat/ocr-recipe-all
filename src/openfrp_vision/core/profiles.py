from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

from openfrp_vision.workflow.model import RecipeGraph


CONFIG_FORMAT = "openfrp-vision/profiles/v1"
DEFAULT_PROFILE_ID = "default"
DEFAULT_PROFILE_NAME = "Default Product"
DEFAULT_LANGUAGE = "en"
DEFAULT_INDICATOR_SETTINGS = {
    "blink_interval_s": 0.2,
    "blink_duration_s": 2.0,
    "geometry": [24, 72, 260, 190],
}
DEFAULT_STATS_SETTINGS = {
    "geometry": [36, 86, 420, 150],
}
DEFAULT_LOG_VIEWER_SETTINGS = {
    "geometry": [48, 86, 820, 560],
}


def default_config_path() -> Path:
    configured = os.environ.get("OPENFRP_PROFILE_CONFIG")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".config" / "openfrp_vision" / "profiles.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _profile_id(name: str, existing: set[str]) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-") or "product"
    candidate = base
    index = 2
    while candidate in existing:
        candidate = f"{base}-{index}"
        index += 1
    return candidate


class ProfileStore:
    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        self.path = path
        self.data = data

    @classmethod
    def load(cls, path: Path | None = None) -> "ProfileStore":
        config_path = path or default_config_path()
        if not config_path.exists():
            return cls(config_path, {"format": CONFIG_FORMAT, "active_profile": DEFAULT_PROFILE_ID, "language": DEFAULT_LANGUAGE, "profiles": {}})

        data = json.loads(config_path.read_text(encoding="utf-8"))
        if data.get("format") != CONFIG_FORMAT:
            raise ValueError(f"Unsupported profile config format: {data.get('format')}")
        data.setdefault("active_profile", DEFAULT_PROFILE_ID)
        data.setdefault("language", DEFAULT_LANGUAGE)
        data.setdefault("profiles", {})
        return cls(config_path, data)

    def ensure_default(self, graph: RecipeGraph) -> None:
        profiles = self.data.setdefault("profiles", {})
        if DEFAULT_PROFILE_ID not in profiles:
            profiles[DEFAULT_PROFILE_ID] = self._profile_payload(DEFAULT_PROFILE_NAME, graph)
            self.data["active_profile"] = DEFAULT_PROFILE_ID
            self.save()

    def profile_ids(self) -> list[str]:
        return list(self.data.get("profiles", {}).keys())

    def profile_name(self, profile_id: str) -> str:
        profile = self.data.get("profiles", {}).get(profile_id, {})
        return str(profile.get("name") or profile_id)

    def active_profile_id(self) -> str:
        active = str(self.data.get("active_profile") or DEFAULT_PROFILE_ID)
        if active in self.data.get("profiles", {}):
            return active
        ids = self.profile_ids()
        return ids[0] if ids else DEFAULT_PROFILE_ID

    def language(self) -> str:
        language = str(self.data.get("language") or DEFAULT_LANGUAGE)
        return language if language in {"en", "zh_CN"} else DEFAULT_LANGUAGE

    def set_language(self, language: str) -> None:
        self.data["language"] = language if language in {"en", "zh_CN"} else DEFAULT_LANGUAGE
        self.save()

    def active_graph_data(self) -> dict[str, Any] | None:
        return self.graph_data(self.active_profile_id())

    def graph_data(self, profile_id: str) -> dict[str, Any] | None:
        profile = self.data.get("profiles", {}).get(profile_id)
        if not isinstance(profile, dict):
            return None
        graph = profile.get("graph")
        return graph if isinstance(graph, dict) else None

    def indicator_settings(self, profile_id: str) -> dict[str, Any]:
        profile = self.data.get("profiles", {}).get(profile_id)
        if not isinstance(profile, dict):
            return dict(DEFAULT_INDICATOR_SETTINGS)
        settings = profile.get("indicator")
        if not isinstance(settings, dict):
            return dict(DEFAULT_INDICATOR_SETTINGS)
        return {**DEFAULT_INDICATOR_SETTINGS, **settings}

    def stats_settings(self, profile_id: str) -> dict[str, Any]:
        profile = self.data.get("profiles", {}).get(profile_id)
        if not isinstance(profile, dict):
            return dict(DEFAULT_STATS_SETTINGS)
        settings = profile.get("stats")
        if not isinstance(settings, dict):
            return dict(DEFAULT_STATS_SETTINGS)
        return {**DEFAULT_STATS_SETTINGS, **settings}

    def log_viewer_settings(self, profile_id: str) -> dict[str, Any]:
        profile = self.data.get("profiles", {}).get(profile_id)
        if not isinstance(profile, dict):
            return dict(DEFAULT_LOG_VIEWER_SETTINGS)
        settings = profile.get("log_viewer")
        if not isinstance(settings, dict):
            return dict(DEFAULT_LOG_VIEWER_SETTINGS)
        return {**DEFAULT_LOG_VIEWER_SETTINGS, **settings}

    def set_active(self, profile_id: str) -> None:
        if profile_id not in self.data.get("profiles", {}):
            raise KeyError(profile_id)
        self.data["active_profile"] = profile_id
        self.save()

    def save_profile(
        self,
        profile_id: str,
        graph: RecipeGraph,
        name: str | None = None,
        indicator: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
        log_viewer: dict[str, Any] | None = None,
    ) -> None:
        profiles = self.data.setdefault("profiles", {})
        current = profiles.get(profile_id, {})
        display_name = name or str(current.get("name") or profile_id)
        indicator_settings = indicator or self.indicator_settings(profile_id)
        stats_settings = stats or self.stats_settings(profile_id)
        log_viewer_settings = log_viewer or self.log_viewer_settings(profile_id)
        profiles[profile_id] = self._profile_payload(display_name, graph, indicator_settings, stats_settings, log_viewer_settings)
        self.data["active_profile"] = profile_id
        self.save()

    def create_profile(
        self,
        name: str,
        graph: RecipeGraph,
        indicator: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
        log_viewer: dict[str, Any] | None = None,
    ) -> str:
        profiles = self.data.setdefault("profiles", {})
        profile_id = _profile_id(name, set(profiles.keys()))
        profiles[profile_id] = self._profile_payload(
            name.strip() or profile_id,
            graph,
            indicator or DEFAULT_INDICATOR_SETTINGS,
            stats or DEFAULT_STATS_SETTINGS,
            log_viewer or DEFAULT_LOG_VIEWER_SETTINGS,
        )
        self.data["active_profile"] = profile_id
        self.save()
        return profile_id

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")

    def _profile_payload(
        self,
        name: str,
        graph: RecipeGraph,
        indicator: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
        log_viewer: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "updated_at": _now_iso(),
            "graph": graph.to_dict(),
            "indicator": {**DEFAULT_INDICATOR_SETTINGS, **(indicator or {})},
            "stats": {**DEFAULT_STATS_SETTINGS, **(stats or {})},
            "log_viewer": {**DEFAULT_LOG_VIEWER_SETTINGS, **(log_viewer or {})},
        }
