"""Generate OSMOSE-native config files from a flat parameter dict."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


class OsmoseConfigWriter:
    """Write a flat parameter dict to OSMOSE-compatible config files.

    Parameters are categorized by key prefix and routed to the appropriate
    sub-file. The master file (osm_all-parameters.csv) includes
    ``osmose.configuration.*`` references to each sub-file that was created.
    """

    # Each entry: (tuple_of_prefixes, sub_filename, config_key_suffix).
    # Order matters: more-specific prefixes must appear before their
    # less-specific parents (e.g. "species.bioen." before "species.").
    ROUTING: list[tuple[tuple[str, ...], str, str]] = [
        (
            ("temperature.", "species.bioen.", "species.beta."),
            "osm_param-bioenergetics.csv",
            "bioenergetics",
        ),
        (
            ("species.", "growth.", "population.", "reproduction."),
            "osm_param-species.csv",
            "species",
        ),
        (("grid.",), "osm_param-grid.csv", "grid"),
        (("predation.",), "osm_param-predation.csv", "predation"),
        (
            ("mortality.fishing", "fisheries.", "mpa."),
            "osm_param-fishing.csv",
            "fishing",
        ),
        (("movement.",), "osm_param-movement.csv", "movement"),
        (("ltl.",), "osm_param-ltl.csv", "ltl"),
        (("output.",), "osm_param-output.csv", "output"),
        (("economy.", "economic."), "osm_param-economics.csv", "economics"),
    ]

    # Prefixes that explicitly belong in the master file.
    MASTER_PREFIXES: tuple[str, ...] = (
        "simulation.",
        "mortality.subdt",
        "mortality.additional",
        "mortality.starvation",
        "stochastic.",
    )

    def write(self, config: dict[str, Any], output_dir: Path) -> None:
        """Write *config* to OSMOSE files under *output_dir*.

        Parameters
        ----------
        config:
            Flat mapping of OSMOSE parameter keys to their values.
        output_dir:
            Directory in which the config files will be created.
            Created (with parents) if it does not already exist.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        buckets = self._route_params(config)

        # Write sub-files and collect references for the master file.
        references: dict[str, str] = {}
        for prefixes, filename, suffix in self.ROUTING:
            params = buckets.get(suffix, {})
            if not params:
                continue
            self._write_file(output_dir / filename, params)
            references[f"osmose.configuration.{suffix}"] = filename

        # Write master file (master params + references to sub-files).
        master_params: dict[str, str] = dict(buckets.get("master", {}))
        master_params.update(references)
        self._write_file(output_dir / "osm_all-parameters.csv", master_params)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _route_params(self, config: dict[str, Any]) -> dict[str, dict[str, str]]:
        """Categorise each key in *config* into the correct bucket."""
        buckets: dict[str, dict[str, str]] = {}

        for key, value in config.items():
            bucket = self._classify(key)
            buckets.setdefault(bucket, {})[key] = str(value)

        return buckets

    def _classify(self, key: str) -> str:
        """Return the bucket name for *key*."""
        # Check explicit master prefixes first.
        for prefix in self.MASTER_PREFIXES:
            if key.startswith(prefix):
                return "master"

        # Check sub-file routing (order-dependent).
        for prefixes, _filename, suffix in self.ROUTING:
            for prefix in prefixes:
                if key.startswith(prefix):
                    return suffix

        # Fallback: master file.
        return "master"

    @staticmethod
    def _write_file(filepath: Path, params: dict[str, str]) -> None:
        """Write *params* to *filepath* as sorted ``key ; value`` lines.

        Uses a temp-file + os.replace() pattern so the target file is
        never left in a partially-written state if the process is
        interrupted mid-write.
        """
        content = "".join(f"{k} ; {v}\n" for k, v in sorted(params.items()))
        tmp_fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                f.write(content)
            os.replace(tmp_path, filepath)
        except Exception:
            os.unlink(tmp_path)
            raise
