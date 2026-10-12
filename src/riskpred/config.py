from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    raw: dict

    def __getitem__(self, k):
        return self.raw[k]

    @property
    def raw_dir(self) -> Path:
        return (ROOT / self.raw["data"]["raw_dir"]).resolve()

    @property
    def out_dir(self) -> Path:
        p = ROOT / self.raw["data"]["out_dir"]
        p.mkdir(parents=True, exist_ok=True)
        return p


def load(path: str | Path | None = None) -> Config:
    p = Path(path) if path else ROOT / "configs" / "default.toml"
    with open(p, "rb") as f:
        return Config(tomllib.load(f))
