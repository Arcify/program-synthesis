"""Utilities for managing temporary workspaces when executing candidate programs."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Workspace:
    """Creates an isolated directory for program execution."""

    root: Path

    @classmethod
    def create(cls, prefix: str = "synthesis-") -> "Workspace":
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        return cls(root=temp_dir)

    def write_file(self, relative_path: str, content: str) -> Path:
        target = self.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def read_file(self, relative_path: str) -> str:
        target = self.root / relative_path
        return target.read_text(encoding="utf-8")

    def cleanup(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def __enter__(self) -> "Workspace":  # pragma: no cover - trivial context wrapper
        return self

    def __exit__(self, *exc_info) -> Optional[bool]:  # pragma: no cover
        self.cleanup()
        return None
