from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	from cv_bench.run_all import PROJECT_RUNNERS as PROJECT_RUNNERS
	from cv_bench.run_all import execute as execute

__all__ = ["PROJECT_RUNNERS", "execute"]


def __getattr__(name: str) -> Any:
	if name == "PROJECT_RUNNERS":
		from cv_bench.run_all import PROJECT_RUNNERS

		return PROJECT_RUNNERS
	if name == "execute":
		from cv_bench.run_all import execute

		return execute
	raise AttributeError(f"module 'cv_bench' has no attribute {name!r}")