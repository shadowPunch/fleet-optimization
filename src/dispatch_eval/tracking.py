"""Experiment tracking via Weights & Biases.

Every calibration, validation and evaluation run goes through `tracked_run`
so its config, seeds, metrics and result tables land in one W&B run. Only
configs and metrics are logged, never raw trip data.

Switches (environment variables):
- `DISPATCH_WANDB=0` disables tracking entirely; `tracked_run` then yields a
  no-op run so the code runs unchanged offline.
- `WANDB_MODE=offline` (standard W&B) records locally for a later
  `wandb sync` — used on Kaggle, so no API key has to live there.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

WANDB_PROJECT = "fleet-dispatch-eval"


def tracking_enabled() -> bool:
    return os.environ.get("DISPATCH_WANDB", "1") != "0"


class _NullRun:
    """Stand-in with the subset of the wandb Run API this project uses."""

    name = "untracked"
    summary: dict[str, Any] = {}

    def log(self, *_args: Any, **_kwargs: Any) -> None: ...

    def log_table(self, *_args: Any, **_kwargs: Any) -> None: ...


class _Run:
    def __init__(self, run: Any) -> None:
        self._run = run
        self.name = run.name
        self.summary = run.summary

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._run.log(metrics, step=step)

    def log_table(self, key: str, rows: list[dict[str, Any]]) -> None:
        """Log a list of homogeneous dicts as a W&B table."""
        if not rows:
            return
        import wandb

        columns = list(rows[0].keys())
        self._run.log({key: wandb.Table(columns=columns, data=[[r[c] for c in columns] for r in rows])})


@contextmanager
def tracked_run(
    name: str, job_type: str, config: dict[str, Any], tags: list[str] | None = None
) -> Iterator[_Run | _NullRun]:
    """Open a W&B run for the duration of the block.

    A block that raises marks the run failed (exit code 1), so broken runs
    are distinguishable from finished ones in the W&B UI.
    """
    if not tracking_enabled():
        yield _NullRun()
        return

    import wandb

    run = wandb.init(
        project=WANDB_PROJECT, name=name, job_type=job_type, config=config, tags=tags or []
    )
    try:
        yield _Run(run)
    except BaseException:
        run.finish(exit_code=1)
        raise
    run.finish()
