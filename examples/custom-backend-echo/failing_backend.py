"""A backend whose ``setup()`` raises — to demonstrate the startup-failure path.

Serving this must make the runtime exit non-zero and ``/healthz`` never reach
``200``. See the README's acceptance-criterion 5.
"""

from __future__ import annotations

from inferencekey.backend import BackendContext, CustomBackend, Job, Result


class FailingSetupBackend(CustomBackend):
    def setup(self, ctx: BackendContext) -> None:
        raise RuntimeError("boom: model failed to load")

    def process(self, job: Job) -> Result:  # pragma: no cover — never reached
        return Result(output={})
