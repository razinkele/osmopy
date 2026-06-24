# Native Predation Kernel Spike

This is a throwaway spike harness to benchmark a pure-Python rewrite of the predation kernel. The spike runs provenance guards first (to verify the correct OSMOSE is imported and the Numba compilation is enabled), then measures kernel performance in isolation. See the spike specification in `.superpowers/sdd/task-*/brief.md` for detailed requirements and each task's requirements.

To run the spike:

```bash
PYTHONPATH=. .venv/bin/python scripts/spikes/native_predation/run_spike.py
```

This harness will be deleted after the spike is complete.
