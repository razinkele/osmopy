---
name: run-tests
description: Run the OSMOSE test suite, optionally targeting a specific file or enabling coverage
disable-model-invocation: true
---

Run the OSMOSE test suite from `/home/razinka/osmose/osmose-python/`.

## Arguments

- `file` (optional): specific test file, e.g., `test_schema.py`
- `coverage` (optional): if "yes", include coverage reporting

## Commands

- **All tests**: `.venv/bin/python -m pytest -v`
- **Single file**: `.venv/bin/python -m pytest tests/{file} -v`
- **With coverage**: append `--cov=osmose --cov-report=term-missing`

## Rules

- Always run from `/home/razinka/osmose/osmose-python/`
- Use `.venv/bin/python`, never system python
- Report a summary of pass/fail counts after the run
