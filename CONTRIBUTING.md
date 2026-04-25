# Contributing to pysampler

Thanks for your interest in improving `pysampler`! This document describes how
to set up a development environment, run the tests, and submit changes.

## Code of conduct

Be respectful, on-topic, and constructive. Discrimination and personal attacks
are not tolerated in any project space (issues, pull requests, discussions).

## Reporting issues

Before opening a new issue, please search existing issues to avoid duplicates.

A good bug report includes:

- A clear, minimal reproducer (Python snippet or C++ build command).
- The exact error message / stack trace.
- Output of `nvidia-smi`, `nvcc --version`, and `python -c "import torch; print(torch.__version__, torch.version.cuda)"`.
- The CMake cache values you used (e.g. `ENABLE_OPENVKL`, `ENABLE_WITCHER`,
  `CMAKE_CUDA_ARCHITECTURES`).
- The commit SHA of `pysampler` and your OS / GPU model.

Security-sensitive reports should be sent privately rather than filed as
public issues — see `SECURITY.md` (or email the maintainer) if/when present.

## Development setup

### One-shot setup

From the repository root:

```bash
./setup_venv.sh
source .venv/bin/activate
```

This creates a `uv`-managed venv, installs ISPC and a CUDA-matched PyTorch
wheel, and builds `pysampler` in editable mode via `scikit-build-core`.

### Manual setup

If you'd rather wire it up by hand:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
# Make sure `nvcc` and `ispc` are on PATH.
uv pip install -v -e .[test] \
  --config-settings cmake.define.CMAKE_CUDA_ARCHITECTURES=86
```

Toggle backends with extra `--config-settings cmake.define.ENABLE_*=OFF`
(see `pyproject.toml` for the defaults).

### Iterating on C++ / CUDA

`scikit-build-core` reuses `build/` between installs, so iterative rebuilds
are fast:

```bash
uv pip install -v --no-build-isolation -e .
```

If you change CMake-level settings or third-party deps, do a clean build:

```bash
./setup_venv.sh --clean
./setup_venv.sh
```

## Running tests

```bash
pytest -v tests
```

Tests are CUDA-only and are auto-skipped when no GPU is available. New tests
should follow the same pattern (`pytestmark = pytest.mark.skipif(...)` or
per-test `pytest.importorskip` / `pytest.skip(...)` guards) so the suite
remains usable in CPU-only environments.

When adding a new sampler backend, please add at least one test that:

1. Constructs a sampler from a tiny synthetic input (or a generated temp file).
2. Calls `decode()` at known coordinates and checks values within a stated
   tolerance.
3. Cleans up any temp files in a `try/finally`.

## Style

- **Python:** PEP 8, 4-space indent, type hints where it improves clarity.
  Keep imports grouped (stdlib / third-party / local).
- **C++ / CUDA:** C++17, 2-space indent, `lower_snake_case` for free
  functions and locals, `PascalCase` for types, `m_` prefix for non-public
  member variables, header guards via `#pragma once`.
- **CMake:** lowercase commands, 2-space indent, prefer
  `target_link_libraries(... PRIVATE ...)` over global linking.
- **Comments:** explain the *why*, not the *what*. Don't narrate obvious
  code. Cross-reference upstream issues / papers for non-obvious choices.

Avoid renaming or reformatting unrelated code in the same PR as a feature
or bug-fix change — keep diffs focused.

## Commit messages

Use short, imperative subject lines (≤ 72 chars), optionally with a scoped
prefix. Examples:

```
sampler_cuda: clamp coords to [0,1]^3 before tex3D fetch
build: bump VTK-m to 2.3.0
docs: document the (count, n_channels) values layout
```

Squash fix-up commits before pushing the final version.

## Pull requests

1. Fork the repo and create a topic branch off `main`:
   `git checkout -b my-feature`.
2. Make your changes in focused commits.
3. Ensure `pytest -v tests` passes locally on a CUDA host.
4. Open a PR against `main` with:
   - A clear description of *what* and *why*.
   - Notes on backward-incompatible changes, if any.
   - Links to related issues (`Fixes #123`).
5. Be ready to iterate on review feedback. Prefer additional commits over
   force-pushes during review; squash-merge happens at the end.

By opening a pull request, you agree to license your contribution under the
terms of `LICENSE` (Apache-2.0).

## Adding a new sampler backend

1. Add a new `csrc/sampler_<backend>.{h,cpp,cu}` implementing
   `pysampler::SamplerBase` (see `csrc/sampler.h`).
2. Wire it up in `csrc/sampler.cpp::create_sampler(...)` behind an
   `ENABLE_<BACKEND>` flag, mirroring `IFF_CUDA_DEVICE` / `IFF_OPENVKL_DEVICE`.
3. Add a `cmake/dep_<backend>.cmake` if external deps are needed (use
   `FetchContent` for source-built deps).
4. Expose a new `type=` string in `csrc/pysampler.cpp::py_create_sampler`
   with the kwargs your backend expects.
5. Update `pyproject.toml` (`[tool.scikit-build.cmake.define]`) and the
   README's "Per-backend usage" / device-support matrix.
6. Add a test under `tests/`.

## Releasing

(Maintainer-only.) Release flow:

1. Bump `version` in `pyproject.toml` and `CITATION.cff`.
2. Update `CHANGELOG.md` (when present) with the date and notable changes.
3. Tag the release: `git tag -a vX.Y.Z -m "Release vX.Y.Z" && git push --tags`.
4. Build wheels and publish (when CI is wired up).

## Questions?

Open a discussion or a "question" issue. Thank you for contributing!
