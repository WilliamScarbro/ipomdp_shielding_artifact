# INSTALL

Two supported paths. Docker is recommended and fully self-contained.

## Option A — Docker (recommended)

```bash
# From the artifact root (the directory containing this file):
docker build -t ipomdp-shielding .

# Regenerate every paper figure/table into the host's ./figures (default CMD):
docker run --rm -v "$PWD/figures:/artifact/figures" ipomdp-shielding

# Fast end-to-end sanity check instead:
docker run --rm ipomdp-shielding ./run_experiments.sh smoke
```

## Option B — Local Python 3.11

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps -e .

./run_experiments.sh figures     # regenerate all figures + tables -> ./figures/
./run_experiments.sh smoke       # fast end-to-end sanity check
```

## Minimal working example (a few seconds)

To confirm the environment is healthy without regenerating everything:

```bash
./run_experiments.sh smoke
```

Expected final line:

```
[smoke] ALL CHECKS PASSED
```

This regenerates the key figures from bundled data and runs a tiny (2-trial)
live Monte Carlo shielding sweep on a bundled controller.

See `README.md` for the full command-to-figure mapping and `REQUIREMENTS.md` for
environment constraints.
