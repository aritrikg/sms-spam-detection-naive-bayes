#!/usr/bin/env bash
set -euo pipefail
python experiment.py
python plot_results.py
python confidence_intervals.py | tee results/ci_output.txt
