#!/usr/bin/env bash
# Runs the model ladder concurrently, one process per model.
#
# Groq meters tokens per minute per model, so a serial run leaves every other
# model's allowance idle while it waits on one. Running them side by side costs
# nothing extra and is bounded by the slowest single model rather than by their
# sum.
#
# Each process gets its own SQLite log and results file: five writers on one
# database contend for the write lock, and a benchmark should not spend its time
# waiting on itself. Merge afterwards with examples/merge_results.py.
#
# The critic is off here, and that is not a shortcut. It runs on a single small
# model, so every process would queue behind that one model's token allowance
# while the models under test sat idle — measured at roughly two and a half
# minutes of waiting around a generation that took under a second. Semantic
# accuracy comes from the hand-written labels regardless, which cost no requests
# at all; the critic's second opinion is measured separately, on one slice, in
# examples/run_critic_slice.py.
#
# Usage: bash examples/run_bench_parallel.sh [K]

set -u
K="${1:-3}"
PYTHON="${PYTHON:-../.venv/Scripts/python.exe}"

# Read from the probe, never hardcoded. A run of this benchmark lost two of its
# five models overnight when Groq dropped them from its line-up, which is the
# same argument the project makes about capabilities applied to existence.
mapfile -t MODELS < <("$PYTHON" examples/ladder.py --provider groq)
if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "no probed models; run: so-agent probe --provider groq" >&2
  exit 1
fi

mkdir -p runs

for model in "${MODELS[@]}"; do
  slug="${model//\//-}"
  LOG_DB_PATH="runs/${slug}.db" \
  "$PYTHON" examples/run_bench.py \
      --provider groq \
      --models "$model" \
      --k "$K" \
      --no-review \
      --out "runs/${slug}.json" \
      > "runs/${slug}.log" 2>&1 &
  echo "started $model (pid $!)"
done

wait
echo "all models finished"
