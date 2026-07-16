#!/usr/bin/env bash
# run_experiments.sh — full experiment loop (teacher feedback):
#   * regenerate tasks for N in {100, 200, 500, 1000}
#   * run every policy (Local, Random, RoundRobin, Greedy, DQN) for each N
#   * collect .sca/.vec into results/n<N>/
#   * generate comparison plots (per-N + multi-task-count)
#
# Usage:
#   ./run_experiments.sh                 # train DQN (3000 ep) + full loop
#   SKIP_TRAIN=1 ./run_experiments.sh    # reuse existing dqn_weights/
#   QUICK=1   ./run_experiments.sh       # fast DQN smoke train + full loop
set -e
cd "$(dirname "$0")"

# --- source OMNeT++ environment if opp tools are not on PATH ---
if ! command -v opp_msgc >/dev/null 2>&1; then
  OMNETPP_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
  # fall back to the standard location relative to this project
  for cand in "../../../omnetpp-6.4.0" "../../.."; do
    if [ -f "$cand/setenv" ]; then OMNETPP_ROOT="$cand"; break; fi
  done
  if [ -n "$OMNETPP_ROOT" ] && [ -f "$OMNETPP_ROOT/setenv" ]; then
    # shellcheck disable=SC1091
    source "$OMNETPP_ROOT/setenv" >/dev/null
  fi
fi

BIN=../src/Tutorial
TASK_COUNTS=(100 200 500 1000)
POLICIES=(Local Random RoundRobin Greedy DQN)

echo "============================================================"
echo " Building the simulation binary"
echo "============================================================"
( cd ../src && make -j4 >/dev/null 2>&1 || make >/dev/null 2>&1 )
echo "  built: $BIN"

# --- DQN training (mirrors heterogeneous CPU/GPU/QPU model) ---
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  echo "============================================================"
  echo " Training DQN policy"
  echo "============================================================"
  if [ "${QUICK:-0}" = "1" ]; then
    python3 train_dqn.py --quick
  else
    python3 train_dqn.py
  fi
else
  echo "SKIP_TRAIN=1 -> reusing existing dqn_weights/"
fi

# --- main experiment loop ---
for N in "${TASK_COUNTS[@]}"; do
  echo "============================================================"
  echo " N=$N tasks: regenerating tasks.csv"
  echo "============================================================"
  python3 generate_tasks.py -n "$N" -o tasks.csv >/dev/null

  OUT="results/n${N}"
  rm -rf "$OUT"
  mkdir -p "$OUT"

  for POL in "${POLICIES[@]}"; do
    echo "  -> running $POL (N=$N)..."
    "$BIN" -u Cmdenv -c "$POL" -n .:../src \
           --result-dir="$OUT" --seed-set=0 \
           >/dev/null 2>&1
  done

  # per-N plots
  python3 plot_results.py --results "$OUT"
done

echo "============================================================"
echo " Multi-task-count comparison (100/200/500/1000)"
echo "============================================================"
python3 plot_results.py --multi --results results

# Refresh the top-level results/ dashboard from the largest task count (n1000)
# so results/plots/summary_dashboard.png etc. reflect the latest run, not stale
# files from a previous experiment.
LAST="results/n${TASK_COUNTS[-1]}"
if [ -d "$LAST" ]; then
  cp -f "$LAST"/*.sca "$LAST"/*.vec results/ 2>/dev/null || true
  rm -f results/plots/*.png 2>/dev/null || true
  python3 plot_results.py --results results
fi

echo ""
echo "DONE. Charts in results/plots/ (per-N charts in results/n<N>/plots/)"
