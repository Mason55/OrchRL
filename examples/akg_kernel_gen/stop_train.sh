#!/usr/bin/env bash
set -euo pipefail

PATTERNS=(
  "python3 -m examples.akg_kernel_gen.train_verl_base"
  "bash examples/akg_kernel_gen/train.sh"
  "bash pull_train.sh"
)

declare -A seen=()
pids=()

for pattern in "${PATTERNS[@]}"; do
  while IFS= read -r pid; do
    [[ -z "${pid:-}" ]] && continue
    [[ "$pid" == "$$" ]] && continue
    [[ -n "${seen[$pid]:-}" ]] && continue
    seen["$pid"]=1
    pids+=("$pid")
  done < <(pgrep -f "$pattern" || true)
done

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "[INFO] No AKG training processes found."
  exit 0
fi

echo "[INFO] Stopping AKG training processes: ${pids[*]}"
ps -fp "${pids[@]}" || true

kill "${pids[@]}" || true
sleep 3

remaining=()
for pid in "${pids[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    remaining+=("$pid")
  fi
done

if [[ "${#remaining[@]}" -gt 0 ]]; then
  echo "[WARN] Force killing remaining processes: ${remaining[*]}"
  kill -9 "${remaining[@]}" || true
fi

echo "[INFO] Done."
