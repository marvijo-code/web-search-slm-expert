#!/usr/bin/env bash
# One training lane on a CPU cloud machine.
# usage: bash run_lane.sh <name> <train-minutes> [train.py model args...]
#   GEN=<path to gen_data.py>  use another data generator (default data/gen_data.py)
#   GEN_ARGS="..."             extra generator flags (e.g. --copy 0.4 for gen_agent.py)
# Writes runs/<name>/ : train_log.jsonl, stdout.log, eval.json, weights/, parity_expected.json, hardware.txt, STATUS
set -euo pipefail
cd "$(dirname "$0")"
NAME=$1; MIN=$2; shift 2
OUT=runs/$NAME
GEN=${GEN:-data/gen_data.py}
mkdir -p "$OUT"
echo RUNNING > "$OUT/STATUS"
trap 'echo "FAILED (line $LINENO)" > "$OUT/STATUS"' ERR

# pip may put torch under a different python than the one called "python"
PY=""
for c in python python3 python3.13 python3.12 python3.11; do
  if command -v $c >/dev/null && $c -c 'import torch' 2>/dev/null; then PY=$(command -v $c); break; fi
done
[ -n "$PY" ] || { echo "no python with torch" > "$OUT/STATUS"; exit 1; }
mkdir -p /tmp/pyshim && ln -sf "$PY" /tmp/pyshim/python && export PATH=/tmp/pyshim:$PATH

{ echo "nproc: $(nproc)"; lscpu | grep -E 'Model name|Flags' | sed 's/Flags:.*\(amx[a-z_0-9]*\).*/AMX: yes (\1)/'; free -g | head -2; python -c 'import torch;print("torch", torch.__version__)'; } > "$OUT/hardware.txt" 2>&1 || true

DATA=data/build_$NAME
[ -f $DATA/train.jsonl ] || python "$GEN" --out $DATA --train 200000 ${GEN_ARGS:-} > "$OUT/data_stats.txt"

# Speed probe: plain fp32 vs bfloat16 autocast. Keep whichever is faster.
FP=$(python train.py --data $DATA --out /tmp/probe_$NAME --probe 25 "$@" | grep '"probe"' | python -c 'import sys,json;print(json.loads(sys.stdin.read())["sec_per_step"])')
BF=$(python train.py --data $DATA --out /tmp/probe_$NAME --probe 25 --bf16 "$@" | grep '"probe"' | python -c 'import sys,json;print(json.loads(sys.stdin.read())["sec_per_step"])')
EXTRA=""
if python -c "import sys; sys.exit(0 if $BF < 0.85 * $FP else 1)"; then EXTRA="--bf16"; fi
echo "probe fp32 ${FP}s/step  bf16 ${BF}s/step  -> using '${EXTRA:-fp32}'" | tee "$OUT/probe.txt"

echo TRAINING > "$OUT/STATUS"
python train.py --data $DATA --out "$OUT" --minutes "$MIN" $EXTRA "$@" > "$OUT/stdout.log" 2>&1
echo EVALUATING > "$OUT/STATUS"
python evaluate.py --ckpt "$OUT/ckpt.pt" --data $DATA --out "$OUT/eval.json" >> "$OUT/stdout.log" 2>&1
python export.py --ckpt "$OUT/ckpt.pt" --out "$OUT/weights" >> "$OUT/stdout.log" 2>&1
python tests/parity_dump.py "$OUT/ckpt.pt" "$OUT/parity_expected.json" >> "$OUT/stdout.log" 2>&1
echo DONE > "$OUT/STATUS"
