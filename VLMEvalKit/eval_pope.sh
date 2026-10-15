#!/bin/bash
# eval_pope.sh
# 对比 Baseline vs MMTok 各变体在 POPE 上的指标
# 用法：bash eval_pope.sh [N]   N=0 表示全量，默认 0

set -e
N=${1:-50}
DATA=POPE
WORK_DIR="./outputs/pope"
RESULT_FILE="${WORK_DIR}/summary_$(date +%Y%m%d_%H%M%S).txt"

MODELS=(
    # "Qwen3.6-27B-base"
    # "Qwen3.6-27B-MMTok-r90"
    # "Qwen3.6-27B-MMTok-r70"
    # "Qwen3.6-27B-MMTok-r50"
    "Qwen2.5-VL-7B-Instruct"
    "Qwen2.5-VL-7B-Pruned-r70"
    # "Qwen2.5-VL-7B-MMTok-r90"
    # "Qwen2.5-VL-7B-MMTok-r70"
    # "Qwen2.5-VL-7B-MMTok-r50"
)

mkdir -p "$WORK_DIR"

echo "============================================"
echo " POPE 对比实验"
echo " Samples : $N  (0=全量)"
echo " Models  : ${#MODELS[@]} variants"
echo " Work dir: $WORK_DIR"
echo "============================================"

# ---------- 推理 + 评测 ----------
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo ">>> [Infer+Eval] $MODEL  on $DATA  (N=$N)"
    python -u run.py \
        --data "$DATA" \
        --model "$MODEL" \
        --max-samples "$N" \
        --judge exact_matching \
        --work-dir "$WORK_DIR"
done

# ---------- 汇总对比表 ----------
echo ""
{
echo "============================================"
echo " POPE 结果汇总"
echo " Overall=F1, acc, precision, recall"
echo "============================================"

python - "$WORK_DIR" "${MODELS[@]}" << 'PYEOF'
import sys, glob, json
import pandas as pd

work_dir = sys.argv[1]
models   = sys.argv[2:]

def find_latest(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    return files[-1] if files else None

rows = []
for m in models:
    f = find_latest(f"{work_dir}/{m}/**/*POPE*.csv")
    if f is None:
        f = find_latest(f"{work_dir}/{m}/**/*POPE*.xlsx")
    f1 = acc = prec = rec = None
    if f:
        try:
            df = pd.read_csv(f) if f.endswith('.csv') else pd.read_excel(f)
            row = df[df['split'] == 'Overall'].iloc[0] if 'split' in df.columns else df.iloc[0]
            if 'Overall'   in df.columns: f1   = float(row['Overall'])
            if 'acc'       in df.columns: acc  = float(row['acc'])
            if 'precision' in df.columns: prec = float(row['precision'])
            if 'recall'    in df.columns: rec  = float(row['recall'])
        except Exception:
            pass

    timing_files = sorted(glob.glob(f"{work_dir}/{m}/**/*_timing.json", recursive=True))
    mean_t = total_t = None
    if timing_files:
        with open(timing_files[-1]) as tf:
            d = json.load(tf)
        mean_t  = d.get('mean_seconds')
        total_t = d.get('total_seconds')

    rows.append({'model': m, 'f1': f1, 'acc': acc, 'prec': prec, 'rec': rec,
                 'mean_s': mean_t, 'total_s': total_t})

base_t = next((r['mean_s'] for r in rows if r['mean_s'] is not None), None)

print(f"\n{'Model':<36} {'F1(%)':>8} {'Acc(%)':>8} {'Prec(%)':>9} {'Recall(%)':>10} {'Mean(s)':>9} {'Total(s)':>10} {'Speedup':>9}")
print("-" * 102)
for r in rows:
    f1_s   = f"{r['f1']:.2f}"   if r['f1']   is not None else "—"
    acc_s  = f"{r['acc']:.2f}"  if r['acc']  is not None else "—"
    prec_s = f"{r['prec']:.2f}" if r['prec'] is not None else "—"
    rec_s  = f"{r['rec']:.2f}"  if r['rec']  is not None else "—"
    mt     = f"{r['mean_s']:.2f}"  if r['mean_s']  is not None else "—"
    tt     = f"{r['total_s']:.1f}" if r['total_s'] is not None else "—"
    sp     = f"{base_t/r['mean_s']:.2f}x" if (base_t and r['mean_s']) else "—"
    print(f"{r['model']:<36} {f1_s:>8} {acc_s:>8} {prec_s:>9} {rec_s:>10} {mt:>9} {tt:>10} {sp:>9}")

print(f"\n(计时来自 *_timing.json，结果目录: {work_dir})")
PYEOF
} | tee "$RESULT_FILE"

echo ""
echo ">>> 汇总已保存到 $RESULT_FILE"
