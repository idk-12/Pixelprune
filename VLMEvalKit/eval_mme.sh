#!/bin/bash
# eval_mme.sh
# 对比 Baseline vs MMTok 各变体在 MME 上的指标
# 用法：bash eval_mme.sh [N]   N=0 表示全量，默认 0

set -e
N=${1:-50}
DATA=MME
WORK_DIR="./outputs/mme"
RESULT_FILE="${WORK_DIR}/summary_$(date +%Y%m%d_%H%M%S).txt"

MODELS=(
    # "Qwen3.6-27B-base"
    # "Qwen3.6-27B-MMTok-r90"
    "Qwen3.6-27B-MMTok-r70"
    # "Qwen3.6-27B-MMTok-r50"
    # "Qwen2.5-VL-7B-Instruct"
    # "Qwen2.5-VL-7B-Pruned-r70"
    # "Qwen2.5-VL-7B-MMTok-r90"
    # "Qwen2.5-VL-7B-MMTok-r70"
    # "Qwen2.5-VL-7B-MMTok-r50"
)

mkdir -p "$WORK_DIR"

echo "============================================"
echo " MME 对比实验"
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
echo " MME 结果汇总"
echo " perception / reasoning / total  (各子类分数之和)"
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
    f = find_latest(f"{work_dir}/{m}/**/*MME*.csv")
    if f is None:
        f = find_latest(f"{work_dir}/{m}/**/*MME*.xlsx")
    percept = reason = None
    if f:
        try:
            df = pd.read_csv(f) if f.endswith('.csv') else pd.read_excel(f)
            row = df.iloc[0]
            if 'perception' in df.columns:
                percept = float(row['perception'])
            if 'reasoning' in df.columns:
                reason  = float(row['reasoning'])
        except Exception:
            pass

    timing_files = sorted(glob.glob(f"{work_dir}/{m}/**/*_timing.json", recursive=True))
    mean_t = total_t = None
    if timing_files:
        with open(timing_files[-1]) as tf:
            d = json.load(tf)
        mean_t  = d.get('mean_seconds')
        total_t = d.get('total_seconds')

    rows.append({'model': m, 'percept': percept, 'reason': reason,
                 'mean_s': mean_t, 'total_s': total_t})

base_t = next((r['mean_s'] for r in rows if r['mean_s'] is not None), None)

print(f"\n{'Model':<36} {'Percept':>9} {'Reason':>9} {'Total':>9} {'Mean(s)':>9} {'Total(s)':>10} {'Speedup':>9}")
print("-" * 95)
for r in rows:
    p, re = r['percept'], r['reason']
    p_str = f"{p:.1f}"       if p  is not None else "—"
    r_str = f"{re:.1f}"      if re is not None else "—"
    t_str = f"{p+re:.1f}"    if (p is not None and re is not None) else "—"
    mt    = f"{r['mean_s']:.2f}"  if r['mean_s']  is not None else "—"
    tt    = f"{r['total_s']:.1f}" if r['total_s'] is not None else "—"
    sp    = f"{base_t/r['mean_s']:.2f}x" if (base_t and r['mean_s']) else "—"
    print(f"{r['model']:<36} {p_str:>9} {r_str:>9} {t_str:>9} {mt:>9} {tt:>10} {sp:>9}")

print(f"\n(计时来自 *_timing.json，结果目录: {work_dir})")
PYEOF
} | tee "$RESULT_FILE"

echo ""
echo ">>> 汇总已保存到 $RESULT_FILE"
