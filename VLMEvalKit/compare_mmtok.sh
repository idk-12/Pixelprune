#!/bin/bash
# compare_mmtok.sh
# 用 N 条样本对比 Qwen3.6-27B baseline vs MMTok 各变体的精度和速度
# 用法：bash compare_mmtok.sh [N]   默认 N=500

set -e
N=${1:-0}
DATA=TextVQA_VAL
# 输出根目录：outputs/<benchmark小写去后缀>
BENCH_DIR=$(echo "$DATA" | tr '[:upper:]' '[:lower:]' | sed 's/_val$//' | sed 's/_dev.*$//')
WORK_DIR="./outputs/${BENCH_DIR}"
RESULT_FILE="${WORK_DIR}/summary_$(date +%Y%m%d_%H%M%S).txt"
MODELS=(
    # "Qwen3.6-27B-base"
    # "Qwen3.6-27B-MMTok-r90"
    "Qwen3.6-27B-MMTok-r70"
    # "Qwen3.6-27B-MMTok-r50"
    # "Qwen2.5-VL-7B-Instruct"
    # "Qwen2.5-VL-7B-MMTok-r90"
    # "Qwen2.5-VL-7B-MMTok-r70"
    # "Qwen2.5-VL-7B-MMTok-r50"
)

echo "============================================"
echo " MMTok vs Baseline 对比实验"
echo " Dataset : $DATA"
echo " Samples : $N"
echo " Models  : ${#MODELS[@]} variants"
echo "============================================"

# ---------- 推理 + 评测 ----------
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo ">>> [Infer+Eval] $MODEL  (${N} samples)"
    python -u run.py \
        --data $DATA \
        --model "$MODEL" \
        --max-samples $N \
        --judge exact_matching \
        --work-dir "$WORK_DIR"
done

# ---------- 汇总对比表 ----------
echo ""
{
echo "============================================"
echo " 结果汇总"
echo "============================================"

python - "$N" "$WORK_DIR" "${MODELS[@]}" << 'PYEOF'
import sys, json, glob
import pandas as pd

n_samples = sys.argv[1]
work_dir  = sys.argv[2]
models    = sys.argv[3:]

rows = []
for m in models:
    # 计时：取最新的 timing json
    timing_files = sorted(glob.glob(f"{work_dir}/{m}/**/*_timing.json", recursive=True))
    mean_t = total_t = None
    if timing_files:
        with open(timing_files[-1]) as f:
            d = json.load(f)
        mean_t = d.get('mean_seconds')
        total_t = d.get('total_seconds')

    # 精度：取最新的 TextVQA csv/xlsx
    acc = None
    for pat in [f"{work_dir}/{m}/**/*TextVQA*.csv", f"{work_dir}/{m}/**/*TextVQA*.xlsx"]:
        files = sorted(glob.glob(pat, recursive=True))
        for cf in reversed(files):
            try:
                df = pd.read_csv(cf) if cf.endswith('.csv') else pd.read_excel(cf)
                if 'Overall' in df.columns:
                    acc = float(df['Overall'].iloc[-1])
                    break
                if 'overall' in df.columns:
                    acc = float(df['overall'].iloc[-1])
                    break
            except Exception:
                pass
        if acc is not None:
            break

    rows.append({'model': m, 'acc': acc, 'mean_s': mean_t, 'total_s': total_t})

# 第一个有计时数据的模型作为基线
base_t = next((r['mean_s'] for r in rows if r['mean_s'] is not None), None)

print(f"\n{'Model':<36} {'Acc(%)':>8} {'Mean(s)':>9} {'Total(s)':>10} {'Speedup':>9}")
print("-" * 76)
for r in rows:
    sp  = f"{base_t / r['mean_s']:.2f}x" if base_t and r['mean_s'] else "—"
    acc = f"{r['acc']:.2f}"  if r['acc']   is not None else "—"
    mt  = f"{r['mean_s']:.2f}" if r['mean_s'] is not None else "—"
    tt  = f"{r['total_s']:.1f}" if r['total_s'] is not None else "—"
    print(f"{r['model']:<36} {acc:>8} {mt:>9} {tt:>10} {sp:>9}")

print(f"\n(基于 {n_samples} 条样本，计时来自 *_timing.json，精度来自 TextVQA evaluate 输出，结果目录: {work_dir})")
PYEOF
} | tee "$RESULT_FILE"

echo ""
echo ">>> 汇总已保存到 $RESULT_FILE"
