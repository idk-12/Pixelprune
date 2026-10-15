import os, sys, pandas as pd
os.chdir('/data/lijie/VLMEvalKit')
sys.path.insert(0, '/data/lijie/VLMEvalKit')

from vlmeval.config import supported_VLM
from vlmeval.dataset.utils.yorn import YOrN_Extraction

print("=== 加载模型 ===", flush=True)
model = supported_VLM['Qwen2.5-VL-7B-Instruct']()
print("模型加载完成\n", flush=True)

lmu = os.path.expanduser('~/LMUData')

# ---------- POPE ----------
print("=== POPE 样本 ===", flush=True)
pope_df = pd.read_csv(f'{lmu}/POPE.tsv', sep='\t', nrows=5)
for _, row in pope_df.iterrows():
    img_path = f"{lmu}/images/POPE/{row['index']}.jpg"
    if not os.path.exists(img_path):
        print(f"  [skip] 图片不存在: {img_path}")
        continue
    msg = [
        {'type': 'image', 'value': img_path},
        {'type': 'text',  'value': row['question']},
    ]
    raw = model.generate(msg)
    extracted = YOrN_Extraction(raw)
    correct = 'O' if extracted == row['answer'] else 'X'
    print(f"[{correct}] Q: {row['question']}", flush=True)
    print(f"    GT={row['answer']}  extracted={extracted}  raw={repr(raw[:100])}\n", flush=True)

# ---------- MME ----------
print("=== MME 样本 ===", flush=True)
mme_df = pd.read_csv(f'{lmu}/MME.tsv', sep='\t', nrows=6)
for _, row in mme_df.iterrows():
    img_path = f"{lmu}/images/MME/{row['image_path']}"
    if not os.path.exists(img_path):
        print(f"  [skip] 图片不存在: {img_path}")
        continue
    msg = [
        {'type': 'image', 'value': img_path},
        {'type': 'text',  'value': row['question']},
    ]
    raw = model.generate(msg)
    extracted = YOrN_Extraction(raw)
    correct = 'O' if extracted == row['answer'] else 'X'
    print(f"[{correct}] [{row['category']}] Q: {row['question']}", flush=True)
    print(f"    GT={row['answer']}  extracted={extracted}  raw={repr(raw[:100])}\n", flush=True)
