#!/bin/sh

# Download and prepare the video evaluation datasets used by PixelPrune.
#
# Usage:
#   sh download_video_datasets.sh
#   sh download_video_datasets.sh Video-MME MLVU_MCQ LongVideoBench
#
# To store the datasets on another disk:
#   LMUData=/path/with/at-least-1.2TiB-free sh download_video_datasets.sh

set -eu

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
VLMEVALKIT_DIR=$(CDPATH= cd "${SCRIPT_DIR}/../VLMEvalKit" 2>/dev/null && pwd || true)
PYTHON_BIN=${PYTHON_BIN:-python}

if [ -z "$VLMEVALKIT_DIR" ] || [ ! -f "${VLMEVALKIT_DIR}/vlmeval/__init__.py" ]; then
    echo "Error: VLMEvalKit must be next to pixelprune_npu." >&2
    echo "Expected: ${SCRIPT_DIR}/../VLMEvalKit/vlmeval/__init__.py" >&2
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

LMUData=${LMUData:-${SCRIPT_DIR}/eval/LMUData}
MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-${LMUData}/modelscope}
MODELSCOPE_CACHE_HOME=${MODELSCOPE_CACHE_HOME:-${MODELSCOPE_CACHE}}
MODELSCOPE_CACHE_DIR=${MODELSCOPE_CACHE_DIR:-${MODELSCOPE_CACHE}}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-${LMUData}/.cache}
HF_HOME=${HF_HOME:-${XDG_CACHE_HOME}/huggingface}
HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}
VLMEVALKIT_USE_MODELSCOPE=1
PYTHONPATH="${SCRIPT_DIR}:${VLMEVALKIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

export LMUData
export VLMEVALKIT_DIR
export MODELSCOPE_CACHE MODELSCOPE_CACHE_HOME MODELSCOPE_CACHE_DIR
export XDG_CACHE_HOME HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE
export VLMEVALKIT_USE_MODELSCOPE PYTHONPATH

mkdir -p "$MODELSCOPE_CACHE" "$HF_HUB_CACHE"

if [ "$#" -eq 0 ]; then
    DATASETS="Video-MME MLVU_MCQ LongVideoBench"
else
    DATASETS="$*"
fi
export DATASETS

echo "PixelPrune dataset downloader"
echo "  PixelPrune:  $SCRIPT_DIR"
echo "  VLMEvalKit:  $VLMEVALKIT_DIR"
echo "  Data root:   $LMUData"
echo "  Datasets:    $DATASETS"
echo
echo "The complete download can occupy about 1.1 TiB."
echo "Interrupted downloads can be resumed by running this script again."
echo

"$PYTHON_BIN" - <<'PY'
import os
import sys

try:
    from vlmeval.dataset.longvideobench import LongVideoBench
    from vlmeval.dataset.mlvu import MLVU_MCQ
    from vlmeval.dataset.videomme import VideoMME
except ModuleNotFoundError as exc:
    kit_dir = os.environ["VLMEVALKIT_DIR"]
    raise SystemExit(
        "Missing a Python dependency required by VLMEvalKit "
        f"({exc.name}).\nInstall the evaluation environment first, for example:\n"
        f"  {sys.executable} -m pip install -e {kit_dir}"
    ) from exc


def download_videomme():
    VideoMME(dataset="Video-MME", nframe=64)


def download_mlvu():
    MLVU_MCQ(dataset="MLVU_MCQ", nframe=64)


def download_longvideobench():
    LongVideoBench(dataset="LongVideoBench", nframe=64)


downloaders = {
    "Video-MME": download_videomme,
    "MLVU_MCQ": download_mlvu,
    "LongVideoBench": download_longvideobench,
}

requested = os.environ["DATASETS"].split()
unknown = [name for name in requested if name not in downloaders]
if unknown:
    choices = ", ".join(downloaders)
    raise SystemExit(
        f"Unknown dataset(s): {', '.join(unknown)}. Supported values: {choices}"
    )

for name in requested:
    print(f"\n===== Downloading and preparing {name} =====", flush=True)
    downloaders[name]()
    print(f"===== {name} is ready =====", flush=True)

print("\nAll requested datasets are ready.", flush=True)
PY

echo "Data is available under: $LMUData"
