#!/usr/bin/env bash
set -euo pipefail

run_dir=${1:?run directory required}
model_dir=${2:?model directory required}
dataset_dir=${3:?dataset directory required}
python_bin=${4:-python3}
downloader=${5:-download_lcb_assets.py}

mkdir -p "$run_dir" "$model_dir" "$dataset_dir"

nohup env HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=60 \
  "$python_bin" "$downloader" model \
  --output "$model_dir" --receipt "$run_dir/model_status.json" --workers 8 \
  >"$run_dir/model.log" 2>&1 &
echo $! >"$run_dir/model.pid"

nohup env HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=60 \
  "$python_bin" "$downloader" dataset \
  --output "$dataset_dir" --receipt "$run_dir/dataset_status.json" --workers 4 \
  >"$run_dir/dataset.log" 2>&1 &
echo $! >"$run_dir/dataset.pid"

sleep 2
printf 'MODEL_PID=%s\n' "$(cat "$run_dir/model.pid")"
printf 'DATASET_PID=%s\n' "$(cat "$run_dir/dataset.pid")"
ps -p "$(cat "$run_dir/model.pid")" -o pid=,stat=,etime=,cmd=
ps -p "$(cat "$run_dir/dataset.pid")" -o pid=,stat=,etime=,cmd=
for receipt in "$run_dir"/*status.json; do
  printf 'STATUS=%s\n' "$receipt"
  cat "$receipt"
done
