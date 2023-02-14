#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port 29598 parallel_pred_based.py --launcher pytorch ${PY_ARGS}

