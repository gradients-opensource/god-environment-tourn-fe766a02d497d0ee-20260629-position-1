#!/bin/bash
set -e

redis-server --daemonize yes
sleep 10

echo "*****Running text trainer"
source /workspace/.grpo_env/bin/activate
python3 /workspace/scripts/text_trainer.py "$@"
deactivate