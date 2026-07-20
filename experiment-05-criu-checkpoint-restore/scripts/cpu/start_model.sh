#!/bin/bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
setsid ~/venv/bin/python cpu_counter.py < /dev/null &> model.log &
echo "model PID: $!"
