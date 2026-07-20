#!/bin/bash
# Usage: sudo ./checkpoint.sh <pid>
# Dumps the process state into ./ckpt/ and KILLS the process (default CRIU
# behavior — that's the point: it must be dead so restore proves resurrection).
set -e
PID=$1
mkdir -p ckpt
criu dump -t "$PID" -D ckpt -v4 -o dump.log
echo "dumped PID $PID -> ckpt/ ($(du -sh ckpt | cut -f1))"
