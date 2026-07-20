#!/bin/bash
# Usage: sudo ./restore.sh
# Resurrects the process from ./ckpt/ (detached). It resumes appending to
# counter.log from where it left off.
set -e
criu restore -D ckpt -d -v4 -o restore.log
echo "restored — tail -f counter.log to watch it resume"
