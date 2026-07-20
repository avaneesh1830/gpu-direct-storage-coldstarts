#!/bin/bash
# Start the counter detached from the terminal (no TTY = simplest CRIU dump).
# Prints the PID you'll pass to checkpoint.sh.
setsid python3 counter.py < /dev/null &> counter.log &
echo "counter PID: $!"
