#!/bin/bash
set -e

# Start Xvfb for GUI applications if needed
Xvfb :99 -screen 0 1024x768x16 &

# Execute the command passed to docker run
exec "$@"
