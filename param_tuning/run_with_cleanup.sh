#!/bin/bash
# Wrapper script that runs a command and ensures cleanup afterward

# Run the command passed as arguments
"$@"
EXIT_CODE=$?

# Aggressive cleanup
echo "=== Cleaning up after test ==="

# Kill all Ray-related processes
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "ray_" 2>/dev/null || true
pkill -9 -f "raylet" 2>/dev/null || true
pkill -9 -f "gcs_server" 2>/dev/null || true
pkill -9 -f "redis-server.*ray" 2>/dev/null || true
pkill -9 -f "DefaultWorker" 2>/dev/null || true

# Wait a moment for processes to die
sleep 2

# Check if any Ray processes are still running
if pgrep -f "ray::" > /dev/null 2>&1; then
    echo "WARNING: Ray processes still running after cleanup!"
    pgrep -af "ray::" || true
fi

# Clear CUDA memory by resetting GPUs (if nvidia-smi available)
if command -v nvidia-smi &> /dev/null; then
    echo "Resetting GPUs..."
    nvidia-smi --gpu-reset -i 0,1,2,3 2>/dev/null || echo "GPU reset failed (might be in use)"
fi

# Wait for resources to be fully released (but not 200 seconds!)
sleep 8

echo "=== Cleanup complete ==="

exit $EXIT_CODE


