#!/bin/bash
# Monitor training process and restart on crash
LOGFILE="/u/aseo/Verlog/logs/test_run.log"
TRAINSCRIPT="/u/aseo/Verlog/test.sh"
MONITOR_LOG="/u/aseo/Verlog/logs/monitor.log"
MAIN_PID_FILE="/u/aseo/Verlog/logs/train_pid.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"
}

get_train_pid() {
    pgrep -f "verl.trainer.main_ppo.*ppo_epoch" | head -1
}

check_healthy() {
    local pid=$1
    kill -0 "$pid" 2>/dev/null
}

launch_training() {
    log "Launching training..."
    : > "$LOGFILE"
    cd /u/aseo/Verlog
    source /u/aseo/anaconda3/bin/activate verlog
    bash "$TRAINSCRIPT" >> "$LOGFILE" 2>&1 &
    local pid=$!
    sleep 5
    # Get the actual python process pid
    local ppid
    ppid=$(get_train_pid)
    echo "${ppid:-$pid}" > "$MAIN_PID_FILE"
    log "Training launched, pid=${ppid:-$pid}"
    echo "${ppid:-$pid}"
}

analyze_crash() {
    log "Analyzing crash..."
    local last_error
    last_error=$(tail -200 "$LOGFILE" 2>/dev/null | grep -E "Error|Exception|OOM|killed|SIGKILL|Traceback|oom_kill" | tail -20)
    log "Last errors: $last_error"

    if echo "$last_error" | grep -qiE "oom|out of memory|memory|killed"; then
        log "OOM crash detected"
        echo "oom"
    elif echo "$last_error" | grep -qiE "ActorDiedError|SIGSEGV|SIGKILL|connection error code 2"; then
        log "Actor/worker crash detected"
        echo "actor_died"
    elif echo "$last_error" | grep -qiE "CUDA|cuda"; then
        log "CUDA error detected"
        echo "cuda"
    else
        log "Unknown crash type"
        echo "unknown"
    fi
}

log "=== Monitor started ==="
log "Watching: $LOGFILE"

TRAIN_PID=$(get_train_pid)
if [ -z "$TRAIN_PID" ]; then
    log "No training process found, launching..."
    TRAIN_PID=$(launch_training)
else
    log "Found existing training process: $TRAIN_PID"
fi

CRASH_COUNT=0
LAST_STEP=0
STALL_COUNT=0

while true; do
    sleep 30

    TRAIN_PID=$(get_train_pid)

    if [ -z "$TRAIN_PID" ]; then
        CRASH_COUNT=$((CRASH_COUNT + 1))
        log "Training process died! (crash #$CRASH_COUNT)"

        CRASH_TYPE=$(analyze_crash)
        log "Crash type: $CRASH_TYPE"

        # Wait for ray to clean up
        sleep 10
        pkill -f "ray" 2>/dev/null
        sleep 5

        log "Relaunching training (attempt $CRASH_COUNT)..."
        TRAIN_PID=$(launch_training)
        LAST_STEP=0
        STALL_COUNT=0
        continue
    fi

    # Check progress in log
    CURRENT_STEP=$(grep -oP 'Training Progress:.*?\K\d+(?=/)' "$LOGFILE" 2>/dev/null | tail -1)
    if [ -n "$CURRENT_STEP" ]; then
        if [ "$CURRENT_STEP" -eq "$LAST_STEP" ]; then
            STALL_COUNT=$((STALL_COUNT + 1))
            if [ "$STALL_COUNT" -ge 10 ]; then
                log "WARNING: Training stalled at step $CURRENT_STEP for $((STALL_COUNT * 30))s"
            fi
        else
            STALL_COUNT=0
            LAST_STEP=$CURRENT_STEP
        fi
        log "Status: pid=$TRAIN_PID step=$CURRENT_STEP stall_count=$STALL_COUNT"
    else
        log "Status: pid=$TRAIN_PID (initializing...)"
    fi

    # Log GPU memory
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | awk -F',' '{used+=$1; total+=$2} END {printf "used=%.0fMiB/total=%.0fMiB", used, total}')
    log "GPU memory: $GPU_MEM"
done
