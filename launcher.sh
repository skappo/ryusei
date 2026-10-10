#!/bin/bash
# launcher.sh - Pre-Flight Git-Aware OTA Rollback Checker

SCRIPT_DIR="/media/pi/USB/meteora"
SCRIPT_NAME="meteora.py"  # The main entry file
CRASH_FILE="$SCRIPT_DIR/crash_count.txt"

cd $SCRIPT_DIR

# Read current crash count (default to 0)
if [ -f "$CRASH_FILE" ]; then
    CRASH_COUNT=$(cat "$CRASH_FILE")
else
    CRASH_COUNT=0
fi

# Check if we are in a death loop (3 rapid crashes)
if [ "$CRASH_COUNT" -ge 3 ]; then
    echo "LAUNCHER: Death loop detected ($CRASH_COUNT rapid crashes)! Rolling back via Git..."
    
    # 'HEAD@{1}' tells Git to revert the repository to the state it was in BEFORE the last pull!
    echo "LAUNCHER: Reverting Git repository to previous stable state (HEAD@{1})..."
    git reset --hard HEAD@{1}
    
    if [ $? -eq 0 ]; then
        echo "LAUNCHER: Git Rollback Successful! Restored previous working version."
    else
        echo "LAUNCHER: FATAL Git Rollback Failed. Intervention required."
    fi
    
    # Reset counter after rollback so the good version can run
    echo "0" > "$CRASH_FILE"
else
    # Increment the crash counter BEFORE starting Python.
    # Python will reset this to 0 if it survives its startup phase!
    NEW_COUNT=$((CRASH_COUNT + 1))
    echo "$NEW_COUNT" > "$CRASH_FILE"
fi

echo "LAUNCHER: Handing control to Python..."
# 'exec' replaces bash with Python. Systemd tracks Python directly!
exec python3 $SCRIPT_NAME --config config.json