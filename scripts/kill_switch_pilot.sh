#!/bin/bash
# Kill switch automatique pour le pilot run (timeout 6h).
# Usage : bash scripts/kill_switch_pilot.sh &
ZONE="europe-west4-a"
VM_NAME="axiom-training"
TIMEOUT_SECONDS=21600  # 6h

echo "Kill switch actif — timeout dans ${TIMEOUT_SECONDS}s ($(date -d "+${TIMEOUT_SECONDS} seconds" '+%H:%M:%S' 2>/dev/null || date '+%H:%M:%S'))"
sleep "$TIMEOUT_SECONDS"

echo "=== TIMEOUT atteint — envoi SIGTERM au pilot run ==="
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
PID=\$(cat ~/pilot.pid 2>/dev/null)
if [ -n \"\$PID\" ] && kill -0 \"\$PID\" 2>/dev/null; then
    echo \"SIGTERM -> PID \$PID\"
    kill -SIGTERM \"\$PID\"
    sleep 30
    kill -0 \"\$PID\" 2>/dev/null && kill -9 \"\$PID\" && echo 'SIGKILL envoyé' || echo 'Process terminé proprement'
else
    echo 'Process déjà terminé ou PID absent'
fi
" 2>&1

echo "Kill switch terminé — $(date '+%Y-%m-%d %H:%M:%S')"
