#!/bin/bash
# Monitore le pilot run sur axiom-training.
# Usage : bash scripts/cloud_monitor.sh
ZONE="europe-west4-a"
VM_NAME="axiom-training"

echo "=== AXIOM Cloud Monitor — $(date '+%Y-%m-%d %H:%M:%S') ==="
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
echo '--- Process ---'
ps aux | grep train_cloud | grep -v grep || echo 'PROCESS ABSENT'

echo ''
echo '--- PID ---'
cat ~/pilot.pid 2>/dev/null || echo 'pilot.pid absent'

echo ''
echo '--- Derniers logs ---'
LOGFILE=\$(ls -t ~/axiom/logs/pilot_*.log 2>/dev/null | head -1)
if [ -n \"\$LOGFILE\" ]; then
    echo \"Fichier: \$LOGFILE\"
    tail -20 \"\$LOGFILE\"
else
    echo 'Aucun log trouvé'
fi

echo ''
echo '--- Disk ---'
df -h ~/axiom/ | tail -1
ls -lh ~/axiom/blueprint_v2_pilot.pkl 2>/dev/null || echo 'Blueprint pas encore créé'
ls -lh ~/axiom/data/strategy/ 2>/dev/null | tail -5
" 2>&1
