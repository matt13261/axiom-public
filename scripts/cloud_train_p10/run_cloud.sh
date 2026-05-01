#!/bin/bash
# Deploiement P10.A.bis cloud sur axiom-training-24.
# Convention tmux : session "p10abis" (regle projet cloud training).
set -e
ZONE="europe-west4-a"
VM="axiom-training-24"

echo "=== 1. Demarrer VM ==="
gcloud compute instances start "$VM" --zone="$ZONE" 2>&1 | tail -3

echo "=== 2. Sync code (git pull dans la VM) ==="
gcloud compute ssh "$VM" --zone="$ZONE" --command="
cd ~/axiom 2>/dev/null && git fetch origin && git reset --hard origin/main || \
{ cd ~ && git clone https://github.com/matt13261/axiom-public.git axiom; cd ~/axiom; }
ls cloud_train_p10/ 2>/dev/null || mv scripts/cloud_train_p10 . 2>/dev/null || echo 'cloud_train_p10 path OK'
[ -f data/abstraction/centroides_v2.npz ] || echo 'centroides_v2.npz manquant'
"

echo "=== 3. Lancer dans tmux p10abis ==="
gcloud compute ssh "$VM" --zone="$ZONE" --command="
tmux kill-session -t p10abis 2>/dev/null || true
tmux new-session -d -s p10abis 'cd ~/axiom && python3 -u scripts/cloud_train_p10/eval_p10abis_cloud.py 2>&1 | tee p10abis.log'
sleep 2
tmux ls
"
echo ""
echo "Pour attach: gcloud compute ssh $VM --zone=$ZONE --command='tmux attach -t p10abis'"
