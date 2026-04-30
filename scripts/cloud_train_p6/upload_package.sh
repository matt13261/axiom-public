#!/bin/bash
# Crée un ZIP autonome du repo AXIOM pour upload vers la VM Google Cloud.
# Usage : bash scripts/cloud_train_p6/upload_package.sh
set -e

ZONE="europe-west4-a"
VM_NAME="axiom-training"
ZIP_NAME="axiom_cloud_$(date +%Y%m%d_%H%M%S).zip"

echo "=== Création du package cloud ==="

zip -r "$ZIP_NAME" \
    ai/ \
    abstraction/ \
    engine/ \
    config/ \
    data/abstraction/centroides_v2.npz \
    scripts/cloud_train_p6/ \
    -x "**/__pycache__/*" \
    -x "**/*.pyc" \
    -x "**/*.pt"

echo "OK Package : $ZIP_NAME"
echo "   Taille  : $(du -h "$ZIP_NAME" | cut -f1)"
echo ""
echo "=== Upload vers la VM ==="
gcloud compute scp "$ZIP_NAME" "${VM_NAME}:~/" --zone="$ZONE"
echo "OK Upload terminé"
echo ""
echo "=== Décompression sur la VM ==="
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
    unzip -q ~/$ZIP_NAME -d ~/axiom && echo 'OK Décompression terminée'
"
echo ""
echo "Prêt. Pour se connecter :"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE"
