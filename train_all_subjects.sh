#!/bin/bash
set -e

# Train all subjects: Stage 2 — Cross-Attention Flow Matching + Self-Conditioning
# Stage 1 (ViT VAE) is already trained — reusing existing checkpoints.
#
# Usage:
#   bash train_all_subjects.sh          # train all (01, 02, 05, 07)
#   bash train_all_subjects.sh 01       # train single subject
#   bash train_all_subjects.sh 01 02    # train specific subjects

if [ $# -gt 0 ]; then
    SUBJECTS=("$@")
else
    SUBJECTS=(01 02 05 07)
fi

CONFIG_DIR="src/configs"

echo "=========================================="
echo "Training Pipeline: Stage 2 — XAttn Flow + Self-Conditioning"
echo "Subjects: ${SUBJECTS[*]}"
echo "=========================================="

for sub in "${SUBJECTS[@]}"; do
    echo ""
    echo "=========================================="
    echo " Subject ${sub} — Stage 2: XAttn Flow (noise-robust)"
    echo "=========================================="
    python -m src.train_stage2_xattn_flow \
        --config ${CONFIG_DIR}/subj${sub}/stage2_xattn_flow_vit_vae.yaml
    if [ $? -ne 0 ]; then
        echo "ERROR: Stage 2 failed for subj${sub}!"
        exit 1
    fi

    echo ""
    echo "✅ Subject ${sub} completed!"
done

echo ""
echo "=========================================="
echo "All subjects completed successfully!"
echo "=========================================="
