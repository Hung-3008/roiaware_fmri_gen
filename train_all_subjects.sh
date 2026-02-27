#!/bin/bash
set -e

# Train all subjects: Stage 1 (ViT VAE) → Stage 2 (Masked Brain Modeling)
# Models: ViT VAE + Masked Brain DiT
#
# Usage:
#   bash train_all_subjects.sh          # train all
#   bash train_all_subjects.sh 01       # train single subject
#   bash train_all_subjects.sh 01 02    # train specific subjects

if [ $# -gt 0 ]; then
    SUBJECTS=("$@")
else
    SUBJECTS=(05 07)
    #SUBJECTS=(02)
fi

CONFIG_DIR="src/configs"

echo "=========================================="
echo "Training Pipeline: ViT VAE + Masked DiT"
echo "Subjects: ${SUBJECTS[*]}"
echo "=========================================="

for sub in "${SUBJECTS[@]}"; do
    echo ""
    echo "=========================================="
    echo " Subject ${sub} — Stage 1: ViT VAE"
    echo "=========================================="
    python -m src.train_stage1_mlp_vae \
        --config ${CONFIG_DIR}/subj${sub}/stage1_vit_vae.yaml
    if [ $? -ne 0 ]; then
        echo "ERROR: Stage 1 failed for subj${sub}!"
        exit 1
    fi

    echo ""
    echo "=========================================="
    echo " Subject ${sub} — Stage 2: Masked Brain DiT"
    echo "=========================================="
    python -m src.train_stage2_masked \
        --config ${CONFIG_DIR}/subj${sub}/stage2_masked_vit_vae.yaml
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
