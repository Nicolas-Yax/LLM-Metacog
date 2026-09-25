#!/bin/bash
# Train one probe with fit_unsloth.py as a Slurm job.
# Usage: bash fit_unsloth.sh MODEL BATCH_SIZE NB_EPOCHS DATASET LORA_RANK LR WEIGHT_DECAY WARMUP_RATIO PROBE [GPUS=1] [HOURS=3]
# e.g.   bash fit_unsloth.sh Qwen2.5-7B 8 1 math4_categoriesbalanced 8 5e-5 1e-2 1e-1 end
# Adapt the #SBATCH lines and the environment setup to your cluster.

if [ $# -lt 9 ]; then
    sed -n '3p' "$0"
    exit 1
fi
gpu_count=${10:-1}
time=${11:-3}
cpu_count=$((gpu_count * 8))

mkdir -p logs/$1
cat << EOF > send.slurm
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${gpu_count}
#SBATCH --cpus-per-task=${cpu_count}
#SBATCH --time=${time}:00:00
#SBATCH --output=logs/$1/%j.out
#SBATCH --error=logs/$1/%j.out
##SBATCH --account=<your_account>

# Activate your Python environment here, e.g. conda activate metacog

python fit_unsloth.py --model "$1" --batch_size $2 --nb_epochs $3 --dataset "$4" \\
    --lora_rank $5 --lr $6 --weight_decay $7 --warmup_ratio $8 --probe $9
EOF

sbatch send.slurm
