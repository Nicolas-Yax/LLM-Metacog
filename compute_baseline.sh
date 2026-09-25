#!/bin/bash
# Verbalised-confidence baseline of one base model on one dataset (compute_baseline.py) as a Slurm job.
# Usage: bash compute_baseline.sh MODEL BATCH_SIZE DATASET [GPUS=1] [HOURS=2]
# e.g.   bash compute_baseline.sh Qwen2.5-7B 16 math4_categoriesbalanced
# Adapt the #SBATCH lines and the environment setup to your cluster.

if [ $# -lt 3 ]; then
    sed -n '3p' "$0"
    exit 1
fi
gpu_count=${4:-1}
time=${5:-2}
cpu_count=$((gpu_count * 8))

mkdir -p logs/baseline_$1/$3
cat << EOF > send.slurm
#!/bin/bash
#SBATCH --job-name=baseline_$1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${gpu_count}
#SBATCH --cpus-per-task=${cpu_count}
#SBATCH --time=${time}:00:00
#SBATCH --output=logs/baseline_$1/$3/%j.out
#SBATCH --error=logs/baseline_$1/$3/%j.out
##SBATCH --account=<your_account>

# Activate your Python environment here, e.g. conda activate metacog

python compute_baseline.py --model $1 --batch_size $2 --dataset $3
EOF

sbatch send.slurm
