#!/bin/bash
# Evaluate one trained run on a dataset (compute_trained_performance.py) as a Slurm job.
# Usage: bash compute_trained_performance.sh PROJECT RUN_NAME EVAL_DATASET LAST_ONLY(0|1) [GPUS=1] [HOURS=3]
# e.g.   bash compute_trained_performance.sh default <run folder name> all 1
# EVAL_DATASET can be a dataset name, a comma-separated list or "all".
# Adapt the #SBATCH lines and the environment setup to your cluster.

if [ $# -lt 4 ]; then
    sed -n '3p' "$0"
    exit 1
fi
gpu_count=${5:-1}
time=${6:-3}
cpu_count=$((gpu_count * 8))
last=""
if [ "$4" -eq 1 ]; then
    last="--last_only"
fi

mkdir -p logs/$3/$2
cat << EOF > send.slurm
#!/bin/bash
#SBATCH --job-name=eval_$3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${gpu_count}
#SBATCH --cpus-per-task=${cpu_count}
#SBATCH --time=${time}:00:00
#SBATCH --output=logs/$3/$2/%j.out
#SBATCH --error=logs/$3/$2/%j.out
##SBATCH --account=<your_account>

# Activate your Python environment here, e.g. conda activate metacog

python compute_trained_performance.py --project $1 --checkpoint_name $2 --eval_dataset $3 ${last} --overwrite
EOF

sbatch send.slurm
