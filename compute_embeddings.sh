#!/bin/bash
# Embed the questions of one dataset (compute_embeddings.py) as a Slurm job.
# Usage: bash compute_embeddings.sh DATASET [BATCH_SIZE=32] [MODEL=Qwen3-Embedding-4B] [GPUS=1] [HOURS=1]
# The embedding model is read from basemodels/unsloth/<MODEL>.
# Adapt the #SBATCH lines and the environment setup to your cluster.

if [ $# -lt 1 ]; then
    sed -n '3p' "$0"
    exit 1
fi
dataset=$1
batch_size=${2:-32}
model=${3:-Qwen3-Embedding-4B}
gpu_count=${4:-1}
time=${5:-1}
cpu_count=$((gpu_count * 8))

mkdir -p logs/embeddings_${dataset}/${model}
cat << EOF > send.slurm
#!/bin/bash
#SBATCH --job-name=embed_${dataset}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${gpu_count}
#SBATCH --cpus-per-task=${cpu_count}
#SBATCH --time=${time}:00:00
#SBATCH --output=logs/embeddings_${dataset}/${model}/%j.out
#SBATCH --error=logs/embeddings_${dataset}/${model}/%j.out
##SBATCH --account=<your_account>

# Activate your Python environment here, e.g. conda activate metacog

python compute_embeddings.py --model ${model} --dataset ${dataset} --batch_size ${batch_size}
EOF

sbatch send.slurm
