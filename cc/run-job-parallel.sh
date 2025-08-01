#!/bin/bash
#SBATCH --job-name=streamq-seeds
#SBATCH --account=def-mijungp
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=ammany01@cs.ubc.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -e

ALGORITHM=""
NETWORK=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        -n|--network)
            NETWORK="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 -a|--algorithm <basic|streamq> -n|--network <mlp|kan>"
            echo "Example: $0 --algorithm basic --network mlp"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            echo "Usage: $0 -a|--algorithm <basic|streamq> -n|--network <mlp|kan>"
            exit 1
            ;;
    esac
done

# validation
if [ -z "$ALGORITHM" ] || [ -z "$NETWORK" ]; then
    echo "Error: Both --algorithm and --network are required"
    echo "Usage: $0 -a|--algorithm <basic|streamq> -n|--network <mlp|kan>"
    exit 1
fi

if [[ "$ALGORITHM" != "basic" && "$ALGORITHM" != "streamq" ]]; then
    echo "Error: algorithm must be 'basic' or 'streamq'"
    exit 1
fi

if [[ "$NETWORK" != "mlp" && "$NETWORK" != "kan" ]]; then
    echo "Error: network must be 'mlp' or 'kan'"
    exit 1
fi

# adjust time estimates based on network type
if [ "$NETWORK" = "kan" ]; then
    echo "KAN network detected - using extended time estimate"
    # KAN takes about double the time of MLP
    EXPECTED_HOURS=10
else
    echo "MLP network detected - using standard time estimate"
    EXPECTED_HOURS=6
fi

# job configuration
PROJECT_NAME="streamq-seeds-study"
ENV_NAME="CartPole-v1"
NUM_EPISODES=200
MAX_STEPS=500
NUM_SEEDS=50

# Generate 50 different seeds (algorithm-network specific seed for reproducibility)
SEED_BASE=$(($(echo "${ALGORITHM}-${NETWORK}" | cksum | cut -d' ' -f1) % 10000))
SEEDS=($(python3 -c "import random; random.seed($SEED_BASE); print(' '.join([str(random.randint(1, 100000)) for _ in range($NUM_SEEDS)]))"))

echo "========================================="
echo "StreamQ Seeds Study - Cedar Cluster Job"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODEID"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPU: $SLURM_GRES"
echo "Start Time: $(date)"
echo "Algorithm: $ALGORITHM"
echo "Network: $NETWORK"
echo "Project: $PROJECT_NAME"
echo "Environment: $ENV_NAME"
echo "Episodes per run: $NUM_EPISODES"
echo "Max steps per episode: $MAX_STEPS"
echo "Number of seeds: $NUM_SEEDS"
echo "Expected runtime: ~$EXPECTED_HOURS hours"
echo "Seed base: $SEED_BASE"
echo "========================================="

# Environment setup
echo "Setting up environment..."

# Load Python module
module purge
module load python/3.11.5 scipy-stack

# Set up virtual environment in temporary directory for faster I/O
export VENV_DIR="$SLURM_TMPDIR/py311-streamq"
echo "Creating virtual environment at: $VENV_DIR"

# Create virtual environment
virtualenv --no-download "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Verify Python version
echo "Python version: $(python --version)"
echo "Virtual environment: $VIRTUAL_ENV"

# Copy project to temporary directory for faster I/O
echo "Copying project to temporary directory..."
PROJECT_TMP="$SLURM_TMPDIR/jax-kan"
cp -r "$SLURM_SUBMIT_DIR" "$PROJECT_TMP"
cd "$PROJECT_TMP"

# Install dependencies
echo "Installing dependencies..."
pip install --no-index --upgrade pip

# Install JAX and related packages
echo "Installing JAX ecosystem..."
pip install --no-index jax jaxlib

# Install ML packages available locally
echo "Installing ML packages..."
pip install --no-index numpy scipy matplotlib scikit-learn pandas

# Install packages that may not be available locally
echo "Installing additional packages..."
pip install gymnasium wandb optax chex equinox flax

# Install remaining requirements
echo "Installing remaining requirements..."
pip install -r requirements.txt

# Verify key packages are installed
echo "Verifying installation..."
python -c "import jax; print(f'JAX version: {jax.__version__}')"
python -c "import wandb; print(f'W&B version: {wandb.__version__}')"
python -c "import gymnasium; print(f'Gymnasium version: {gymnasium.__version__}')"

# Check GPU availability
echo "Checking GPU availability..."
python -c "
import jax
print(f'JAX devices: {jax.devices()}')
print(f'JAX default backend: {jax.default_backend()}')
if jax.devices()[0].platform == 'gpu':
    print('✓ GPU detected and available')
else:
    print('⚠ Running on CPU')
"

# W&B setup
echo "Setting up Weights & Biases..."
export WANDB_PROJECT="$PROJECT_NAME"
export WANDB_MODE="online"

if [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: WANDB_API_KEY not set. Make sure to run 'wandb login' or set the API key."
fi

echo "========================================="
echo "Starting training runs for: $ALGORITHM-$NETWORK"
echo "Total runs: $NUM_SEEDS"
echo "========================================="

total_runs=0
successful_runs=0
failed_runs=0

run_experiment() {
    local algorithm=$1
    local network=$2
    local seed=$3
    local run_id=$4
    local total=$5
    
    echo ""
    echo "Run $run_id/$total: $algorithm-$network (seed=$seed)"
    echo "Started at: $(date)"
    
    local run_name="${algorithm}-${network}-seed${seed}"
    
    if python src/wandb_hyperparam_sweep.py \
        --mode single \
        --algorithm "$algorithm" \
        --network "$network" \
        --env "$ENV_NAME" \
        --project "$PROJECT_NAME" \
        --num_episodes "$NUM_EPISODES" \
        --max_steps "$MAX_STEPS" \
        --seed "$seed" \
        --run_name "$run_name"; then
        
        echo "✓ Completed: $run_name"
        return 0
    else
        echo "✗ Failed: $run_name"
        return 1
    fi
}

echo ""
echo "========================================="
echo "Starting $ALGORITHM-$NETWORK experiments"
echo "========================================="

for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    total_runs=$((total_runs + 1))
    
    if run_experiment "$ALGORITHM" "$NETWORK" "$seed" "$total_runs" "$NUM_SEEDS"; then
        successful_runs=$((successful_runs + 1))
    else
        failed_runs=$((failed_runs + 1))
    fi
    
    if (( total_runs % 10 == 0 )); then
        echo ""
        echo "Progress Update:"
        echo "  Completed runs: $total_runs/$NUM_SEEDS"
        echo "  Successful: $successful_runs"
        echo "  Failed: $failed_runs"
        echo "  Success rate: $(( successful_runs * 100 / total_runs ))%"
        echo ""
    fi
done

echo ""
echo "Copying results back to submission directory..."
if [ -d "results" ]; then
    cp -r results "$SLURM_SUBMIT_DIR/"
fi

echo ""
echo "========================================="
echo "Job Summary"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Algorithm-Network: $ALGORITHM-$NETWORK"
echo "Total runs: $total_runs"
echo "Successful runs: $successful_runs"
echo "Failed runs: $failed_runs"
echo "Overall success rate: $(( successful_runs * 100 / total_runs ))%"
echo "End time: $(date)"
echo "Project: $PROJECT_NAME"

echo ""
echo "Weights & Biases:"
echo "  Project: $PROJECT_NAME"
echo "  View results at: https://wandb.ai/$USER/$PROJECT_NAME"

echo "========================================="
echo "Job completed!"
echo "========================================="
