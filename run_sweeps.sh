#!/bin/bash

set -e

PROJECT_NAME="streamq-hyperopt"
COUNT=50
ENV="CartPole-v1"
PYTHON_CMD="/home/ammany01/.pyenv/versions/env-3.11.13/bin/python"
WANDB_CMD="/home/ammany01/.pyenv/versions/env-3.11.13/bin/wandb"

show_help() {
    cat << EOF
StreamQ Hyperparameter Sweep Runner

Usage: $0 [OPTIONS] COMMAND

COMMANDS:
    create-all      Create sweeps for all 4 algorithm/network combinations
    create-basic    Create sweeps for basic streaming algorithms only (MLP and KAN)
    create-streamq  Create sweeps for StreamQ(Lambda) algorithms only (MLP and KAN)
    run-sweep ID    Run a specific sweep with the given ID
    single-test     Run single experiments to test the setup
    list-sweeps     List active sweeps (requires wandb CLI)

OPTIONS:
    -p, --project NAME    Wandb project name (default: $PROJECT_NAME)
    -c, --count N         Number of runs per sweep (default: $COUNT)
    -e, --env ENV         Environment name (default: $ENV)
    -h, --help           Show this help

EXAMPLES:
    # Create all sweeps
    $0 create-all

    # Create only basic streaming sweeps
    $0 create-basic

    # Run a specific sweep (replace SWEEP_ID with actual ID)
    $0 run-sweep USERNAME/PROJECT/SWEEP_ID

    # Test with single experiments
    $0 single-test

    # Create sweeps for specific project
    $0 -p "my-streamq-project" create-all
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--project)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -c|--count)
            COUNT="$2"
            shift 2
            ;;
        -e|--env)
            ENV="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            COMMAND="$1"
            SWEEP_ID="$2"
            break
            ;;
    esac
done

check_wandb() {
    if ! command -v "$WANDB_CMD" &> /dev/null; then
        echo "Error: wandb CLI not found. Install with: pip install wandb"
        exit 1
    fi
    
    # Check if logged in by trying to access status
    if ! $WANDB_CMD status 2>/dev/null | grep -q "Logged in"; then
        echo "Warning: Not logged into wandb. Run 'wandb login' first."
        echo "Continuing anyway (may prompt for login)..."
    fi
}

create_all_sweeps() {
    echo "Creating sweeps for all 4 algorithm/network combinations..."
    echo "Project: $PROJECT_NAME"
    echo "Environment: $ENV"
    echo ""
    
    # Only the four core combinations:
    # 1. Basic Streaming with MLP
    # 2. Basic Streaming with KAN
    # 3. StreamQ(Lambda) with MLP
    # 4. StreamQ(Lambda) with KAN
    
    combinations=(
        "basic mlp"
        "basic kan"
        "streamq mlp"
        "streamq kan"
    )
    
    sweep_ids=()
    
    for combo in "${combinations[@]}"; do
        read -r algorithm network <<< "$combo"
        echo "Creating sweep: $algorithm-$network"
        
        output=$($PYTHON_CMD src/wandb_hyperparam_sweep.py \
            --mode sweep \
            --algorithm "$algorithm" \
            --network "$network" \
            --env "$ENV" \
            --project "$PROJECT_NAME" 2>&1)
        
        # Extract the short ID from the output
        sweep_id=$(echo "$output" | grep "Created sweep" | grep -o 'sweep [a-zA-Z0-9]\+' | cut -d' ' -f2 || echo "")
        
        if [[ -n "$sweep_id" ]]; then
            sweep_ids+=("$sweep_id")
            echo "✓ Created sweep: $sweep_id"
        else
            echo "✗ Failed to create sweep for $algorithm-$network"
            echo "Output: $output"
        fi
        echo ""
    done
    
    echo "Summary of created sweeps:"
    printf '%s\n' "${sweep_ids[@]}"
    
    echo ""
    echo "To run all sweeps in parallel, use:"
    for id in "${sweep_ids[@]}"; do
        echo "  nohup $0 run-sweep $id &"
    done
}

create_basic_sweeps() {
    echo "Creating sweeps for basic streaming algorithms..."
    
    networks=("mlp" "kan")
    for network in "${networks[@]}"; do
        echo "Creating basic-$network sweep..."
        $PYTHON_CMD src/wandb_hyperparam_sweep.py \
            --mode sweep \
            --algorithm basic \
            --network "$network" \
            --env "$ENV" \
            --project "$PROJECT_NAME"
        echo ""
    done
}

create_streamq_sweeps() {
    echo "Creating sweeps for StreamQ(Lambda) algorithms..."
    
    networks=("mlp" "kan")
    for network in "${networks[@]}"; do
        echo "Creating streamq-$network sweep..."
        $PYTHON_CMD src/wandb_hyperparam_sweep.py \
            --mode sweep \
            --algorithm streamq \
            --network "$network" \
            --env "$ENV" \
            --project "$PROJECT_NAME"
        echo ""
    done
}

run_sweep() {
    local sweep_id="$1"
    if [[ -z "$sweep_id" ]]; then
        echo "Error: Sweep ID required"
        echo "Usage: $0 run-sweep SWEEP_ID"
        exit 1
    fi
    
    # Define the entity name - this should be your W&B username or organization
    ENTITY_NAME="kan_rl"
    
    # Set the path to the main Python script that should be run
    MAIN_SCRIPT="src/wandb_hyperparam_sweep.py"
    
    # Check if sweep_id already has entity/project prefix
    if [[ ! "$sweep_id" =~ "/" ]]; then
        # No slashes found, need to add entity and project
        echo "Running sweep with ID: $ENTITY_NAME/$PROJECT_NAME/$sweep_id"
        echo "Count: $COUNT runs"
        echo "Using script: $MAIN_SCRIPT"
        echo ""
        
        # Use wandb agent directly with full sweep ID
        $WANDB_CMD agent "$ENTITY_NAME/$PROJECT_NAME/$sweep_id" --count "$COUNT"
    else
        # Full path provided
        echo "Running sweep: $sweep_id"
        echo "Count: $COUNT runs"
        echo "Using script: $MAIN_SCRIPT"
        echo ""
        
        # Extract components from the full path
        IFS='/' read -ra PARTS <<< "$sweep_id"
        ENTITY=${PARTS[0]}
        PROJECT=${PARTS[1]}
        SWEEP=${PARTS[2]}
        
        # Use wandb agent directly with full sweep ID
        $WANDB_CMD agent "$ENTITY/$PROJECT/$SWEEP" --count "$COUNT"
    fi
}

single_test() {
    echo "Running single test experiments for the four core combinations..."
    echo ""
    
    # Only the four core combinations
    combinations=(
        "basic mlp"
        "basic kan"
        "streamq mlp"
        "streamq kan"
    )
    
    for combo in "${combinations[@]}"; do
        read -r algorithm network <<< "$combo"
        echo "Testing $algorithm-$network..."
        $PYTHON_CMD src/wandb_hyperparam_sweep.py \
            --mode single \
            --algorithm "$algorithm" \
            --network "$network" \
            --env "$ENV" \
            --project "$PROJECT_NAME-test"
        echo ""
    done
}

list_sweeps() {
    echo "Active sweeps in project $PROJECT_NAME:"
    echo "Note: To see active sweeps, please visit: https://wandb.ai/dashboard/sweeps"
    echo "Your project URL should be: https://wandb.ai/<USERNAME>/$PROJECT_NAME/sweeps"
    echo ""
    echo "You can also try running a direct command to get the sweep IDs:"
    echo "  $PYTHON_CMD -c \"import wandb; print(wandb.api.list_sweeps('$PROJECT_NAME'))\""
}

check_wandb

case "$COMMAND" in
    create-all)
        create_all_sweeps
        ;;
    create-basic)
        create_basic_sweeps
        ;;
    create-streamq)
        create_streamq_sweeps
        ;;
    run-sweep)
        run_sweep "$SWEEP_ID"
        ;;
    single-test)
        single_test
        ;;
    list-sweeps)
        list_sweeps
        ;;
    "")
        echo "Error: Command required"
        echo ""
        show_help
        exit 1
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'"
        echo ""
        show_help
        exit 1
        ;;
esac
