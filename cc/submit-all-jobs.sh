#!/bin/bash

# Parallel job submission script for StreamQ seeds study
# Submits 4 separate jobs (one for each algorithm-network combination)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================="
echo "StreamQ Parallel Job Submission - Cedar"
echo "========================================="

if [ ! -f "run-job-parallel.sh" ]; then
    echo -e "${RED}Error: run-job-parallel.sh not found${NC}"
    exit 1
fi

chmod +x run-job-parallel.sh

declare -a COMBINATIONS=(
    "basic mlp"
    "basic kan" 
    "streamq mlp"
    "streamq kan"
)

echo "Submitting 4 parallel jobs..."
echo "Each job will run 50 seeds for one algorithm-network combination"
echo ""

declare -a JOB_IDS=()

for combo in "${COMBINATIONS[@]}"; do
    read -r algorithm network <<< "$combo"
    
    echo -e "${BLUE}Submitting: $algorithm-$network${NC}"
    
    job_output=$(sbatch --job-name="streamq-${algorithm}-${network}" \
                       run-job-parallel.sh \
                       --algorithm "$algorithm" \
                       --network "$network" 2>&1)
    
    if echo "$job_output" | grep -q "Submitted batch job"; then
        job_id=$(echo "$job_output" | grep -o '[0-9]\+$')
        JOB_IDS+=("$job_id")
        echo -e "${GREEN}✓ Submitted job $job_id for $algorithm-$network${NC}"
    else
        echo -e "${RED}✗ Failed to submit $algorithm-$network${NC}"
        echo "Error: $job_output"
    fi
    
    sleep 2
done

echo ""
echo "========================================="
echo "Submission Summary"
echo "========================================="

if [ ${#JOB_IDS[@]} -eq 0 ]; then
    echo -e "${RED}No jobs were submitted successfully${NC}"
    exit 1
fi

echo -e "${GREEN}Successfully submitted ${#JOB_IDS[@]} jobs:${NC}"
for i in "${!JOB_IDS[@]}"; do
    combo="${COMBINATIONS[$i]}"
    job_id="${JOB_IDS[$i]}"
    echo "  Job $job_id: $combo"
done

echo ""
echo -e "${YELLOW}Monitoring Commands:${NC}"
echo "Check all jobs:     squeue -u \$USER"
echo "Monitor progress:   ./monitor-job.sh status"
echo "View logs:          ./monitor-job.sh logs JOB_ID"
echo "Cancel all jobs:    scancel ${JOB_IDS[*]}"

echo ""
echo -e "${YELLOW}Expected Completion Times:${NC}"
echo "MLP jobs (basic-mlp, streamq-mlp):  ~4 hours each"
echo "KAN jobs (basic-kan, streamq-kan):  ~8 hours each"

echo ""
echo -e "${YELLOW}Results:${NC}"
echo "W&B Project: https://wandb.ai/\$USER/streamq-seeds-study"
echo "Total experiments: 200 (4 combinations × 50 seeds each)"

echo ""
echo -e "${BLUE}Job IDs for reference:${NC}"
echo "export STREAMQ_JOBS=\"${JOB_IDS[*]}\""

echo "========================================="
echo -e "${GREEN}All jobs submitted successfully!${NC}"
echo "Use 'squeue -u \$USER' to monitor progress"
echo "========================================="
