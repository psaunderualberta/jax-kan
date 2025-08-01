#!/bin/bash
echo "========================================="
echo "Cedar Cluster Setup Verification"
echo "========================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_counter=0
passed_checks=0
failed_checks=0

run_check() {
    local check_name="$1"
    local check_command="$2"
    local success_msg="$3"
    local failure_msg="$4"
    
    check_counter=$((check_counter + 1))
    echo ""
    echo -e "${YELLOW}Check $check_counter: $check_name${NC}"
    echo "----------------------------------------"
    
    if eval "$check_command" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED: $success_msg${NC}"
        passed_checks=$((passed_checks + 1))
    else
        echo -e "${RED}✗ FAILED: $failure_msg${NC}"
        failed_checks=$((failed_checks + 1))
    fi
    echo "----------------------------------------"
}

echo "This script verifies your Cedar cluster setup."
echo "Run this on a Cedar login node before submitting jobs."
echo ""

# Check 1: Verify we're on Cedar
run_check "Cedar Cluster Detection" \
    "hostname | grep -q cedar" \
    "Running on Cedar cluster" \
    "Not running on Cedar cluster - this script should be run on Cedar"

# Check 2: Check quotas
echo ""
echo -e "${YELLOW}Storage Quotas:${NC}"
quota 2>/dev/null || echo "Quota command not available"

# Check 3: Check SLURM account
if [ -n "$SLURM_ACCOUNT" ]; then
    echo -e "${GREEN}SLURM_ACCOUNT is set to: $SLURM_ACCOUNT${NC}"
else
    echo -e "${YELLOW}SLURM_ACCOUNT not set. Consider adding to ~/.bashrc:${NC}"
    echo "export SLURM_ACCOUNT=def-mijungp"
    echo "export SBATCH_ACCOUNT=\$SLURM_ACCOUNT"
    echo "export SALLOC_ACCOUNT=\$SLURM_ACCOUNT"
fi

# Check 4: Python module availability
run_check "Python 3.11 Module" \
    "module avail python/3.11 2>&1 | grep -q 3.11" \
    "Python 3.11 module is available" \
    "Python 3.11 module not found - check available versions with 'module avail python'"

# Check 5: SciPy stack module
run_check "SciPy Stack Module" \
    "module avail scipy-stack 2>&1 | grep -q scipy-stack" \
    "SciPy stack module is available" \
    "SciPy stack module not found"

# Check 6: Project directory access
PROJECT_DIR="$HOME/projects"
run_check "Project Directory Access" \
    "test -d $PROJECT_DIR" \
    "Project directory exists at $PROJECT_DIR" \
    "Project directory not found at $PROJECT_DIR"

if [ -d "$PROJECT_DIR" ]; then
    echo ""
    echo "Available project directories:"
    ls -la "$PROJECT_DIR" 2>/dev/null || echo "Cannot list project directories"
fi

# Check 7: Git availability
run_check "Git Availability" \
    "command -v git" \
    "Git is available" \
    "Git not found - needed for code synchronization"

# Check 8: SSH key setup for git
run_check "SSH Key Setup" \
    "test -f ~/.ssh/id_rsa.pub" \
    "SSH public key found" \
    "SSH key not found - consider setting up SSH keys for git"

# Check 9: W&B configuration
if [ -f ~/.netrc ] && grep -q "machine api.wandb.ai" ~/.netrc; then
    echo -e "${GREEN}✓ W&B credentials found in ~/.netrc${NC}"
elif [ -n "$WANDB_API_KEY" ]; then
    echo -e "${GREEN}✓ WANDB_API_KEY environment variable is set${NC}"
else
    echo -e "${YELLOW}⚠ W&B not configured. Run 'wandb login' to set up authentication${NC}"
fi

# Check 10: Test module loading
echo ""
echo -e "${YELLOW}Testing module loading...${NC}"
if module purge && module load python/3.11.5 scipy-stack; then
    echo -e "${GREEN}✓ Modules loaded successfully${NC}"
    echo "Python version: $(python --version 2>&1)"
    echo "Python path: $(which python)"
    
    # Test virtualenv
    if command -v virtualenv >/dev/null 2>&1; then
        echo -e "${GREEN}✓ virtualenv is available${NC}"
    else
        echo -e "${RED}✗ virtualenv not available${NC}"
    fi
    
    module purge
else
    echo -e "${RED}✗ Failed to load modules${NC}"
fi

# Check 11: Scratch space
if [ -n "$SCRATCH" ] && [ -d "$SCRATCH" ]; then
    echo -e "${GREEN}✓ Scratch directory available at: $SCRATCH${NC}"
    df -h "$SCRATCH" 2>&1 | head -2
else
    echo -e "${YELLOW}⚠ Scratch directory not set or not accessible${NC}"
fi

# Summary
echo ""
echo "========================================="
echo "Setup Verification Summary"
echo "========================================="
echo -e "Total checks: $check_counter"
echo -e "${GREEN}Passed: $passed_checks${NC}"
echo -e "${RED}Failed: $failed_checks${NC}"

if [ $failed_checks -eq 0 ]; then
    echo -e "${GREEN}"
    echo "🎉 Cedar setup looks good! You can proceed with job submission."
    echo -e "${NC}"
else
    echo -e "${YELLOW}"
    echo "⚠ Some checks failed. Review the issues above before submitting jobs."
    echo -e "${NC}"
fi

echo ""
echo "Additional recommendations:"
echo "1. Set up environment variables in ~/.bashrc:"
echo "   export SLURM_ACCOUNT=def-mijungp"
echo "   export project=~/projects/def-mijungp/\$USER"
echo ""
echo "2. Configure W&B authentication:"
echo "   wandb login"
echo ""
echo "3. Test job submission with a small test job first"
echo ""
echo "Useful commands:"
echo "- Check job queue: squeue -u \$USER"
echo "- Cancel job: scancel JOB_ID"
echo "- Job info: sinfo"
echo "- Account info: sshare -U \$USER"

echo "========================================="
