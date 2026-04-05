#!/bin/bash

# PharmaSimEnvironment Validation Script
# Checks: HF Space deployment, Docker build, Baseline script

set -e  # Exit on error

echo "========================================"
echo "  PharmaSimEnvironment Validation"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
HF_SPACE_URL="${HF_SPACE_URL:-http://localhost:7860}"
REQUIRED_TASKS=3

# Check 1: HF Space or Local Server
echo "1️⃣  Checking server availability..."
HTTP_CODE=$(curl -o /dev/null -s -w "%{http_code}" "$HF_SPACE_URL" || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✓${NC} Server is live (HTTP 200)"
    echo "   URL: $HF_SPACE_URL"
else
    echo -e "${RED}✗${NC} Server failed (HTTP $HTTP_CODE)"
    echo "   URL: $HF_SPACE_URL"
    echo "   Make sure server is running!"
    exit 1
fi

echo ""

# Check 2: Docker build
echo "2️⃣  Checking Docker build..."
if docker build -t pharma-env-validation . > /tmp/docker_build.log 2>&1; then
    echo -e "${GREEN}✓${NC} Docker build successful"
else
    echo -e "${RED}✗${NC} Docker build failed"
    echo "   Check /tmp/docker_build.log for details"
    tail -20 /tmp/docker_build.log
    exit 1
fi

echo ""

# Check 3: OpenEnv spec compliance
echo "3️⃣  Checking OpenEnv compliance..."

# Check models.py exists and has required classes
if grep -q "class PharmacistAction" models.py && \
   grep -q "class PharmacistObservation" models.py && \
   grep -q "class PharmacistState" models.py; then
    echo -e "${GREEN}✓${NC} Pydantic models defined"
else
    echo -e "${RED}✗${NC} Missing required Pydantic models"
    exit 1
fi

# Check environment.py has reset() and step()
if grep -q "def reset" environment.py && grep -q "def step" environment.py; then
    echo -e "${GREEN}✓${NC} Environment has reset() and step() methods"
else
    echo -e "${RED}✗${NC} Missing reset() or step() methods"
    exit 1
fi

echo ""

# Check 4: Task count
echo "4️⃣  Checking task count..."
TASK_COUNT=$(grep -c "task_._" data/task_cases.json || echo "0")

if [ "$TASK_COUNT" -ge "$REQUIRED_TASKS" ]; then
    echo -e "${GREEN}✓${NC} Found $TASK_COUNT tasks (required: $REQUIRED_TASKS)"
else
    echo -e "${YELLOW}⚠${NC}  Only $TASK_COUNT tasks found (required: $REQUIRED_TASKS)"
fi

echo ""

# Check 5: Baseline inference script
echo "5️⃣  Checking baseline script..."

if [ ! -f "inference.py" ]; then
    echo -e "${RED}✗${NC} inference.py not found"
    exit 1
fi

# Try dry run (without actual LLM calls)
echo "   Running syntax check..."
python -m py_compile inference.py 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Baseline script syntax valid"
else
    echo -e "${RED}✗${NC} Baseline script has syntax errors"
    exit 1
fi

echo ""

# Check 6: README completeness
echo "6️⃣  Checking documentation..."

README_SECTIONS=(
    "Real-World Utility"
    "Quick Start"
    "API"
    "Baseline Scores"
    "Disclaimer"
)

MISSING_SECTIONS=()
for section in "${README_SECTIONS[@]}"; do
    if ! grep -q "$section" README.md; then
        MISSING_SECTIONS+=("$section")
    fi
done

if [ ${#MISSING_SECTIONS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓${NC} README.md complete"
else
    echo -e "${YELLOW}⚠${NC}  README missing sections: ${MISSING_SECTIONS[*]}"
fi

echo ""

# Summary
echo "========================================"
echo "         Validation Summary"
echo "========================================"
echo -e "${GREEN}✓${NC} Server accessible"
echo -e "${GREEN}✓${NC} Docker builds"
echo -e "${GREEN}✓${NC} OpenEnv compliant"
echo -e "${GREEN}✓${NC} Baseline script ready"
echo ""
echo "🎉 All critical checks passed!"
echo ""
echo "Next steps:"
echo "  1. Test with: python client.py"
echo "  2. Run baseline: python inference.py"
echo "  3. Deploy to HF Spaces"
echo "  4. Submit HF Space URL before deadline"
echo ""
echo "========================================"
