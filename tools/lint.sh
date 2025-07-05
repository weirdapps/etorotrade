#!/bin/bash
# Run all linters individually
# Usage: ./scripts/lint.sh [fix]

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

FIX_MODE=0
if [ "$1" == "fix" ]; then
    FIX_MODE=1
    echo -e "${YELLOW}Running in auto-fix mode${NC}"
fi

PATHS="yahoofinance trade.py tests"

# Run black
if [ $FIX_MODE -eq 1 ]; then
    echo -e "\n${YELLOW}Running black to format code...${NC}"
    black $PATHS
else
    echo -e "\n${YELLOW}Checking formatting with black...${NC}"
    black --check $PATHS || { echo -e "${RED}Formatting issues found. Run './tools/lint.sh fix' to fix.${NC}"; exit 1; }
fi

# Run isort
if [ $FIX_MODE -eq 1 ]; then
    echo -e "\n${YELLOW}Running isort to sort imports...${NC}"
    isort $PATHS
else
    echo -e "\n${YELLOW}Checking import sorting with isort...${NC}"
    isort --check-only $PATHS || { echo -e "${RED}Import sorting issues found. Run './tools/lint.sh fix' to fix.${NC}"; exit 1; }
fi

# Run flake8
echo -e "\n${YELLOW}Checking code with flake8...${NC}"
flake8 $PATHS || { echo -e "${RED}Linting issues found. Please fix them manually.${NC}"; exit 1; }

# Run mypy
echo -e "\n${YELLOW}Checking types with mypy...${NC}"
mypy $PATHS || { echo -e "${RED}Type checking issues found. Please fix them manually.${NC}"; exit 1; }

# Success
echo -e "\n${GREEN}All code quality checks passed!${NC}"