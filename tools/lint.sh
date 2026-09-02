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

# Run ruff (formatter; also sorts imports via the "I" rule in ruff check below)
if [ $FIX_MODE -eq 1 ]; then
    echo -e "\n${YELLOW}Running ruff to format code...${NC}"
    ruff format $PATHS
else
    echo -e "\n${YELLOW}Checking formatting with ruff...${NC}"
    ruff format --check $PATHS || { echo -e "${RED}Formatting issues found. Run './tools/lint.sh fix' to fix.${NC}"; exit 1; }
fi

# Run ruff check (lint + import order)
if [ $FIX_MODE -eq 1 ]; then
    echo -e "\n${YELLOW}Running ruff check --fix...${NC}"
    ruff check --fix $PATHS
else
    echo -e "\n${YELLOW}Linting with ruff...${NC}"
    ruff check $PATHS || { echo -e "${RED}Linting issues found. Run './tools/lint.sh fix' to fix what is fixable.${NC}"; exit 1; }
fi

# Run mypy
echo -e "\n${YELLOW}Checking types with mypy...${NC}"
mypy $PATHS || { echo -e "${RED}Type checking issues found. Please fix them manually.${NC}"; exit 1; }

# Success
echo -e "\n${GREEN}All code quality checks passed!${NC}"
