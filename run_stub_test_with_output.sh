#!/bin/bash
#
# Run stub provenance test and save database + documentation
#
# This script runs the stub test and preserves all artifacts for examination.
# The output includes:
#   - Complete test database
#   - Exported documentation (HTML, Markdown, JSON, CSV)
#   - Validation report
#   - All test artifacts (run_info.json, openlineage.jsonl, etc.)
#
# Usage:
#   ./run_stub_test_with_output.sh [output_directory]
#
# Examples:
#   ./run_stub_test_with_output.sh
#   ./run_stub_test_with_output.sh ./my_test_output
#

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to PILATES root directory
cd "$SCRIPT_DIR"

OUTPUT_DIR="${1:-./test_output}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  PILATES Stub Test with Preserved Output                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Working directory: ${GREEN}$(pwd)${NC}"
echo -e "Output directory: ${GREEN}$OUTPUT_DIR${NC}"
echo ""

# Run test with output preservation
echo -e "${BLUE}Running stub provenance test...${NC}"
echo ""

PRESERVE_TEST_OUTPUT="$OUTPUT_DIR" PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH" python tests/test_stub_provenance_flow.py

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Test Complete!                                            ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Test artifacts saved to:${NC} ${GREEN}$OUTPUT_DIR${NC}"
echo ""
echo -e "${YELLOW}Available outputs:${NC}"
echo ""
echo -e "  ${BLUE}📄 Documentation (HTML):${NC}"
echo -e "     open $OUTPUT_DIR/*/documentation/schema.html"
echo ""
echo -e "  ${BLUE}💾 Database:${NC}"
echo -e "     duckdb $OUTPUT_DIR/*/test_database.duckdb"
echo ""
echo -e "  ${BLUE}📋 README:${NC}"
echo -e "     cat $OUTPUT_DIR/*/README.md"
echo ""
echo -e "${YELLOW}Quick commands:${NC}"
echo ""
echo -e "  # View all summary views"
echo -e "  ${GREEN}duckdb $OUTPUT_DIR/activitysim_beam/test_database.duckdb -c \"SELECT * FROM run_summary\"${NC}"
echo ""
echo -e "  # View data lineage"
echo -e "  ${GREEN}duckdb $OUTPUT_DIR/activitysim_beam/test_database.duckdb -c \"SELECT * FROM data_lineage_summary\"${NC}"
echo ""
echo -e "  # Open documentation"
echo -e "  ${GREEN}open $OUTPUT_DIR/activitysim_beam/documentation/schema.html${NC}"
echo ""
