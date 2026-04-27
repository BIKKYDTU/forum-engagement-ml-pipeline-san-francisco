#!/bin/bash
### COMMON SETUP; DO NOT MODIFY ###
set -e

# --- CONFIGURE THIS SECTION ---
run_all_tests() {
    echo "Running all tests..."
    if [ -d "/eval_assets/tests" ]; then
        TESTS_DIR="/eval_assets/tests"
    else
        TESTS_DIR="/app/tests"
    fi
    PIPELINE_REPO_ROOT=/app python3 -m pytest "$TESTS_DIR" -v --tb=short --no-header || true
}
# --- END CONFIGURATION SECTION ---

### COMMON EXECUTION; DO NOT MODIFY ###
run_all_tests
