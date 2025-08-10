#!/bin/bash
# Development helper script for common tasks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment exists
check_venv() {
    if [ ! -d ".venv" ]; then
        print_warning "Virtual environment not found. Creating it..."
        python3 -m venv .venv
        source .venv/bin/activate
        ./setup_deps.sh
        print_success "Virtual environment created and dependencies installed"
    else
        print_status "Virtual environment found"
    fi
}

# Activate virtual environment
activate_venv() {
    source .venv/bin/activate
    print_status "Virtual environment activated"
}

# Run tests
run_tests() {
    print_status "Running tests..."
    python -m pytest tests/ -v --tb=short
    if [ $? -eq 0 ]; then
        print_success "All tests passed!"
    else
        print_error "Some tests failed"
        exit 1
    fi
}

# Run linting
run_lint() {
    print_status "Running linting with ruff..."
    python -m ruff check pa_core
    if [ $? -eq 0 ]; then
        print_success "Linting passed!"
    else
        print_error "Linting issues found"
        return 1
    fi
}

# Run type checking
run_typecheck() {
    print_status "Running type checking with pyright..."
    python -m pyright
    if [ $? -eq 0 ]; then
        print_success "Type checking passed!"
    else
        print_error "Type checking issues found"
        return 1
    fi
}

# Run full CI pipeline
run_ci() {
    print_status "Running full CI pipeline..."
    run_lint
    run_typecheck
    run_tests
    print_success "CI pipeline completed successfully!"
}

# Quick development setup
dev_setup() {
    print_status "Setting up development environment..."
    check_venv
    activate_venv
    print_success "Development environment ready!"
    print_status "You can now run:"
    echo "  - './dev.sh test' to run tests"
    echo "  - './dev.sh lint' to run linting"
    echo "  - './dev.sh ci' to run full CI pipeline"
    echo "  - './dev.sh demo' to run a quick demo"
}

# Run demo
run_demo() {
    print_status "Running demo with sample configuration..."
    if [ -f "parameters.csv" ] && [ -f "sp500tr_fred_divyield.csv" ]; then
        pa-convert-params parameters.csv params.yml
        python -m pa_core.cli --config params.yml --index sp500tr_fred_divyield.csv
        print_success "Demo completed! Check Outputs.xlsx for results"
    else
        print_warning "Sample data files not found. Please ensure parameters.csv and sp500tr_fred_divyield.csv exist"
    fi
}

# Start dashboard
start_dashboard() {
    print_status "Starting Streamlit dashboard..."
    python -m streamlit run dashboard/app.py
}

# Show usage
usage() {
    echo "Development helper script for Portable Alpha Extension Model"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup     - Initial development environment setup"
    echo "  test      - Run pytest test suite"
    echo "  lint      - Run ruff linting"
    echo "  typecheck - Run pyright type checking"
    echo "  ci        - Run full CI pipeline (lint + typecheck + test)"
    echo "  demo      - Run CLI demo with sample data"
    echo "  dashboard - Start Streamlit dashboard"
    echo "  help      - Show this help message"
    echo ""
}

# Main script logic
case "${1:-}" in
    setup)
        dev_setup
        ;;
    test)
        check_venv && activate_venv && run_tests
        ;;
    lint)
        check_venv && activate_venv && run_lint
        ;;
    typecheck)
        check_venv && activate_venv && run_typecheck
        ;;
    ci)
        check_venv && activate_venv && run_ci
        ;;
    demo)
        check_venv && activate_venv && run_demo
        ;;
    dashboard)
        check_venv && activate_venv && start_dashboard
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        if [ -z "${1:-}" ]; then
            print_warning "No command specified"
        else
            print_error "Unknown command: $1"
        fi
        echo ""
        usage
        exit 1
        ;;
esac
