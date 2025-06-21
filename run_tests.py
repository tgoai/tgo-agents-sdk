#!/usr/bin/env python3
"""
Test runner script for the multi-agent system.

This script provides convenient ways to run different types of tests
with various configurations and reporting options.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    if description:
        print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description or 'Command'} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description or 'Command'} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Command not found: {cmd[0]}")
        return False


def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests."""
    cmd = ["python3", "-m", "pytest", "tests/", "-m", "unit or not integration"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html"])
    
    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose=False):
    """Run integration tests."""
    cmd = ["python3", "-m", "pytest", "tests/test_integration.py", "-m", "integration"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Integration Tests")


def run_all_tests(verbose=False, coverage=False):
    """Run all tests."""
    cmd = ["python3", "-m", "pytest", "tests/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html"])
    
    return run_command(cmd, "All Tests")


def run_specific_test(test_path, verbose=False):
    """Run a specific test file or test function."""
    cmd = ["python3", "-m", "pytest", test_path]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, f"Specific Test: {test_path}")


def run_linting():
    """Run code linting."""
    success = True
    
    # Run black formatting check
    if not run_command(["python3", "-m", "black", "--check", "src/", "tests/"], "Black Formatting Check"):
        success = False

    # Run ruff linting
    if not run_command(["python3", "-m", "ruff", "check", "src/", "tests/"], "Ruff Linting"):
        success = False
    
    return success


def run_type_checking():
    """Run type checking."""
    return run_command(["python3", "-m", "pyright", "src/"], "Type Checking")


def fix_formatting():
    """Fix code formatting."""
    success = True
    
    # Run black formatting
    if not run_command(["python3", "-m", "black", "src/", "tests/"], "Black Formatting"):
        success = False

    # Run ruff auto-fix
    if not run_command(["python3", "-m", "ruff", "check", "--fix", "src/", "tests/"], "Ruff Auto-fix"):
        success = False
    
    return success


def run_quality_checks():
    """Run all quality checks."""
    print("\n🔍 Running Quality Checks...")
    
    success = True
    
    if not run_linting():
        success = False
    
    if not run_type_checking():
        success = False
    
    return success


def install_dependencies():
    """Install test dependencies."""
    return run_command(["python3", "-m", "pip", "install", "-e", ".[dev]"], "Installing Dependencies")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test runner for multi-agent system")
    
    parser.add_argument(
        "command",
        choices=[
            "unit", "integration", "all", "lint", "type-check", 
            "format", "quality", "install", "specific"
        ],
        help="Test command to run"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Run with coverage reporting"
    )
    
    parser.add_argument(
        "-t", "--test-path",
        help="Specific test path (for 'specific' command)"
    )
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    if project_root.name != "tgo-agent-coordinator":
        print("⚠️  Warning: Not in project root directory")
    
    success = True
    
    if args.command == "unit":
        success = run_unit_tests(args.verbose, args.coverage)
    
    elif args.command == "integration":
        success = run_integration_tests(args.verbose)
    
    elif args.command == "all":
        success = run_all_tests(args.verbose, args.coverage)
    
    elif args.command == "specific":
        if not args.test_path:
            print("❌ --test-path is required for 'specific' command")
            sys.exit(1)
        success = run_specific_test(args.test_path, args.verbose)
    
    elif args.command == "lint":
        success = run_linting()
    
    elif args.command == "type-check":
        success = run_type_checking()
    
    elif args.command == "format":
        success = fix_formatting()
    
    elif args.command == "quality":
        success = run_quality_checks()
    
    elif args.command == "install":
        success = install_dependencies()
    
    if success:
        print(f"\n🎉 {args.command.title()} completed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 {args.command.title()} failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
