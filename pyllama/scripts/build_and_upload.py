#!/usr/bin/env python3
"""
Build and upload pyllama-server to PyPI.

Usage:
    python scripts/build_and_upload.py --build
    python scripts/build_and_upload.py --upload
    python scripts/build_and_upload.py --all
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent


def clean():
    """Clean build artifacts."""
    print("Cleaning build artifacts...")
    
    dirs_to_remove = [
        PROJECT_DIR / "build",
        PROJECT_DIR / "dist",
        PROJECT_DIR / "*.egg-info",
        PROJECT_DIR / "pyllama" / "*.egg-info",
        PROJECT_DIR / "pyllama" / "__pycache__",
    ]
    
    for pattern in dirs_to_remove:
        for path in Path(".").glob(str(pattern).replace(str(PROJECT_DIR) + "/", "")):
            if path.exists():
                print(f"  Removing: {path}")
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    
    for pycache in PROJECT_DIR.rglob("__pycache__"):
        print(f"  Removing: {pycache}")
        shutil.rmtree(pycache)
    
    print("Clean complete")


def build():
    """Build the package."""
    print("Building package...")
    
    os.chdir(PROJECT_DIR)
    
    result = subprocess.run(
        [sys.executable, "-m", "build"],
        check=False
    )
    
    if result.returncode != 0:
        print("Build failed!")
        return False
    
    print("Build complete!")
    print(f"\nBuilt packages:")
    for f in (PROJECT_DIR / "dist").glob("*"):
        print(f"  {f}")
    
    return True


def upload(test=False):
    """Upload to PyPI."""
    print("Uploading to PyPI...")
    
    os.chdir(PROJECT_DIR)
    
    dist_dir = PROJECT_DIR / "dist"
    if not dist_dir.exists() or not list(dist_dir.glob("*")):
        print("No dist files found. Run --build first.")
        return False
    
    cmd = [sys.executable, "-m", "twine", "upload"]
    
    if test:
        cmd.extend(["--repository", "testpypi"])
    
    cmd.append(str(dist_dir / "*"))
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print("Upload failed!")
        return False
    
    print("Upload complete!")
    return True


def check():
    """Check the package with twine."""
    print("Checking package...")
    
    os.chdir(PROJECT_DIR)
    
    result = subprocess.run(
        [sys.executable, "-m", "twine", "check", "dist/*"],
        check=False
    )
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Build and upload pyllama-server")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--build", action="store_true", help="Build the package")
    parser.add_argument("--check", action="store_true", help="Check the package")
    parser.add_argument("--upload", action="store_true", help="Upload to PyPI")
    parser.add_argument("--test", action="store_true", help="Upload to TestPyPI")
    parser.add_argument("--all", action="store_true", help="Clean, build, check and upload")
    
    args = parser.parse_args()
    
    if not any([args.clean, args.build, args.check, args.upload, args.all]):
        parser.print_help()
        return 0
    
    if args.clean or args.all:
        clean()
    
    if args.build or args.all:
        if not build():
            return 1
    
    if args.check or args.all:
        if not check():
            return 1
    
    if args.upload or args.all:
        if not upload(test=args.test):
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())