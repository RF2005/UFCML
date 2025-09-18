#!/usr/bin/env bash
set -euxo pipefail

# Ensure Git LFS assets (model file) are available before installing deps
git lfs install
git lfs pull

pip install --upgrade pip
pip install -r requirements.txt
