#!/bin/bash
set -e
apt update
apt install -y git

pip install poetry==1.7.0

poetry install