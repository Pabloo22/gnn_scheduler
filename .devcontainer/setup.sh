#!/bin/bash
set -e
apt install git-all

pip install poetry==1.7.0

poetry install