.PHONY: clean requirements dataset features train predictions tests

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = fake_user_detection
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
    $(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
    $(PYTHON_INTERPRETER) -m pip install -r requirements.txt
    $(PYTHON_INTERPRETER) -m pip install -e .

## Make Dataset
dataset:
    $(PYTHON_INTERPRETER) -m fake_user_detection make-dataset

features:
    $(PYTHON_INTERPRETER) -m fake_user_detection build-features

train:
    $(PYTHON_INTERPRETER) -m fake_user_detection train-model

predictions:
    $(PYTHON_INTERPRETER) -m fake_user_detection make-predictions

## Delete all compiled Python files
clean:
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
    flake8 src

## Test python environment is setup correctly
tests:
    $(PYTHON_INTERPRETER) -m pytest tests