.PHONY: clean dataset features train predictions tests

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

## Test python environment is setup correctly
tests:
	$(PYTHON_INTERPRETER) -m pytest tests
