import click
import json
import logging
import os
import pandas as pd

from fake_user_detection.config import load_config

CONF = load_config()
OUTPUT_ROOT = CONF["path"]["output_data_root"]


@click.group()
def train():
    pass


@train.command()
def train_model():
    print("Training Model")