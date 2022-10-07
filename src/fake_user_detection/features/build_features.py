import click
import json
import logging
import os
import pandas as pd

from fake_user_detection.config import load_config

CONF = load_config()
OUTPUT_ROOT = CONF["path"]["output_data_root"]

@click.group()
def build():
    pass


@build.command()
def build_features():
    print("Building features")