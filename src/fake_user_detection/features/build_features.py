import click
import json
import logging
import os
import pandas as pd

from fake_user_detection.config import load_config
from fake_user_detection.features.category_interaction import CategoryInteraction
from fake_user_detection.features.event_distribution import EventDistribution
from fake_user_detection.features.event_frequency import EventFrequency

CONF = load_config()
OUTPUT_ROOT = CONF["path"]["output_data_root"]

@click.group()
def build():
    pass


@build.command()
@click.option(
    '--output-root',
    type=str,
    default=OUPUT_ROOT,
    help='Path of output folder, default is {data_set}'.format(
        data_set=OUPUT_ROOT
    )
)
def build_features():
    logging.info("Loading Data")
    X_train = pd.read_csv("{}/train.csv".format(OUTPUT_ROOT))