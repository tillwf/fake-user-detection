import click
import functools as ft
import json
import logging
import os
import pandas as pd

from fake_user_detection.config import load_config
from fake_user_detection.features.category_interaction import CategoryInteraction
from fake_user_detection.features.event_distribution import EventDistribution
from fake_user_detection.features.event_frequency import EventFrequency

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
OUTPUT_ROOT = CONF["path"]["output_data_root"]

FEATURES = [
    CategoryInteraction,
    EventDistribution,
    EventFrequency
]

@click.group()
def build():
    pass


@build.command()
@click.option(
    '--data-path',
    type=str,
    default=DATA_PATH,
    help='Path of train dataset, default is {data_set}'.format(
        data_set=DATA_PATH
    )
)
@click.option(
    '--output-root',
    type=str,
    default=OUTPUT_ROOT,
    help='Path of output folder, default is {data_set}'.format(
        data_set=OUTPUT_ROOT
    )
)
def build_features(data_path, output_root):
    logging.info("Loading Data")

    df = pd.read_csv(data_path)
    train_users = pd.read_csv(os.path.join(OUTPUT_ROOT, "train_users.csv"))
    validation_users = pd.read_csv(os.path.join(OUTPUT_ROOT, "validation_users.csv"))

    logging.info("Computing the features")
    users_features = []
    for feature in FEATURES:
        f = feature.extract_feature(df)
        users_features.append(f)

    logging.info("Merging the features")
    users_features = ft.reduce(
        lambda left, right: pd.merge(left, right, on='UserId', how="outer").fillna(0),
        users_features
    )

    X_train = users_features.merge(train_users, on="UserId", how="right")
    X_validation = users_features.merge(validation_users, on="UserId", how="right")
    
    logging.info("Saving the features")
    X_train.to_csv(os.path.join(OUTPUT_ROOT, "train_features.csv"), index=False)
    X_validation.to_csv(os.path.join(OUTPUT_ROOT, "validation_features.csv"), index=False)