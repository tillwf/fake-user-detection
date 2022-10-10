import click
import json
import logging
import os
import pandas as pd

from tensorflow.keras.models import load_model

from fake_user_detection.config import load_config
from fake_user_detection.features.category_interaction import CategoryInteraction
from fake_user_detection.features.event_distribution import EventDistribution
from fake_user_detection.features.event_frequency import EventFrequency

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
OUTPUT_ROOT = CONF["path"]["output_data_root"]
MODELS_ROOT = CONF["path"]["models_root"]

FEATURE_DICT = {
    "category_interaction": CategoryInteraction,
    "event_distribution": EventDistribution,
    "event_frequency": EventFrequency
}


@click.group()
def predict():
    pass


@predict.command()
@click.option(
    '--data-path',
    type=str,
    default=DATA_PATH,
    help='Path of train dataset, default is {}'.format(
        DATA_PATH
    )
)
@click.option(
    '--output-root',
    type=str,
    default=OUTPUT_ROOT,
    help='Path of output folder, default is {}'.format(
        OUTPUT_ROOT
    )
)
@click.option(
    '--models-root',
    type=str,
    default=MODELS_ROOT,
    help='Path of models folder, default is {}'.format(
        MODELS_ROOT
    )
)
def make_predictions(data_path, output_root, models_root, features):
    logging.info("Make Prediction")

    logging.info("Reading test data")
    test_users = pd.read_csv(os.path.join(OUTPUT_ROOT, "test_users.csv"))

    logging.info("Create features")
    features_class = [FEATURE_DICT[f] for f in features if FEATURE_DICT.get(f)]
    for feature in features_class:
        f = feature.extract_feature(df)
        users_features.append(f)
    X_test = users_features.merge(test_users, on="UserId", how="right")

    logging.info("Loading model")
    model = load_model(os.path.join(MODEL_PATH, "final_model.h5"))

    logging.info("Making predictions")
    predictions = model.predict(dataset_test)

    logging.info("Saving predictions")