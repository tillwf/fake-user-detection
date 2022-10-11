import click
import functools as ft
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
OUTPUT_ROOT = CONF["path"]["output_data_root"]
MODELS_ROOT = CONF["path"]["models_root"]
TESTSET_PATH = CONF["path"]["testset_path"]

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
    '--testset-path',
    type=str,
    default=TESTSET_PATH,
    help='Path of test dataset, default is {}'.format(
        TESTSET_PATH
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
@click.option(
    '--output-root',
    type=str,
    default=OUTPUT_ROOT,
    help='Path of output folder, default is {}'.format(
        OUTPUT_ROOT
    )
)
@click.option(
    '--features',
    type=list,
    default=FEATURE_DICT.keys,
    help='Features used for the training, default is {}'.format(
        FEATURE_DICT.keys
    )
)
def make_predictions(testset_path, models_root, output_root, features, evaluate=True):
    logging.info("Make Prediction")

    logging.info("Reading test data")
    test_data = pd.read_csv(testset_path)

    if "Fake" in test_data:
        y_test = test_data[["UserId", "Fake"]].drop_duplicates().set_index("UserId")
        test_data.pop("Fake")

    logging.info("Create features")
    features_class = [FEATURE_DICT[f] for f in features if FEATURE_DICT.get(f)]
    users_features = []
    for feature in features_class:
        f = feature.extract_feature(test_data)
        users_features.append(f)

    users_features = ft.reduce(
        lambda left, right: pd.merge(left, right, on='UserId', how="outer").fillna(0),
        users_features
    )

    logging.info("Loading model")
    model = load_model(os.path.join(models_root, "final_model.h5"))

    logging.info("Making predictions")
    predictions = pd.DataFrame(
        model.predict(users_features),
        index=users_features.index,
        columns=["predictions"]
    )

    logging.info("Saving predictions")
    predictions.to_csv(os.path.join(OUTPUT_ROOT, "predictions.csv"))
    if evaluate:
        logging.info("Evaluating predictions")
        logging.info(model.evaluate(users_features, y_test))