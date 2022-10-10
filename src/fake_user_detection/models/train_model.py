import click
import json
import logging
import os
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from fake_user_detection.config import load_config

CONF = load_config()
OUTPUT_ROOT = CONF["path"]["output_data_root"]


@click.group()
def train():
    pass


@train.command()
def train_model():
    print("Training Model")

    X_train = pd.read_csv(os.path.join(OUTPUT_ROOT, "train_features.csv")).set_index("UserId")
    y_train = X_train.pop("Fake")
    X_validation = pd.read_csv(os.path.join(OUTPUT_ROOT, "validation_features.csv")).set_index("UserId")
    y_validation = X_validation.pop("Fake")

    normalizer = tf.keras.layers.Normalization(axis=-1)

    linear_model = tf.keras.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])
    linear_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error',
    )

    callbacks = []
    tensorboard_callback = None
    model_checkpoint_callback = None
    best_model_callback = None
    early_stopping_callback = None

    history = linear_model.fit(
        X_train,
        y_train.values,
        epochs=10,
        validation_data=(
            X_validation,
            y_validation.values
        )
    )

    logging.info("Saving Model")
