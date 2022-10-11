import click
import json
import logging
import os
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from fake_user_detection.config import load_config
from fake_user_detection.features.category_interaction import CategoryInteraction
from fake_user_detection.features.event_distribution import EventDistribution
from fake_user_detection.features.event_frequency import EventFrequency

CONF = load_config()
OUTPUT_ROOT = CONF["path"]["output_data_root"]
MODELS_ROOT = CONF["path"]["models_root"]
LOGS_ROOT = CONF["path"]["logs_root"]

FEATURE_DICT = {
    "category_interaction": CategoryInteraction,
    "event_distribution": EventDistribution,
    "event_frequency": EventFrequency
}

@click.group()
def train():
    pass


@train.command()
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
@click.option(
    '--logs-root',
    type=str,
    default=LOGS_ROOT,
    help='Path of logs folder, default is {}'.format(
        LOGS_ROOT
    )
)
@click.option(
    '--features',
    type=str,
    multiple=True,
    default=list(FEATURE_DICT.keys()),
    help='Features used for the training, default is {}'.format(
        list(FEATURE_DICT.keys())
    )
)
def train_model(models_root, output_root, logs_root, features):
    logging.info("Training Model")

    X_train = pd.read_csv(os.path.join(OUTPUT_ROOT, "train_features.csv"),index_col=0, header=[0, 1])
    y_train = pd.read_csv(os.path.join(OUTPUT_ROOT, "train_users.csv")).set_index("UserId")
    X_validation = pd.read_csv(os.path.join(OUTPUT_ROOT, "validation_features.csv"),index_col=0, header=[0, 1])
    y_validation = pd.read_csv(os.path.join(OUTPUT_ROOT, "validation_users.csv")).set_index("UserId")
    
    features_class = [f for f in features if FEATURE_DICT.get(f)]
    X_train = X_train[features_class].droplevel(0, axis=1)
    X_validation = X_validation[features_class].droplevel(0, axis=1)

    normalizer = tf.keras.layers.Normalization(axis=-1)

    linear_model = tf.keras.Sequential([
        normalizer,
        layers.Dense(
            units=1,
            activation='sigmoid',
            input_dim=X_train.shape[1]
        )
    ])
    linear_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="binary_crossentropy",
    )

    callbacks = []

    os.makedirs(models_root, exist_ok=True)
    os.makedirs(logs_root, exist_ok=True)
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=logs_root,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq=100,
        profile_batch=2,
        embeddings_freq=1
    )
    callbacks.append(tensorboard)

    best_model_file = os.path.join(MODELS_ROOT, "best_model_so_far.h5")
    best_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_model_file,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    callbacks.append(best_model_checkpoint)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=5,
        monitor="val_loss"
    )
    callbacks.append(early_stopping)

    history = linear_model.fit(
        X_train,
        y_train.values,
        callbacks=callbacks,
        epochs=100,
        validation_data=(
            X_validation,
            y_validation.values
        )
    )

    logging.info("Saving Model")
    linear_model.load_weights(best_model_file)
    linear_model.save(os.path.join(MODELS_ROOT, "final_model.h5"))
