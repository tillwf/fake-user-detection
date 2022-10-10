import click
import json
import logging
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from fake_user_detection.config import load_config

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
OUPUT_ROOT = CONF["path"]["output_data_root"]


@click.group()
def dataset():
    pass


@dataset.command()
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
    default=OUPUT_ROOT,
    help='Path of output folder, default is {}'.format(
        OUPUT_ROOT
    )
)
def make_dataset(data_path, output_root):
    logging.info("Making Dataset")
    logging.info(data_path)
    logging.info(output_root)

    df = pd.read_csv(data_path)
    unique_users = df[["UserId", "Fake"]].drop_duplicates()

    train_users, validation_test_users = train_test_split(
        unique_users,
        shuffle=True,
        stratify=unique_users["Fake"],
        test_size=0.33
    )
    validation_users, test_users = train_test_split(
        validation_test_users,
        shuffle=True,
        stratify=validation_test_users["Fake"],
        test_size=0.5
    )

    logging.info("Saving Files")
    
    train_users.to_csv(os.path.join(output_root, "train_users.csv"), index=False)
    validation_users.to_csv(os.path.join(output_root, "validation_users.csv"), index=False)
    test_users.to_csv(os.path.join(output_root, "test_users.csv"), index=False)
    
    logging.info("Train Size: {} ({:.2%} fake users)".format(
        len(train_users),
        sum(train_users["Fake"])/len(train_users)
    ))
    logging.info("Validation Size: {} ({:.2%} fake users)".format(
        len(validation_users),
        sum(validation_users["Fake"])/len(validation_users)
    ))
    logging.info("Test Size: {} ({:.2%} fake users)".format(
        len(test_users),
        sum(test_users["Fake"])/len(test_users)
    ))

