import click
from fake_user_detection.data.make_dataset import dataset
from fake_user_detection.features.build_features import build
from fake_user_detection.models.train_model import train
from fake_user_detection.models.predict_model import predict

cli = click.CommandCollection(sources=[
    dataset,
    build,
    predict,
    train,
])

if __name__ == '__main__':
    cli()
