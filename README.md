# Fake User Detection

## Installation

Python version: `3.10.7`

```
pip install -U pip
pip install -e .
```

## Commands

### Help

```
python -m fake_user_detection
```

Will display all possible command with their description. You can display each command documentation with:

```
python -m fake_user_detection <command> --help
```

### Dataset Creation

Using the raw data we want to make a train/validation/test split while keeping the "Fake" column distribution.

For the default values you can do:

```
make dataset
```

else

```
python -m fake_user_detection make-dataset \
	--data-path data/raw/fake_users.csv \
	--output-root data/processed/ 
```

### Features Construction

This command will compute all the features define in the global variable `FEATURE_DICT`.
We will always compute all the features and select them during the training step.

```
make features
```

else

```
python -m fake_user_detection build-features \
	--data-path data/raw/fake_users.csv \
	--output-root data/processed/
```

### Train the model


```
make train
```

or

```
python -m fake_user_detection train-model \
	--output-root data/processed/ \
	--models-root models \
	--logs-root logs\
	--features "category_interaction" \
	--features "event_distribution" \
	--features "event_frequency"
```

### Make predictions

```
make predictions
```

or

```
python -m fake_user_detection make-predictions \
	--testset-path data/processed/test_data.csv \
	--models-root models \
	--output-root data/processed/ \
	--features "category_interaction" \
	--features "event_distribution" \
	--features "event_frequency"
```

### Make tests

```
make tests
```

or

```
pytest tests
```