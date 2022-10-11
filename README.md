# Fake User Detection

This program aims at detecting fake users based on their activity.

## Installation

Python version: `3.10.7`

```bash
pip install -U pip
pip install -e .
```

Create the folder `data/raw` and copy paste the csv file `fake_users.csv`.

## Commands

### Help

```bash
python -m fake_user_detection
```

Will display all possible commands with their description. You can display each command documentation with:

```bash
python -m fake_user_detection <command> --help
```

### Configuration

Every default parameters are stored in the `config.yml` file.

### Dataset Creation

Using the raw data we want to make a train/validation/test split while keeping the "Fake" column distribution.

For the default values you can do:

```bash
make dataset
```

else

```bash
python -m fake_user_detection make-dataset \
    --data-path data/raw/fake_users.csv \
    --output-root data/processed/ 
```

### Features Construction

This command will compute all the features defined by the global variable `FEATURE_DICT`.
We will always compute all the features and select them, afterwards, during the training step.

```bash
make features
```

else

```bash
python -m fake_user_detection build-features \
    --data-path data/raw/fake_users.csv \
    --output-root data/processed/
```

You can add features by implementing a class which inherit from `Feature` and implement the method `extract_feature`. Then update the `FEATURE_DICT` global variable to add your class as a new feature constructor.

#### Category Interaction

Computes the number of category each user interacted with and its proportion for each user. Returns two columns called `n_unique_category` and `n_unique_category_proportion`.

#### Event Distribution

Computes the percentage of interaction of each category for each user: this will give as many columns as categories.

#### Event Frequency

Computes how many time we observed the event `click_ad` in a row. The column is called `n_consecutive_click_ad`.

### Train the model

The Logistic Regression is implemented using Tensorflow to be able to visualize easily the training process using Tensorboard, to save and use the model quickly and to be able to complexify it without changing too much the code.

```bash
make train
```

or

```bash
python -m fake_user_detection train-model \
    --output-root data/processed/ \
    --models-root models \
    --logs-root logs\
    --features "category_interaction" \
    --features "event_distribution" \
    --features "event_frequency"
```
The model are saved in the `models` folder using its name and its version: `models/Linear/1`. It helps comparing results of different approaches.

### Make predictions

Save the prediction and print the performance if the `Fake` columns exists and the parameter `evaluate` is True (for debugging only).

```bash
make predictions
```

or

```bash
python -m fake_user_detection make-predictions \
    --testset-path data/processed/test_data.csv \
    --models-root models \
    --output-root data/processed/ \
    --features "category_interaction" \
    --features "event_distribution" \
    --features "event_frequency"
```

### Make tests

```bash
make tests
```

or

```bash
pytest tests
```

## Future Work

 - Dockerfile
 - Complete tests
 - More docstring
 - Implement a `Trainer` class to be able to change easily the library which makes the training and the prediction
 - remove duplicate code for feature generation