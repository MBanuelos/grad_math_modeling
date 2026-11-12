"""Build the Module 3 nonlinear-model investigation notebook."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Lectures" / "Module3_DeepLearning" / "14_NonlinearModelComparison.ipynb"


def src(text):
    return [line + "\n" for line in text.splitlines()] if text else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": src(text)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src(text)}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "course": {"audience": "student", "module": 3},
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


cells = [
    md("""[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbanuelos/grad_math_modeling/blob/main/Lectures/Module3_DeepLearning/14_NonlinearModelComparison.ipynb#copy=true)"""),
    md("""# Nonlinear Model Investigation: Choosing a Model from Evidence

## Learning objectives

By the end of this investigation, you should be able to:

- distinguish a classification problem from a regression problem;
- compare a simple baseline, a linear model, and nonlinear tree-based models;
- use repeated validation results to make a model choice;
- keep a final test set separate from model-building decisions; and
- write a short evidence-based recommendation for a model.

This is an investigation. You will choose one dataset, make a prediction plan, compare models, and justify a choice. There is no single required winning model."""),
    md("""## Vocabulary for this investigation

- A **feature** is an input measurement used to make a prediction. A **target** is the value we want to predict.
- **Classification** predicts a label or category, such as an activity or a digit. **Regression** predicts a number, such as a house value.
- **Training data** are used to fit a model. **Validation data** are used to compare possible model choices. **Test data** are saved until the end for one final evaluation.
- A **baseline** is a deliberately simple prediction method. It tells us whether a more complicated model is actually adding value.
- **Cross-validation** repeatedly divides the training data into a fitting portion and a temporary validation portion. It gives several validation scores instead of relying on one split.
- A **score** is a numerical summary of prediction quality. A score alone is not a model choice: we also consider variability, complexity, explanation, and the kinds of errors made.
- **Data leakage** occurs when information from validation or test data influences a model-building choice.
- A **pipeline** is a single object that performs data preparation and model fitting in the correct order. It prevents preparation steps from learning from validation or test data.
- A **category-valued feature** contains labels such as occupation or work class. Before most models can use it, the labels must be converted to numerical columns. This conversion is called **encoding**."""),
    md("""## Choose a dataset

Set `DATASET` in the next cell to one of the following values.

| Value | Prediction task | Dataset |
|---|---|---|
| `"digits"` | classification | Identify handwritten digits from 64 pixel measurements. This built-in option is the quickest to run. |
| `"adult"` | classification | Predict whether Census Income is above a threshold from demographic and employment information. |
| `"har"` | classification | Identify a person's activity from smartphone sensor measurements. |
| `"housing"` | regression | Predict California neighborhood house values from census and geographic measurements. |

The Adult, Human Activity Recognition (HAR), and California Housing options download public teaching datasets the first time they are used. The code limits the development sample for the larger options so that the investigation remains practical in Colab."""),
    code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from sklearn.datasets import load_digits, fetch_california_housing, fetch_openml
from sklearn.model_selection import train_test_split

RANDOM_STATE = 232
DATASET = "digits"  # Choose: "digits", "adult", "har", or "housing"
DATA_DIRECTORY = Path("module3_investigation_data")
DATA_DIRECTORY.mkdir(exist_ok=True)
"""),
    md("""## Load the selected data

Read this cell before running it. It produces `X` (features), `y` (target), and `task` (`"classification"` or `"regression"`). For HAR, the dataset's published training/test split is retained: we will use its training portion for cross-validation and its published test portion only at the end."""),
    code("""def load_har(data_directory):
    # Download and read the UCI Human Activity Recognition teaching data.
    archive = data_directory / "uci_har.zip"
    url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
    if not archive.exists():
        print("Downloading the Human Activity Recognition data...")
        urlretrieve(url, archive)

    with ZipFile(archive) as zf:
        features = pd.read_csv(zf.open("UCI HAR Dataset/features.txt"), sep=r"\\s+", header=None,
                               names=["index", "feature"])["feature"].tolist()
        activities = pd.read_csv(zf.open("UCI HAR Dataset/activity_labels.txt"), sep=r"\\s+", header=None,
                                 names=["code", "activity"]).set_index("code")["activity"]
        X_train = pd.read_csv(zf.open("UCI HAR Dataset/train/X_train.txt"), sep=r"\\s+", header=None,
                              names=features)
        y_train = pd.read_csv(zf.open("UCI HAR Dataset/train/y_train.txt"), header=None)[0].map(activities)
        X_test = pd.read_csv(zf.open("UCI HAR Dataset/test/X_test.txt"), sep=r"\\s+", header=None,
                             names=features)
        y_test = pd.read_csv(zf.open("UCI HAR Dataset/test/y_test.txt"), header=None)[0].map(activities)
    return X_train, X_test, y_train, y_test


published_test = None
if DATASET == "digits":
    data = load_digits(as_frame=True)
    X, y = data.data, data.target.astype(str)
    task = "classification"
    dataset_name = "Handwritten Digits"
elif DATASET == "adult":
    data = fetch_openml(name="adult", version=2, as_frame=True)
    X = data.data.replace("?", np.nan)
    y = data.target.astype(str).str.contains(">50K")
    task = "classification"
    dataset_name = "Adult Census Income"
elif DATASET == "har":
    X, X_published_test, y, y_published_test = load_har(DATA_DIRECTORY)
    published_test = (X_published_test, y_published_test)
    task = "classification"
    dataset_name = "Human Activity Recognition"
elif DATASET == "housing":
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    task = "regression"
    dataset_name = "California Housing"
else:
    raise ValueError("DATASET must be 'digits', 'adult', 'har', or 'housing'.")

print(f"Dataset: {dataset_name}")
print(f"Task: {task}")
print(f"Observations: {len(X):,}; features: {X.shape[1]}")
X.head()"""),
    md("""## Part 1 — Audit the prediction problem

Before choosing a model, inspect the target and the features. For classification, check how many examples belong to each label. For regression, summarize the numerical target. This audit helps you decide which score is meaningful and what errors deserve attention."""),
    code("""if task == "classification":
    display(pd.DataFrame({"count": y.value_counts(), "proportion": y.value_counts(normalize=True).round(3)}))
else:
    display(y.describe().to_frame("median house value"))

feature_types = pd.DataFrame({"data type": X.dtypes.astype(str)})
display(feature_types["data type"].value_counts().to_frame("number of features"))
print("Missing values:", int(X.isna().sum().sum()))"""),
    md("""### Investigation prompt 1

Identify the target and describe the prediction task in one sentence. Is this classification or regression? Based on the target summary, what kinds of errors might matter to someone using this model?

> **YOUR RESPONSE:**"""),
    md("""## Part 2 — Create a protected test set

The test set is not a prize for the highest score. It is evidence we save until after selecting a model. For Digits, Adult, and Housing, create a new test set containing 20% of the observations. For classification, preserve the label proportions using `stratify=y`. For HAR, use its published training and test portions rather than making a new split.

### Your tasks

1. Create `X_train`, `X_test`, `y_train`, and `y_test`.
2. Use `RANDOM_STATE` so classmates can reproduce your split.
3. Print the number of observations in your training and protected test sets.
4. For Adult or Housing, if runtime becomes a concern, take a reproducible sample of at most 12,000 observations from `X_train, y_train` only. Do not sample from the protected test set."""),
    code("""# TODO 1: Create a protected test set.
# Hint: use train_test_split for Digits, Adult, and Housing.
# Hint: when published_test is not None, use it as X_test, y_test.

# TODO 2: If needed, create a smaller development-training sample here.

# TODO 3: Print the size of each split.
"""),
    md("""## Scores used below

For the classification choices, the notebook uses **accuracy**, the proportion of labels predicted correctly. Adult income has noticeably fewer higher-income observations, so it also reports **balanced accuracy**: the average accuracy computed separately for each class. A model cannot obtain a high balanced accuracy merely by predicting the more common class.

For California Housing, the notebook uses **mean absolute error (MAE)**: the average size of a prediction error, ignoring whether the prediction was too high or too low. Smaller MAE is better. California Housing values are measured in units of \\$100,000, so an MAE of `0.40` corresponds to an average error of about \\$40,000."""),
    md("""### Investigation prompt 2

Before running any models, name the score you think should be most important for your selected dataset and explain why. For Adult, compare the usefulness of ordinary accuracy and balanced accuracy.

> **YOUR RESPONSE:**"""),
    md("""## How to use a Pipeline

A `Pipeline` joins several steps into one model-like object. When you call `.fit()` on a pipeline, it learns the preparation steps using training data and then fits the prediction model. When you call `.predict()`, it applies those same learned preparation steps before making predictions.

This matters during cross-validation. `cross_validate` fits a fresh copy of the entire pipeline inside each training fold. That means the validation fold is not used to calculate a missing-value replacement or a standardization value. This protects against data leakage.

The example below is for an all-numerical classification problem. It fills missing values, standardizes each feature, and fits logistic regression. `named_steps` lets you inspect the individual pieces. This is an example of the pattern; it is not the completed solution for Part 3."""),
    code("""from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

numerical_example = Pipeline([
    ("fill missing values", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
])

numerical_example.named_steps"""),
    md("""### Reading the example

- `SimpleImputer` replaces a missing numerical value with the median learned from the training data.
- `StandardScaler` changes each numerical feature to a common scale using the training-data mean and spread.
- `LogisticRegression` is the final prediction model.

For a dataset with category-valued features, Part 3 will also use `ColumnTransformer` and `OneHotEncoder`. `ColumnTransformer` lets one pipeline prepare numerical columns one way and category-valued columns another way. The important structure remains the same: preparation steps and the model stay together in one pipeline."""),
    md("""## Part 3 — Build comparable model candidates

Every candidate below should include its data-preparation steps inside a pipeline. For category-valued features, the pipeline should fill missing entries and turn each category into numerical indicator columns. For numerical features, the linear model should standardize measurements so that one unit in a large-scale feature does not dominate merely because of its units.

### Your tasks

1. Identify the numerical and category-valued feature columns in `X_train`.
2. Build a preparation workflow for a linear model: fill missing numerical values, standardize numerical values, and encode category-valued features when present.
3. Build a preparation workflow for tree-based models. Tree-based models do not require numerical standardization, but they still need missing values handled and category-valued features encoded.
4. Make a dictionary named `models` with a baseline, a linear model, a decision tree, a random forest, and a gradient-boosting model. Use the classifier versions for a classification dataset and the regressor versions for Housing.
5. Create a five-fold cross-validation object. Use `StratifiedKFold` for classification and `KFold` for regression.

Suggested starting points: `DummyClassifier` or `DummyRegressor` for the baseline; `LogisticRegression` or `Ridge` for the linear model; and the tree models imported below. The goal is not to declare one family universally best. It is to learn which evidence supports a choice on this dataset."""),
    code("""from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# TODO: complete the Part 3 tasks here.
# Keep all preparation inside your Pipeline objects.
# Your completed cell should define: numeric_features, categorical_features,
# preparation workflow(s), models, cv, and scoring.
"""),
    md("""### Investigation prompt 3

Before fitting the candidates, make a prediction. Which model do you expect to outperform the baseline, and why? Which model might be easiest to explain to a nontechnical audience? Your answer is a hypothesis, not a commitment to select that model later.

> **YOUR RESPONSE:**"""),
    md("""## Part 4 — Compare candidates with cross-validation

Write the comparison yourself. Fit every entry in `models` five times using `cross_validate`, `X_train`, `y_train`, and your cross-validation object. Create a results table that includes the mean validation score and the variation among the five validation scores for every model. The protected test set must remain untouched.

For regression, scikit-learn returns negative MAE because its scoring convention treats larger numbers as better. Reverse the sign before reporting MAE so that smaller values are better.

### Your tasks

1. Create an empty list called `rows`.
2. Loop through your models, cross-validate each one, and record its name, mean score, and score variation.
3. Convert `rows` to a clearly labeled DataFrame called `cv_results`.
4. For classification, include accuracy and balanced accuracy. For regression, report positive MAE."""),
    code("""# TODO: Cross-validate each model and create cv_results.
# Do not use X_test or y_test in this cell.

# rows = []
# for name, model in models.items():
#     ...
"""),
    md("""### Investigation prompt 4 — A score is not a model choice

Use the cross-validation table to select a provisional model. Your justification must include:

1. the relevant validation score and its variation;
2. a comparison with the baseline; and
3. one additional consideration, such as interpretability, computation time, or the error pattern you expect.

Do **not** consult the protected test score yet.

> **YOUR RESPONSE:**"""),
    md("""## Part 5 — Investigate one reasonable variation

Choose one model from the table and change one meaningful setting. Examples: use a shallower or deeper decision tree, change the number of trees in the forest, or adjust the number of boosting stages. State why that setting is worth investigating before you run it. Then build and compare the variation using the same cross-validation object.

This is model selection: you may inspect cross-validation results because they come from the training data. Do not repeatedly use the protected test set while trying variations."""),
    code("""# TODO: Build one variation of a candidate model.
# Then use cross_validate with X_train, y_train, cv, and scoring.
# Add the variation to your comparison table or make a second small table.
"""),
    md("""### Investigation prompt 5

Describe the variation you tested. Did its cross-validation evidence change your provisional choice? Explain your conclusion without using the protected test data.

> **YOUR RESPONSE:**"""),
    md("""## Part 6 — Lock the choice and evaluate once

Only after writing your selection rationale should you select one model below. Fit it on all available training data and evaluate it once on the protected test set. If the final result differs from cross-validation, that is useful information about uncertainty; it is not a reason to restart selection using the test set.

### Your tasks

1. Set `selected_name` to the key for your chosen entry in `models`, or replace that entry with your chosen variation.
2. Fit the selected model on `X_train, y_train` and make `test_predictions` for `X_test`.
3. Report both classification scores, or positive MAE for Housing.
4. Do not change the model after seeing this result."""),
    code("""# TODO: Complete the final protected-test evaluation here.
# selected_name = ...
# selected_model = ...
# test_predictions = ...
"""),
    md("""## Part 7 — Look beyond one number

The final score summarizes performance, but it does not reveal every kind of mistake. For classification, a confusion matrix shows which labels are confused. For regression, a residual plot shows errors: `actual value - predicted value`. A useful model review asks where and how it makes mistakes.

### Your tasks

1. For classification, create a confusion matrix from `y_test` and `test_predictions`.
2. For regression, calculate residuals (`y_test - test_predictions`) and make a scatter plot of predicted value versus residual.
3. Identify one error pattern or limitation that a single summary score hides."""),
    code("""# TODO: Make a protected-test confusion matrix or residual plot.
# Use y_test and test_predictions created in Part 6.
"""),
    md("""### Final investigation report

Write a concise recommendation for a classmate who must use one model on this dataset.

1. Name the dataset, target, and task.
2. State which model you selected and summarize the cross-validation evidence that led to the choice.
3. Report the one protected-test result separately.
4. Describe one error pattern or limitation that a single score does not capture.

> **YOUR RESPONSE:**"""),
    md("""## Lesson summary

Model selection is an argument based on evidence, not a contest for the largest number in a table. A credible workflow defines the task and score, compares models using training-data cross-validation, chooses a model for stated reasons, and then evaluates that locked choice once on protected test data. The next module lessons introduce PyTorch tools for building another family of nonlinear models: neural networks."""),
]


OUTPUT.write_text(json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8")
print(OUTPUT.relative_to(ROOT))
