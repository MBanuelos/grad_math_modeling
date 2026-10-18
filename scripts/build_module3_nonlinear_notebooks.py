"""Generate Module 3 nonlinear-model notebooks."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Lectures" / "Module3_DeepLearning"


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


COMMON = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

RANDOM_STATE = 232
wine = load_wine(as_frame=True)
X, y = wine.data, wine.target
class_names = wine.target_names
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
print("features:", list(X.columns))
print("training observations:", len(X_train), "test observations:", len(X_test))"""


def badge(filename):
    return md(
        "[![](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"(https://colab.research.google.com/github/mbanuelos/grad_math_modeling/blob/main/Lectures/Module3_DeepLearning/{filename}#copy=true)"
    )


LESSONS = {
    "11_DecisionTrees.ipynb": [
        badge("11_DecisionTrees.ipynb"),
        md("""# Decision Trees: Nonlinear, Interpretable Models

## Learning objectives

By the end of this lesson, you should be able to:

- explain how a decision tree divides feature space with sequential questions;
- describe Gini impurity and how it guides a split;
- fit, visualize, and evaluate a classification tree;
- explain how tree depth affects overfitting.

**Student Learning Outcome:** Fit a nonlinear supervised-learning model and justify evaluation choices using held-out data."""),
        md("""## Why a nonlinear model?

Linear models use one weighted combination of features. A decision tree asks sequential threshold questions such as ``is alcohol at most 13.2?'' Every root-to-leaf path defines a region of feature space, so trees represent interactions and nonlinear decision boundaries without manually adding polynomial features.

We will classify three wine cultivars from chemical measurements. The class label is known for every observation, so this is supervised classification."""),
        code(COMMON + "\n\nX.head()"),
        md("""## Part 1 — Nodes, leaves, and impurity

A **node** contains a subset of training observations. A **split** sends observations left or right using one feature threshold. A terminal node is a **leaf**; its predicted class is the majority class among training observations in that leaf.

For class proportions $p_1,\ldots,p_C$, Gini impurity is

$$G=1-\sum_{c=1}^C p_c^2.$$

It is zero when every observation in the node has the same class. The tree selects splits that lower weighted child-node impurity."""),
        md("""### Checkpoint 1

A node contains 8 observations: 6 from class 0 and 2 from class 1. Is its Gini impurity zero? Would two pure child nodes lower impurity? Explain.

> **YOUR ANSWER:**"""),
        md("""## Part 2 — Fit and inspect a small tree

We limit `max_depth` so that the tree remains readable. This is a modeling choice, not a property of the data. The test set remains untouched while the tree is fitted."""),
        code("""from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
tree.fit(X_train, y_train)
test_predictions = tree.predict(X_test)
print("test accuracy:", round(accuracy_score(y_test, test_predictions), 3))
print("number of leaves:", tree.get_n_leaves())

plt.figure(figsize=(18, 8))
plot_tree(tree, feature_names=X.columns, class_names=class_names,
          filled=True, rounded=True, proportion=True)
plt.show()"""),
        md("""### Practice 2

Choose one root-to-leaf path in the plot. Translate its threshold questions into an ordinary-language rule. What class does its leaf predict, and what evidence in the plot supports your answer?

> **YOUR ANSWER:**"""),
        code("""ConfusionMatrixDisplay.from_predictions(y_test, test_predictions, display_labels=class_names)
plt.title("Decision-tree test predictions")
plt.show()"""),
        md("""## Part 3 — Flexibility versus generalization

A deeper tree can carve training data into more regions. This can reduce training error while increasing test error. The table below is exploratory; use cross-validation rather than repeatedly selecting the best final test result in a real workflow."""),
        code("""rows = []
for depth in range(1, 9):
    candidate = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE).fit(X_train, y_train)
    rows.append({"max depth": depth, "training accuracy": candidate.score(X_train, y_train),
                 "test accuracy": candidate.score(X_test, y_test), "leaves": candidate.get_n_leaves()})
pd.DataFrame(rows).round(3)"""),
        md("""### Practice 3

Identify a depth where training accuracy is noticeably higher than test accuracy. What pattern would count as evidence of overfitting, and why would cross-validation be preferable for selecting a final depth?

> **YOUR ANSWER:**"""),
        md("""## Lesson summary

Decision trees create nonlinear decision regions through interpretable threshold rules. Their flexibility is useful, but unrestricted trees can memorize training noise. Next, random forests reduce the instability of one tree by aggregating many randomized trees."""),
    ],
    "12_RandomForests.ipynb": [
        badge("12_RandomForests.ipynb"),
        md("""# Random Forests: Ensembles of Decision Trees

## Learning objectives

By the end of this lesson, you should be able to:

- explain bootstrap sampling and random feature selection in a random forest;
- compare a single decision tree with an ensemble on held-out data;
- interpret out-of-bag estimates and feature importance cautiously;
- describe why averaging less-correlated trees can reduce variance.

**Student Learning Outcome:** Compare nonlinear models using a held-out evaluation and interpret diagnostics in context."""),
        md("""## From one tree to many

A single tree can change substantially when its training data change slightly. A **random forest** fits many trees to bootstrap samples: samples drawn with replacement from the training data. At each split, each tree considers only a random subset of features. The forest predicts the class receiving the most votes.

These sources of randomness make individual trees less alike. Voting across less-correlated trees can reduce variance while retaining nonlinear boundaries."""),
        code(COMMON),
        md("""## Part 1 — Baseline tree and forest

Use the same train/test split for both models so their test scores are comparable. A test score estimates performance on this held-out set; it is not a guarantee of future performance."""),
        code("""from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

single_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
forest = RandomForestClassifier(n_estimators=300, max_features="sqrt", oob_score=True,
                                random_state=RANDOM_STATE, n_jobs=-1)
for name, model in [("single tree", single_tree), ("random forest", forest)]:
    model.fit(X_train, y_train)
    print(f"{name:>14}: train={model.score(X_train, y_train):.3f}, test={model.score(X_test, y_test):.3f}")
print("forest out-of-bag accuracy:", round(forest.oob_score_, 3))"""),
        md("""### What is out-of-bag (OOB) accuracy?

Each bootstrap sample leaves out some training observations. For an observation, the trees that did not train on it can vote on its class. Combining those votes gives an **out-of-bag estimate**, an internal validation-style diagnostic. It is useful, but it does not replace a final test set."""),
        md("""### Checkpoint 1

Why does a forest use both bootstrap samples and random feature subsets? Explain why fitting 300 identical trees to the same data would not provide the same benefit.

> **YOUR ANSWER:**"""),
        md("""## Part 2 — Feature importance is not causation

`feature_importances_` reports how much splits using a feature reduced impurity across the trees. A large value can indicate usefulness to this fitted forest, but correlated features can share or mask importance, and importance does not establish a causal effect."""),
        code("""importance = pd.Series(forest.feature_importances_, index=X.columns).sort_values()
importance.plot.barh(figsize=(8, 5), color="tab:green")
plt.xlabel("mean decrease in impurity")
plt.title("Random-forest impurity-based feature importance")
plt.show()"""),
        md("""### Practice 2

Name the two most important displayed features. Give one reason this ranking might change with a new sample of wines or a strongly correlated measurement.

> **YOUR ANSWER:**"""),
        md("""## Part 3 — Number of trees and stability

Adding trees generally stabilizes a forest's vote; it does not authorize unlimited reuse of test data. Explore the training workflow below, then reserve a final claim for a held-out evaluation."""),
        code("""rows = []
for count in [1, 5, 20, 100, 300]:
    candidate = RandomForestClassifier(n_estimators=count, max_features="sqrt",
                                       random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
    rows.append({"trees": count, "training accuracy": candidate.score(X_train, y_train),
                 "test accuracy": candidate.score(X_test, y_test)})
pd.DataFrame(rows).round(3)"""),
        md("""## Lesson summary

Random forests trade some single-tree interpretability for more stable predictions. Scores, feature-importance plots, and OOB estimates still require careful interpretation. Next, boosting will build an ensemble sequentially rather than averaging independently grown trees."""),
    ],
    "13_GradientBoosting_XGBoost.ipynb": [
        badge("13_GradientBoosting_XGBoost.ipynb"),
        md("""# Gradient Boosting and XGBoost

## Learning objectives

By the end of this lesson, you should be able to:

- contrast bagging/random forests with sequential boosting;
- explain learning rate, number of stages, and tree complexity;
- fit and evaluate a gradient-boosting classifier;
- describe XGBoost as a regularized, efficient gradient-boosting implementation.

**Student Learning Outcome:** Explain the tradeoffs of a nonlinear ensemble and evaluate it on data not used for fitting."""),
        md("""## Boosting: learn from previous errors

Random forests grow trees independently and vote. **Gradient boosting** grows weak trees sequentially: each new tree is chosen to reduce the loss left by the current ensemble. For classification, later trees focus on observations earlier trees have not modeled well.

The learning rate controls how much each new tree changes the ensemble. Smaller learning rates often require more stages. Deep trees or too many stages can overfit, so validation remains essential."""),
        code(COMMON),
        md("""## Part 1 — Fit a gradient-boosting model

`HistGradientBoostingClassifier` is a scikit-learn gradient-boosting implementation. It provides the core boosting workflow without requiring a separate package."""),
        code("""from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

boosted = HistGradientBoostingClassifier(learning_rate=0.08, max_iter=150,
                                          max_leaf_nodes=12, l2_regularization=0.5,
                                          random_state=RANDOM_STATE)
boosted.fit(X_train, y_train)
predictions = boosted.predict(X_test)
print("training accuracy:", round(boosted.score(X_train, y_train), 3))
print("test accuracy:", round(accuracy_score(y_test, predictions), 3))
ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=class_names)
plt.title("Gradient-boosting test predictions")
plt.show()"""),
        md("""### Checkpoint 1

Contrast the role of trees in a random forest with their role in gradient boosting. Why might a very large number of boosting stages be risky?

> **YOUR ANSWER:**"""),
        md("""## Part 2 — Hyperparameters express a tradeoff

- `learning_rate` shrinks each tree's contribution.
- `max_iter` is the number of boosting stages.
- `max_leaf_nodes` limits the complexity of each stage.
- `l2_regularization` penalizes overly flexible updates.

Choose these settings with cross-validation or a validation set, not by repeatedly selecting the best final test result."""),
        code("""rows = []
for rate, stages in [(0.03, 300), (0.08, 150), (0.2, 60)]:
    candidate = HistGradientBoostingClassifier(learning_rate=rate, max_iter=stages,
                                                max_leaf_nodes=12, l2_regularization=0.5,
                                                random_state=RANDOM_STATE).fit(X_train, y_train)
    rows.append({"learning rate": rate, "stages": stages,
                 "training accuracy": candidate.score(X_train, y_train),
                 "test accuracy": candidate.score(X_test, y_test)})
pd.DataFrame(rows).round(3)"""),
        md("""### Practice 2

Choose two rows with similar training accuracy but different test accuracy. What does this suggest about tuning an ensemble rather than assuming that more stages are always better?

> **YOUR ANSWER:**"""),
        md("""## Part 3 — Where XGBoost fits

**XGBoost** is a widely used implementation of gradient-boosted trees. It adds efficient computation, regularization options, and practical training tools. The statistical idea is still boosting: add regularized trees that improve the current model's loss.

The optional cell below runs only when the `xgboost` package is installed. In Colab, use `!pip -q install xgboost` in a separate setup cell if needed. The course concepts do not depend on a package-specific score."""),
        code("""try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False

if xgboost_available:
    xgb = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08,
                        subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss",
                        random_state=RANDOM_STATE)
    xgb.fit(X_train, y_train)
    print("XGBoost test accuracy:", round(xgb.score(X_test, y_test), 3))
else:
    print("Optional package unavailable. The required boosting model is above.")"""),
        md("""## Lesson summary

Boosting builds an additive model sequentially; random forests average independently grown trees. Both are nonlinear ensembles, and both must be selected and evaluated without test-set leakage. The next lesson compares nonlinear models with a linear baseline before we move to neural networks."""),
    ],
    "14_NonlinearModelComparison.ipynb": [
        badge("14_NonlinearModelComparison.ipynb"),
        md("""# Choosing Nonlinear Models: Evaluation and Comparison

## Learning objectives

By the end of this lesson, you should be able to:

- compare a linear classifier with tree-based nonlinear models;
- use cross-validation to estimate performance during model selection;
- distinguish model selection from final test evaluation;
- communicate a model choice using accuracy, variability, and interpretability.

**Student Learning Outcome:** Select and justify a model using evidence from a reproducible evaluation workflow."""),
        md("""## A model score is not a model choice

Decision trees, random forests, and gradient boosting can represent nonlinear boundaries. Their apparent performance depends on the data split, hyperparameters, and metric. We will compare them with a regularized logistic-regression baseline using cross-validation on training data. Only after choosing a candidate should we consult the held-out test set.

Accuracy is appropriate here because the three wine classes are reasonably balanced. Other settings may require metrics that account for class imbalance or unequal error costs."""),
        code(COMMON),
        md("""## Part 1 — Define comparable candidates

Logistic regression benefits from scaled numerical features. Tree-based models do not require scaling for their split rules. A `Pipeline` keeps scaling inside each cross-validation fit, preventing information from a validation fold from affecting its training transform."""),
        code("""from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

models = {
    "logistic regression": Pipeline([("scale", StandardScaler()),
                                     ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))]),
    "decision tree": DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
    "random forest": RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                              random_state=RANDOM_STATE, n_jobs=-1),
    "gradient boosting": HistGradientBoostingClassifier(learning_rate=0.08, max_iter=150,
                                                          max_leaf_nodes=12, l2_regularization=0.5,
                                                          random_state=RANDOM_STATE),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)"""),
        md("""## Part 2 — Cross-validate on training data

**Stratified** folds preserve class proportions approximately in every fold. Each candidate is fit five times, each time holding out a different training fold. We summarize mean and standard deviation of validation accuracy; the test set is still untouched."""),
        code("""rows = []
for name, model in models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring="accuracy", return_train_score=True)
    rows.append({"model": name, "mean validation accuracy": scores["test_score"].mean(),
                 "validation SD": scores["test_score"].std(),
                 "mean training accuracy": scores["train_score"].mean()})
cv_results = pd.DataFrame(rows).sort_values("mean validation accuracy", ascending=False)
cv_results.round(3)"""),
        md("""### Practice 2

Which candidate would you investigate further? Justify your choice with mean validation accuracy and one non-score consideration, such as interpretability, training cost, or stability across folds.

> **YOUR ANSWER:**"""),
        md("""## Part 3 — One final test evaluation

For demonstration, select the model with the largest mean validation accuracy, refit it on all training data, and evaluate it once on the held-out test set. In a real report, model choices must be fixed before this final evaluation."""),
        code("""chosen_name = cv_results.iloc[0]["model"]
chosen_model = models[chosen_name].fit(X_train, y_train)
print("chosen model:", chosen_name)
print("final test accuracy:", round(chosen_model.score(X_test, y_test), 3))"""),
        md("""### Checkpoint 3

Why would reporting the best test accuracy among all four models be misleading as an unbiased final result? Describe the role cross-validation played in avoiding this problem.

> **YOUR ANSWER:**"""),
        md("""## Bridge to deep learning

Tree ensembles learn nonlinear structure by partitioning feature space. Neural networks learn nonlinear structure by composing weighted linear transformations with activation functions and optimizing parameters with gradients. The next lessons introduce PyTorch, the tensor and automatic-differentiation library used to build those networks."""),
        md("""## Lesson summary

There is no universally best nonlinear model. A credible comparison keeps a final test set separate, uses cross-validation for model selection, and weighs performance against interpretability, computation, and consequences of errors."""),
    ],
}


def main():
    MODULE.mkdir(parents=True, exist_ok=True)
    for filename, cells in LESSONS.items():
        path = MODULE / filename
        path.write_text(json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
