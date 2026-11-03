"""Append idempotent end-of-lesson exercise blocks to Module 3 chapters."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Lectures" / "Module3_DeepLearning"
TAG = "module3-end-exercises"


def source(text):
    return [line + "\n" for line in text.splitlines()]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {"tags": [TAG]}, "source": source(text)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [TAG]},
        "outputs": [],
        "source": source(text),
    }


def block(introduction, exercises, starter=""):
    cells = [markdown("## End-of-lesson exercises\n\n" + introduction)]
    for heading, prompt in exercises:
        cells.append(markdown(f"### {heading}\n\n{prompt}\n\n> **YOUR RESPONSE:**"))
    if starter:
        cells.append(code(starter))
    return cells


EXERCISES = {
    "11_DecisionTrees.ipynb": block(
        "Apply the decision-tree workflow to the Wine data without changing the held-out test set until your choices are fixed.",
        [
            ("Exercise 1 — Set up a development comparison", "Split `X_train, y_train` again into a development-training set and validation set. Fit trees with at least four values of `max_depth`. Make a table of training and validation accuracy, and identify a depth you would carry forward."),
            ("Exercise 2 — Explain a rule", "Fit the selected tree on the development-training data and plot it. Translate one complete root-to-leaf path into an if--then rule. State the predicted cultivar and use the leaf counts to explain how much training evidence supports it."),
            ("Exercise 3 — Final check and reflection", "Refit your selected depth on all of `X_train, y_train`, then evaluate once on `X_test, y_test`. Compare that result with your validation result. Name one advantage and one limitation of using a shallow tree here."),
        ],
        """# Starter: do not use X_test or y_test while selecting max_depth.
from sklearn.model_selection import train_test_split

X_dev_train, X_val, y_dev_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, stratify=y_train, random_state=RANDOM_STATE
)

rows = []
for depth in [1, 2, 3, 4, 5, 6]:
    # TODO: fit a DecisionTreeClassifier on X_dev_train, y_dev_train.
    # TODO: add training and validation accuracy to rows.
    pass
pd.DataFrame(rows)""",
    ),
    "12_RandomForests.ipynb": block(
        "Use the forest as an empirical study of stability, evaluation, and cautious feature interpretation.",
        [
            ("Exercise 1 — Compare forest settings", "Fit at least three forests that vary `n_estimators` and either `max_features` or `min_samples_leaf`. Use a validation split from the training data. Which setting is your preliminary choice, and what evidence supports it?"),
            ("Exercise 2 — OOB versus validation", "For the forest settings that use `bootstrap=True` and `oob_score=True`, compare OOB accuracy to validation accuracy. Are they identical? Explain what each estimates and why a final test set is still needed."),
            ("Exercise 3 — Interpret importance carefully", "Plot the feature importances for your selected forest. Identify two leading features, then write a two-sentence interpretation that avoids treating importance as a causal claim."),
        ],
        """# Starter: make choices using a validation split from the training data.
from sklearn.model_selection import train_test_split

X_dev_train, X_val, y_dev_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, stratify=y_train, random_state=RANDOM_STATE
)
settings = [
    {"n_estimators": 50, "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 200, "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 200, "max_features": None, "min_samples_leaf": 2},
]
# TODO: fit every setting, record OOB and validation accuracy, and choose one.
""",
    ),
    "13_GradientBoosting_XGBoost.ipynb": block(
        "Practice selecting a boosted model without turning the test set into a tuning tool.",
        [
            ("Exercise 1 — Build a tuning table", "On a validation split from `X_train, y_train`, compare at least four boosting settings. Vary learning rate and number of stages together, and vary one complexity control such as `max_leaf_nodes` or `l2_regularization`. Record training and validation accuracy."),
            ("Exercise 2 — Explain the tradeoff", "Choose two settings with different learning rates. Explain how their number of stages and validation results illustrate the tradeoff between small updates and the number of updates."),
            ("Exercise 3 — Compare model families", "Compare your selected boosted model with the random forest or decision tree from earlier lessons on the same validation split. Make a preliminary recommendation that considers both predictive performance and one practical consideration."),
        ],
        """# Starter: reserve X_test and y_test for a single final evaluation.
from sklearn.model_selection import train_test_split

X_dev_train, X_val, y_dev_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, stratify=y_train, random_state=RANDOM_STATE
)
settings = [(0.03, 300, 8), (0.05, 200, 12), (0.10, 100, 12), (0.20, 60, 20)]
rows = []
for rate, stages, leaves in settings:
    # TODO: fit HistGradientBoostingClassifier and record train/validation accuracy.
    pass
pd.DataFrame(rows)""",
    ),
    "14_NonlinearModelComparison.ipynb": block(
        "Use a complete model-selection workflow and communicate a defensible decision.",
        [
            ("Exercise 1 — Define a comparison", "Add one reasonable variation to at least two candidates in `models` (for example, a shallower tree and a more-regularized logistic regression). Explain why your choices are plausible before running cross-validation."),
            ("Exercise 2 — Cross-validate and select", "Run the same stratified cross-validation procedure for every candidate. Make a table with mean validation accuracy and standard deviation. Select a model using the table and one non-score consideration."),
            ("Exercise 3 — Lock the choice", "Refit only your selected model on all training data and evaluate it once on the test data. Write a short results statement that distinguishes the cross-validation evidence from the final test result."),
        ],
        """# Starter: add candidates here; keep any preprocessing inside a Pipeline.
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

exercise_models = {
    "tree depth 2": DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE),
    # TODO: add at least two more candidates.
}
# TODO: use the existing `cv` object to compare every candidate on X_train, y_train.
""",
    ),
    "15_IntroToPytorch.ipynb": block(
        "Use tensors, autograd, and a manual optimization loop to fit and evaluate a small model.",
        [
            ("Exercise 1 — Inspect gradients", "Create a scalar tensor `w` with `requires_grad=True` and define a differentiable loss such as `(3*w - 6)**2`. Compute the gradient with `backward()`, then explain what the sign of the gradient says about a gradient-descent update."),
            ("Exercise 2 — Fit a line", "Generate noisy data from a line of your choice. Starting from random values of `a` and `b`, adapt the notebook's gradient-descent loop to fit `y_hat = a*x + b`. Plot the data and fitted line, and report the learned parameters."),
            ("Exercise 3 — Investigate a learning rate", "Repeat the fit with a learning rate that is much smaller and one that is much larger. Describe what happens to loss and explain why learning rate is an optimization choice rather than a cosmetic setting."),
        ],
        """# Starter: use a fresh scalar and clear its gradient before another backward pass.
w = torch.tensor(0.0, requires_grad=True)
loss = (3 * w - 6) ** 2
loss.backward()
print("loss:", loss.item(), "gradient:", w.grad.item())
# TODO: take one manual gradient-descent step and verify that the loss decreases.
""",
    ),
    "17_PyTorch_NN.ipynb": block(
        "Extend the neural-network workflow to a small classification problem and compare its behavior with a simpler model.",
        [
            ("Exercise 1 — Make nonlinear data", "Create a two-feature binary classification data set for which a straight line is not a natural separator (for example, concentric circles). Plot the observations by class."),
            ("Exercise 2 — Train an MLP", "Build a PyTorch network with one hidden layer and a nonlinear activation. Train it with a suitable binary classification loss, then report training accuracy. Change the hidden-layer width once and compare results."),
            ("Exercise 3 — Explain the architecture", "In 3--4 sentences, explain what the hidden layer and activation function add beyond a single linear layer. Include one limitation of evaluating only training accuracy."),
        ],
        """# Starter: the lesson already imports torch, nn, and F.
from sklearn.datasets import make_circles

X_np, y_np = make_circles(n_samples=300, noise=0.10, factor=0.45, random_state=232)
X_ex = torch.tensor(X_np, dtype=torch.float32)
y_ex = torch.tensor(y_np.reshape(-1, 1), dtype=torch.float32)
# TODO: define an nn.Sequential model, loss function, optimizer, and training loop.
""",
    ),
    "18_LogisticReg.ipynb": block(
        "Practice the connection between a linear score, a sigmoid output, binary cross-entropy, and classification decisions.",
        [
            ("Exercise 1 — Generate and visualize data", "Create a two-feature binary data set with `make_classification` or adapt the notebook's synthetic-data generator. Plot the points and state whether a linear boundary seems plausible."),
            ("Exercise 2 — Train logistic regression", "Fit a one-layer PyTorch logistic-regression model using a sigmoid-compatible binary loss. Report loss and accuracy, then display five predicted probabilities alongside true labels."),
            ("Exercise 3 — Compare a nonlinear alternative", "Add one hidden layer with a nonlinear activation. Compare the two models on a train/test split and explain whether the added flexibility appears justified by the evidence."),
        ],
        """# Starter: make a reproducible two-feature classification problem.
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X_np, y_np = make_classification(n_samples=300, n_features=2, n_redundant=0,
                                 n_informative=2, class_sep=1.0, random_state=232)
# TODO: create train/test tensors, fit a one-layer model, and evaluate probabilities.
""",
    ),
    "19_DefiningNNs.ipynb": block(
        "Extend the MNIST workflow while keeping training, evaluation, and error analysis distinct.",
        [
            ("Exercise 1 — Modify one architecture choice", "Change exactly one architectural choice in the notebook model: hidden-layer width, number of hidden layers, or activation function. Predict how it may affect parameter count, training time, or accuracy before training."),
            ("Exercise 2 — Evaluate by class", "After training, compute a confusion matrix or per-class accuracy on the test loader. Identify one pair of digits the model confuses and display several examples from that error category."),
            ("Exercise 3 — Reflect on generalization", "Compare training and test accuracy. Does the difference suggest underfitting, overfitting, or neither clearly? Propose one next experiment and explain what result would support your hypothesis."),
        ],
        """# Starter: collect predictions without computing gradients.
all_true, all_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in testset:
        logits = model(X_batch.view(-1, 784))
        all_pred.extend(torch.argmax(logits, dim=1).tolist())
        all_true.extend(y_batch.tolist())
# TODO: use sklearn.metrics.ConfusionMatrixDisplay or a pandas table to summarize errors.
""",
    ),
    "20_IntroGoogleColab_FastAI.ipynb": block(
        "Use Colab or a local environment to complete a small, reproducible fastai image-classification workflow.",
        [
            ("Exercise 1 — Set up the environment", "Open a copy in Colab, confirm that a GPU is available if one is assigned, and install/import fastai if necessary. Record the runtime type and one reproducibility choice, such as a random seed or a train/validation split."),
            ("Exercise 2 — Train a small classifier", "Use fastai's MNIST sample data (or another instructor-approved built-in data set) to build `DataLoaders`, call `cnn_learner`, and fine-tune for a small number of epochs. Record the validation metric after each epoch."),
            ("Exercise 3 — Inspect predictions", "Use `ClassificationInterpretation.from_learner` to view a confusion matrix and several high-loss examples. State one pattern you observe and one change you would test next."),
        ],
        """# Run this in Colab if fastai is not already available.
# !pip -q install fastai

from fastai.vision.all import *
path = untar_data(URLs.MNIST_SAMPLE)
dls = ImageDataLoaders.from_folder(path, train="train", valid="valid", seed=42,
                                   item_tfms=Resize(28), bs=64)
# TODO: create a cnn_learner, fine_tune it, and inspect its predictions.
""",
    ),
}


def main():
    for filename, cells in EXERCISES.items():
        path = MODULE / filename
        notebook = json.loads(path.read_text(encoding="utf-8"))
        notebook["cells"] = [
            cell for cell in notebook["cells"] if TAG not in cell.get("metadata", {}).get("tags", [])
        ]
        notebook["cells"].extend(cells)
        path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
