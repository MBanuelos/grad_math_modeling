"""Build shorter, self-contained Module 0 notebooks from the original day files.

The original notebooks are intentionally retained as source material.  This script
creates the public lesson notebooks (no audience suffix) and matching instructor
editions (``_instructor`` suffix) used by the course.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Lectures" / "Module0_TimeSeries"
COLAB_BADGE = "https://colab.research.google.com/assets/colab-badge.svg"
COLAB_BASE = (
    "https://colab.research.google.com/github/mbanuelos/"
    "grad_math_modeling/blob/main/Lectures/Module0_TimeSeries"
)


COMMON_TS_SETUP = """# Shared setup from the preceding lesson
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
%matplotlib inline

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, parse_dates=['Month'])
df.columns = ['Date', 'Passengers']
"""


DAY2_SETUP = """# Shared setup from the preceding lesson
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

np.random.seed(42)
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
%matplotlib inline

def simulate_ar1(phi, n=200, sigma=1.0, x0=0.0):
    # Simulate x[t] = phi*x[t-1] + noise.
    x = np.zeros(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = phi * x[t - 1] + np.random.normal(0, sigma)
    return x
"""


DAY2_LAG_SETUP = """# Shared setup from the preceding lesson
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LinearRegression

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, parse_dates=['Month'])
df.columns = ['Date', 'Passengers']
df['Log_Passengers'] = np.log(df['Passengers'])
df['Log_Diff'] = df['Log_Passengers'].diff()
series = df['Log_Diff'].dropna()

# The preceding lesson selected p=12 from the seasonal PACF signature.
p = 12
result = AutoReg(series, lags=p).fit()
"""


DAY3_BASE_SETUP = """# Shared forecasting setup from the preceding lesson
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AutoReg
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
%matplotlib inline

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, parse_dates=['Month'])
df.columns = ['Date', 'Passengers']
df['Log_Passengers'] = np.log(df['Passengers'])
df['Log_Diff'] = df['Log_Passengers'].diff()
series = df['Log_Diff'].dropna().values

def make_lag_matrix(series, n_lags):
    series = np.asarray(series)
    X, y = [], []
    for t in range(n_lags, len(series)):
        X.append(series[t - n_lags:t])
        y.append(series[t])
    return np.asarray(X), np.asarray(y)

N_LAGS = 12
X, y = make_lag_matrix(series, N_LAGS)
split = int(len(X) * 0.80)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

naive_preds = np.concatenate([[y_train[-1]], y_test[:-1]])
train_series = series[:split + N_LAGS]
ar_result = AutoReg(train_series, lags=N_LAGS).fit()
ar_preds = ar_result.predict(
    start=len(train_series),
    end=len(train_series) + len(y_test) - 1,
    dynamic=False,
)
"""


DAY3_CLASSICAL_SETUP = DAY3_BASE_SETUP + """

rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

gb_model = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, random_state=42
)
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)

def compute_metrics(y_true, y_pred, label='Model'):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f'{label:<30}  RMSE={rmse:.5f}   MAE={mae:.5f}')
    return {'RMSE': rmse, 'MAE': mae}

results = {
    'Naive Baseline': compute_metrics(y_test, naive_preds, 'Naive Baseline'),
    'AR(12)': compute_metrics(y_test, ar_preds, 'AR(12)'),
    'Random Forest': compute_metrics(y_test, rf_preds, 'Random Forest'),
    'Gradient Boosting': compute_metrics(y_test, gb_preds, 'Gradient Boosting'),
}
"""


DAY3_SEQUENCE_SETUP = DAY3_CLASSICAL_SETUP + """

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_test_s = scaler_X.transform(X_test)
y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

class MLPForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

mlp = MLPForecaster(input_size=N_LAGS, hidden_size=32)
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
for _ in range(300):
    mlp.train()
    optimizer.zero_grad()
    loss = loss_fn(mlp(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()

mlp.eval()
with torch.no_grad():
    mlp_preds_s = mlp(X_test_t).numpy()
mlp_preds = scaler_y.inverse_transform(mlp_preds_s.reshape(-1, 1)).ravel()
results['MLP'] = compute_metrics(y_test, mlp_preds, 'MLP')
"""


DAY4_DATA_SETUP = """# Shared setup from the preceding lesson
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(42)
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
%matplotlib inline

url = 'https://raw.githubusercontent.com/christophM/interpretable-ml-book/master/data/bike-sharing-daily.csv'
df = pd.read_csv(url, parse_dates=['dteday'])
df = df[['dteday', 'cnt', 'temp', 'hum', 'windspeed', 'workingday', 'holiday']].copy()
df.columns = ['Date', 'Rentals', 'Temp', 'Humidity', 'Windspeed', 'WorkingDay', 'Holiday']
df['Rentals_Diff'] = df['Rentals'].diff()
"""


DAY4_NEURAL_SETUP = DAY4_DATA_SETUP + """

%pip install -q pytorch-lightning

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as L
torch.manual_seed(42)
L.seed_everything(42)

def make_lag_matrix(series, n_lags):
    series = np.asarray(series)
    X, y = [], []
    for t in range(n_lags, len(series)):
        X.append(series[t - n_lags:t])
        y.append(series[t])
    return np.asarray(X), np.asarray(y)

def make_lag_matrix_exog(series, exog, n_lags):
    series, exog = np.asarray(series), np.asarray(exog)
    X, y = [], []
    for t in range(n_lags, len(series)):
        X.append(np.append(series[t - n_lags:t], exog[t]))
        y.append(series[t])
    return np.asarray(X), np.asarray(y)

def compute_metrics(y_true, y_pred, label='Model'):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f'{label:<30}  RMSE={rmse:.4f}   MAE={mae:.4f}')
    return {'RMSE': rmse, 'MAE': mae}

N_LAGS = 7
TRAIN_FRAC = 0.80
target_series = df['Rentals_Diff'].dropna().values
temp_aligned = df['Temp'].iloc[1:].values

X_uni, y_uni = make_lag_matrix(target_series, N_LAGS)
split = int(len(X_uni) * TRAIN_FRAC)
X_uni_train, X_uni_test = X_uni[:split], X_uni[split:]
y_uni_train, y_uni_test = y_uni[:split], y_uni[split:]
lr_uni = LinearRegression().fit(X_uni_train, y_uni_train)
lr_uni_preds = lr_uni.predict(X_uni_test)

X_exog, y_exog = make_lag_matrix_exog(target_series, temp_aligned, N_LAGS)
split_exog = int(len(X_exog) * TRAIN_FRAC)
X_exog_train, X_exog_test = X_exog[:split_exog], X_exog[split_exog:]
y_exog_train, y_exog_test = y_exog[:split_exog], y_exog[split_exog:]
lr_exog = LinearRegression().fit(X_exog_train, y_exog_train)
lr_exog_preds = lr_exog.predict(X_exog_test)

results = {
    'Univariate (lags only)': compute_metrics(
        y_uni_test, lr_uni_preds, 'Univariate Linear'
    ),
    'Exogenous (lags + Temp)': compute_metrics(
        y_exog_test, lr_exog_preds, 'Exogenous Linear'
    ),
}
"""


@dataclass(frozen=True)
class Lesson:
    number: int
    filename: str
    title: str
    source_day: int
    student_range: tuple[int, int]
    instructor_range: tuple[int, int] | None
    minutes: str
    objectives: tuple[str, ...]
    prerequisite: str | None = None
    setup: str | None = None


LESSONS = (
    Lesson(1, "01_TimeSeriesFoundations", "Time Series Foundations: Visualization and Decomposition", 1, (1, 15), (1, 13), "75–90", (
        "Explain why temporal ordering changes how data should be analyzed.",
        "Identify trend, seasonality, and changing variance in a time plot.",
        "Interpret a multiplicative seasonal decomposition.",
    )),
    Lesson(2, "02_StationarityAndTransformations", "Stationarity: Rolling Statistics, ADF, and Transformations", 1, (15, 28), (13, 24), "75–90", (
        "Use rolling statistics to diagnose changes in mean and variance.",
        "Interpret an Augmented Dickey–Fuller test.",
        "Apply log and differencing transformations for stationarity.",
    ), "Lesson 1", COMMON_TS_SETUP),
    Lesson(3, "03_AutocorrelationAndDiagnostics", "Autocorrelation and Reusable Time-Series Diagnostics", 1, (28, 40), (24, 34), "75–90", (
        "Interpret ACF and PACF plots and their confidence bands.",
        "Recognize seasonal dependence at meaningful lags.",
        "Combine visual, ADF, and correlation evidence in a diagnostic workflow.",
    ), "Lessons 1–2", COMMON_TS_SETUP + "\ndf['Log_Passengers'] = np.log(df['Passengers'])\ndf['Log_Diff'] = df['Log_Passengers'].diff()\n"),
    Lesson(4, "04_AutoregressiveModels", "Autoregressive Models: Intuition and Stability", 2, (1, 15), (1, 15), "80–95", (
        "Simulate AR(1) and AR(2) processes.",
        "Explain how autoregressive coefficients control memory and stability.",
        "Connect AR order to ACF and PACF signatures.",
    ), "Lessons 1–3"),
    Lesson(5, "05_AROrderAndFitting", "Selecting and Fitting Autoregressive Models", 2, (15, 25), (15, 23), "70–85", (
        "Prepare a stationary real-world series for autoregression.",
        "Use the PACF to propose an AR order.",
        "Fit and interpret an autoregressive model.",
    ), "Lesson 4", DAY2_SETUP),
    Lesson(6, "06_LagEmbeddedFeatures", "Lag-Embedded Feature Matrices", 2, (25, 42), (23, 41), "85–100", (
        "Reframe autoregression as supervised learning.",
        "Construct univariate and multivariate lag matrices.",
        "Relate an AR model to linear regression on lagged features.",
    ), "Lessons 4–5", DAY2_LAG_SETUP),
    Lesson(7, "07_TimeAwareEvaluation", "Time-Aware Evaluation and Forecasting Baselines", 3, (1, 14), (1, 17), "75–90", (
        "Create a chronological train/test split without leakage.",
        "Implement a naive last-value forecast.",
        "Fit an AR baseline using training data only.",
    ), "Lesson 6"),
    Lesson(8, "08_TreeBasedForecasting", "Tree-Based Forecasting and Error Metrics", 3, (14, 25), (17, 24), "75–90", (
        "Fit random-forest and gradient-boosting forecasters to lag features.",
        "Compute and interpret RMSE and MAE.",
        "Compare learned models with naive and AR baselines.",
    ), "Lesson 7", DAY3_BASE_SETUP),
    Lesson(9, "09_MLPForecasting", "Multilayer Perceptrons for Time-Series Forecasting", 3, (25, 32), (24, 29), "80–95", (
        "Scale time-series features without leaking test information.",
        "Implement and train an MLP forecaster in PyTorch.",
        "Evaluate neural forecasts on the original target scale.",
    ), "Lessons 7–8", DAY3_CLASSICAL_SETUP),
    Lesson(10, "10_SequenceModels", "Sequence Models: RNNs, LSTMs, and Model Comparison", 3, (32, 45), (29, 39), "90–110", (
        "Explain hidden state, recurrence, and vanishing gradients.",
        "Implement and train RNN and LSTM forecasters.",
        "Compare classical, tree-based, MLP, RNN, and LSTM forecasts.",
    ), "Lessons 7–9", DAY3_SEQUENCE_SETUP),
    Lesson(11, "11_ExogenousVariables", "Exogenous Variables and the Bike-Rental Series", 4, (1, 20), None, "75–90", (
        "Identify plausible external predictors for a forecasting problem.",
        "Explore a new target and its candidate exogenous variables.",
        "Repeat the stationarity workflow on daily data.",
    ), "Lessons 1–3"),
    Lesson(12, "12_ExogenousLagMatrices", "Exogenous-Aware Lag Matrices", 4, (20, 34), None, "80–95", (
        "Build a univariate forecasting baseline on daily data.",
        "Align same-time exogenous information with lagged target features.",
        "Compare univariate and exogenous-aware linear models.",
    ), "Lesson 11", DAY4_DATA_SETUP),
    Lesson(13, "13_ExogenousNeuralForecasting", "Exogenous Neural Forecasting and Module Synthesis", 4, (34, 59), None, "90–110", (
        "Organize PyTorch training with Dataset, DataLoader, and LightningModule.",
        "Train an MLP with lagged and exogenous features.",
        "Synthesize how different forecasting models represent memory.",
    ), "Lessons 9–12", DAY4_NEURAL_SETUP),
)


SKIP_CELLS = {
    # Full-day schedules conflict with the new 75–110 minute lesson structure.
    (4, True): {2},
    (7, True): {1},
    # Lightning is introduced in Lesson 13, not during initial data exploration.
    (11, False): {5},
}


DAY4_INTRO_IMPORTS = """# Imports used in this lesson
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
%matplotlib inline

print('All imports successful!')
"""


CODE_REPLACEMENTS = {
    (11, False, 6): DAY4_INTRO_IMPORTS,
}


def intro_cell(lesson: Lesson, instructor: bool) -> nbformat.NotebookNode:
    audience = "Instructor edition" if instructor else "Student edition"
    warning = (
        "\n> **Do not distribute to students.** This edition contains solutions, "
        "teaching notes, and model answers.\n"
        if instructor
        else ""
    )
    prerequisite = (
        f"\n**Prerequisite:** {lesson.prerequisite}  \n" if lesson.prerequisite else ""
    )
    goals = "\n".join(f"- {objective}" for objective in lesson.objectives)
    source = f"Original Day {lesson.source_day}"
    text = f"""# {'📋 INSTRUCTOR NOTEBOOK — ' if instructor else ''}{lesson.title}

**Module 0 · Lesson {lesson.number} of {len(LESSONS)} · {audience}**  
**Estimated class time:** {lesson.minutes} minutes  
**Source sequence:** {source}  
{prerequisite}{warning}
## Learning objectives

By the end of this lesson, you should be able to:

{goals}
"""
    cell = nbformat.v4.new_markdown_cell(text.strip())
    audience_code = "i" if instructor else "s"
    cell.id = f"lesson-{lesson.number:02d}-intro-{audience_code}"
    return cell


def colab_cell(lesson: Lesson) -> nbformat.NotebookNode:
    markdown = (
        f"[![Open In Colab]({COLAB_BADGE})]"
        f"({COLAB_BASE}/{lesson.filename}.ipynb)"
    )
    cell = nbformat.v4.new_markdown_cell(markdown)
    cell.id = "open-in-colab"
    cell.metadata["tags"] = ["colab-button"]
    return cell


def setup_cells(
    code: str, lesson: Lesson, instructor: bool
) -> list[nbformat.NotebookNode]:
    audience_code = "i" if instructor else "s"
    explanation = nbformat.v4.new_markdown_cell(
        "## Setup for this lesson\n\n"
        "This cell recreates the data and completed prerequisites from earlier lessons, "
        "so this notebook can be run in a fresh kernel."
    )
    explanation.id = f"lesson-{lesson.number:02d}-setup-note-{audience_code}"
    cell = nbformat.v4.new_code_cell(code.rstrip() + "\n")
    cell.id = f"lesson-{lesson.number:02d}-setup-code-{audience_code}"
    cell.metadata["tags"] = ["setup"]
    return [explanation, cell]


def clean_copied_cell(
    cell: nbformat.NotebookNode,
    lesson: Lesson,
    instructor: bool,
    source_index: int,
) -> nbformat.NotebookNode:
    cell = copy.deepcopy(cell)
    audience_code = "i" if instructor else "s"
    cell.id = f"lesson-{lesson.number:02d}-src-{source_index:03d}-{audience_code}"
    replacement = CODE_REPLACEMENTS.get((lesson.number, instructor, source_index))
    if replacement is not None:
        cell.source = replacement
        cell.outputs = []
        cell.execution_count = None

    if cell.cell_type == "markdown":
        if lesson.number == 11:
            cell.source = cell.source.replace(
                "Today you will:", "Across Lessons 11–13 you will:"
            ).replace("## Part 4.5 - Classes", "## Python Classes: A Short Refresher")
    if cell.cell_type == "code":
        cell.source = cell.source.replace(
            "Duration: ~5.5 hours of lecture + lab",
            "This material is taught across Lessons 7–10.",
        )
        cell.outputs = [output for output in cell.outputs if output.output_type != "error"]
        if not cell.outputs:
            cell.execution_count = None
    return cell


def build_notebook(lesson: Lesson, instructor: bool) -> nbformat.NotebookNode:
    suffix = "instructor" if instructor else "student"
    source_path = MODULE / f"day{lesson.source_day}_{suffix}.ipynb"
    source = nbformat.read(source_path, as_version=4)
    cell_range = lesson.instructor_range if instructor else lesson.student_range
    if cell_range is None:
        raise ValueError(f"No instructor source exists for lesson {lesson.number}")

    start, stop = cell_range
    cells = [intro_cell(lesson, instructor)]
    if not instructor:
        cells.insert(0, colab_cell(lesson))
    if lesson.setup:
        cells.extend(setup_cells(lesson.setup, lesson, instructor))
    skip = SKIP_CELLS.get((lesson.number, instructor), set())
    cells.extend(
        clean_copied_cell(source.cells[index], lesson, instructor, index)
        for index in range(start, stop)
        if index not in skip
    )

    notebook = nbformat.v4.new_notebook(cells=cells, metadata=copy.deepcopy(source.metadata))
    notebook.metadata["course"] = {
        "module": 0,
        "lesson": lesson.number,
        "audience": "instructor" if instructor else "student",
        "source_notebook": source_path.name,
    }

    # The Day 4 source stores ipywidget models directly under the MIME key.
    # nbconvert expects the standard wrapper with state/version fields.
    widget_mime = "application/vnd.jupyter.widget-state+json"
    widget_metadata = notebook.metadata.get("widgets", {})
    widget_state = widget_metadata.get(widget_mime)
    if widget_state and "state" not in widget_state:
        widget_metadata[widget_mime] = {
            "state": widget_state,
            "version_major": 2,
            "version_minor": 0,
        }

    nbformat.validate(notebook)
    return notebook


def main() -> None:
    for lesson in LESSONS:
        student_path = MODULE / f"{lesson.filename}.ipynb"
        nbformat.write(build_notebook(lesson, instructor=False), student_path)
        print(f"wrote {student_path.relative_to(ROOT)}")

        if lesson.instructor_range is not None:
            instructor_path = MODULE / f"{lesson.filename}_instructor.ipynb"
            nbformat.write(build_notebook(lesson, instructor=True), instructor_path)
            print(f"wrote {instructor_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
