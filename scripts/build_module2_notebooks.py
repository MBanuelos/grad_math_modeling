"""Expand and normalize the student-facing Module 2 notebooks.

Run from the repository root with `python scripts/build_module2_notebooks.py`.
It is intentionally idempotent: it repairs Markdown LaTeX escapes and adds
the marked extension cells only once.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Lectures" / "Module2_DataSimilarity"


def source(text: str) -> list[str]:
    return [line + "\n" for line in text.strip().splitlines()]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {"module2_extension": True}, "source": source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {"module2_extension": True}, "outputs": [], "source": source(text),
    }


EXTENSIONS = {
    "01_SupervisedLearning.ipynb": [
        markdown(r"""## Part 3 — Residuals and the loss function

A **residual** is $e_i = y_i - \hat y_i$: the signed vertical distance from an observation to a prediction. Squaring residuals makes positive and negative errors comparable and penalizes large misses. Mean squared error is

$$MSE = \frac{1}{n}\sum_{i=1}^n (y_i - \hat y_i)^2.$$

Least squares finds coefficients that minimize this quantity. Gradient descent reaches the same goal iteratively by taking small steps opposite the gradient of the loss."""),
        code("""training_predictions = model.predict(X_train)
residuals = y_train - training_predictions
print(f"training MSE: {mean_squared_error(y_train, training_predictions):.2f}")
plt.axhline(0, color="black", linewidth=1)
plt.scatter(training_predictions, residuals)
plt.xlabel("predicted response"); plt.ylabel("residual (actual − predicted)")
plt.show()"""),
        markdown("""### Check your understanding

1. What would a residual plot look like if the model underpredicted large values?
2. Why does MSE react strongly to outliers?
3. Does a zero training residual prove perfect prediction on new data?

> **YOUR ANSWER:**"""),
        markdown("""## Part 4 — Regression versus classification

Regression predicts a numerical value, such as temperature or sale price. Classification predicts a label, such as spam/not spam. For a binary classifier, changing the probability threshold trades false positives against false negatives; accuracy is therefore not always a sufficient success measure."""),
        markdown("""### Exit ticket

Describe one supervised-learning problem from your field. Name its features, target, problem type, and one especially costly error.

> **YOUR ANSWER:**"""),
    ],
    "02_NLP_TFIDF.ipynb": [
        markdown(r"""## Part 0 — Build a bag-of-words matrix by hand

Start with two documents: A = `cats chase mice` and B = `dogs chase cats`. Using vocabulary `[cats, chase, dogs, mice]`, A becomes $[1,1,0,1]$ and B becomes $[1,1,1,0]$. The rows are documents and the columns are vocabulary terms. This **document–term matrix** retains counts, but not word order, grammar, or most meaning."""),
        code("""tiny_documents = ["cats chase mice", "dogs chase cats"]
tiny_vectorizer = CountVectorizer()
tiny_counts = tiny_vectorizer.fit_transform(tiny_documents)
pd.DataFrame(tiny_counts.toarray(), index=["A", "B"],
             columns=tiny_vectorizer.get_feature_names_out())"""),
        markdown("""### Pair practice — predict before running

Write the document–term row for `mice chase mice`. Would `cats chase mice` and `mice chase cats` have identical bag-of-words vectors? What limitation does that reveal?

> **YOUR ANSWER:**"""),
        markdown(r"""## Part 2 — Read the TF–IDF formula

Term frequency measures local importance in one document; inverse document frequency downweights terms that occur throughout the corpus. In a simplified form,

$$\operatorname{tfidf}(t,d)=\operatorname{tf}(t,d)\log\left(\frac{N}{df(t)}\right).$$

Here $N$ is the number of documents and $df(t)$ counts the documents containing $t$. Library implementations may apply smoothing and normalization, so their numerical values can differ from hand calculations. The formula is **TF–IDF**, not “tfid”: it combines term frequency and inverse document frequency."""),
        code("""for row_number, document in enumerate(documents):
    strongest = tfidf_frame.iloc[row_number].nlargest(3)
    print(f"Document {row_number + 1}: {document}")
    print(strongest.round(2).to_dict())"""),
        markdown("""## Part 3 — Preprocessing is a modeling choice

Lowercasing merges `Excellent` and `excellent`. Stop-word removal can save space, but removing `not` can harm sentiment analysis. Bigrams preserve short phrases such as `not good`, while producing a larger, sparser vocabulary.

### Practice

Compare the unigram and bigram feature matrices. Find one bigram that adds information a unigram loses. Propose a preprocessing rule for product reviews and one possible downside.

> **YOUR ANSWER:**"""),
        markdown("""## Vocabulary checkpoint

- **Corpus:** the full document collection.
- **Vocabulary:** the chosen unique terms/features.
- **Token:** a unit, often a word or short phrase.
- **Sparse matrix:** a matrix containing mostly zeros.

Which of these changes when a new document has a new word? Which changes when an old word is repeated?

> **YOUR ANSWER:**"""),
    ],
    "03_NLP_SentimentAnalysis.ipynb": [
        markdown("""## Part 0 — From a score to a decision

Logistic regression predicts a probability between 0 and 1, then a threshold (often 0.5) turns it into a class. Text features are TF–IDF values, while sentiment labels are human measurements with uncertainty—not unquestionable facts."""),
        code("""examples = ["excellent and useful", "cold and disappointing", "not bad"]
for review, probability in zip(examples, sentiment_model.predict_proba(examples)[:, 1]):
    print(f"{probability:.2f} positive probability — {review}")"""),
        markdown("""### Prediction practice

Predict each example’s label before running the cell. If the model disagrees, consider negation, unfamiliar words, and ambiguity.

> **YOUR ANSWER:**"""),
        markdown("""## Part 3 — Baselines and class balance

A classifier should outperform a sensible baseline. If 90% of reviews are positive, an always-positive classifier earns 90% accuracy while learning nothing about negative reviews. Precision asks whether positive predictions are correct; recall asks whether truly positive examples are found. The right metric depends on the cost of each error."""),
        code("""majority_label = int(np.mean(y_train) >= 0.5)
print(f"majority-class baseline accuracy: {np.mean(y_test == majority_label):.2f}")
print(f"model accuracy: {np.mean(sentiment_model.predict(X_test) == y_test):.2f}")"""),
        markdown("""### Discussion and responsibility

If negative reviews are sent to a support team, which error is worse: flagging a positive review or missing a negative review? Name one population, dialect, or setting underrepresented by this toy dataset, and the evaluation data needed before deployment.

> **YOUR ANSWER:**"""),
        markdown("""## Responsible interpretation

Sentiment systems can mishandle sarcasm, quoted language, dialect, genre, and context-dependent terms. Test on representative held-out data, document annotation rules, and retain a human-review or correction route when errors can affect people."""),
    ],
    "04_DimensionReduction_PCA.ipynb": [
        markdown(r"""## Part 0 — Centering, scaling, and directions of variation

PCA finds directions in which observations vary most. Centering subtracts each feature’s mean; scaling is often important when units differ. For centered $X$, the covariance matrix is

$$C=\frac{1}{n-1}X^TX.$$

Its eigenvectors are principal directions, and its eigenvalues quantify variance along them. SVD finds related directions without first building $C$."""),
        code("""scaled_digits = StandardScaler().fit_transform(X_digits)
pca_all = PCA().fit(scaled_digits)
cumulative_variance = np.cumsum(pca_all.explained_variance_ratio_)
plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance)
plt.axhline(0.90, color="black", linestyle="--", label="90% variance")
plt.xlabel("number of components"); plt.ylabel("cumulative explained variance")
plt.legend(); plt.show()"""),
        markdown("""### Written response 2

Estimate the fewest components that preserve 90% of explained variance. Why can a two-dimensional plot still be useful even if it preserves much less?

> **YOUR ANSWER:**"""),
        markdown("""## Part 3 — Reconstruction and information loss

Projection to fewer components is lossy. `inverse_transform` maps a reduced point back to the original feature space, producing an approximation. Broad structure can remain while fine detail disappears—the same principle behind low-rank image compression."""),
        code("""scaler = StandardScaler().fit(X_digits)
for n_components in [2, 10, 30]:
    reducer = PCA(n_components=n_components).fit(scaled_digits)
    reconstruction = scaler.inverse_transform(reducer.inverse_transform(reducer.transform(scaled_digits[[0]])))
    plt.imshow(reconstruction.reshape(8, 8), cmap="gray_r")
    plt.title(f"reconstruction: {n_components} PCs"); plt.axis("off"); plt.show()"""),
        markdown("""### Practice and caution

Which details disappear first in the reconstructions? Why is PCA for visualization different from PCA before regression? Remember: PCA does not use $y$, so high-variance directions are not automatically the most predictive. Fit scaling and PCA on training data only to avoid leakage.

> **YOUR ANSWER:**"""),
    ],
    "05_Similarity_SVD.ipynb": [
        markdown("""## Part 0 — Similarity depends on representation

Similarity depends on both an encoding and a metric. Euclidean distance measures straight-line separation; cosine similarity measures vector angle. Cosine similarity is useful when overall document length or rating activity should not dominate comparison."""),
        code("""from sklearn.metrics.pairwise import euclidean_distances
print("Cosine similarities")
print(np.round(cosine_similarity(documents), 2))
print("Euclidean distances")
print(np.round(euclidean_distances(documents), 2))"""),
        markdown("""### Partner check

Compare A to B and A to C using both metrics. Which comparison changes most? Explain with vector length and direction.

> **YOUR ANSWER:**"""),
        markdown(r"""## Part 3 — Reading an SVD

For a compact $m\times n$ decomposition $M=U\Sigma V^T$, columns of $U$ describe row patterns, diagonal values in $\Sigma$ measure their strength, and rows of $V^T$ describe column patterns. Keeping only the largest $k$ singular values gives a rank-$k$ approximation."""),
        code("""for k in range(1, min(M.shape) + 1):
    error = np.linalg.norm(M - rank_k_approximation(M, k), ord="fro")
    print(f"rank {k}: reconstruction error = {error:.3f}")"""),
        markdown("""### Written response 2

How does $k$ affect reconstruction error? Why is low rank a form of compression, and what might it remove from a real image or dataset?

> **YOUR ANSWER:**"""),
        markdown("""## Connect the ideas

TF–IDF documents, images, and ratings can all be matrices. Similarity compares rows or columns directly; SVD discovers latent directions; PCA uses related directions to reduce features. Next, we use these ideas with sparse ratings."""),
    ],
    "06_RecommendationSystems.ipynb": [
        markdown("""## Part 0 — Missing is not zero

An empty ratings-matrix cell usually means “not rated,” not “zero stars.” Treating all missing entries as zero distorts similarity. A simple baseline is a popularity list: easy and often strong, but it can overexpose already-popular items."""),
        code("""movie_summary = (ratings.groupby("movie_id")["rating"].agg(mean_rating="mean", rating_count="size")
                 .join(movies.set_index("movie_id")[["title", "genres"]]))
movie_summary.query("rating_count >= 100").sort_values("mean_rating", ascending=False).head(10)"""),
        markdown("""### Compare two recommenders

Compare cosine-neighbor recommendations for one movie with the popularity list. Which is more personalized? Which may be more dependable for a new user?

> **YOUR ANSWER:**"""),
        markdown("""## Part 3 — Evaluation and holdout data

An offline test hides ratings, builds recommendations from the remaining data, then evaluates hidden items. RMSE measures rating-prediction error; ranking metrics ask whether useful items appear near the top. Never evaluate only on ratings used to build the system."""),
        code("""eligible = subset["user_id"].value_counts()
eligible = eligible[eligible >= 2].index
held_out = subset[subset["user_id"].isin(eligible)].groupby("user_id", group_keys=False).sample(n=1, random_state=232)
training_rows = subset.drop(held_out.index)
print(f"training ratings: {len(training_rows):,}")
print(f"held-out ratings: {len(held_out):,}")"""),
        markdown("""### Design reflection

What is the cold-start problem for new users and movies? How can popularity create a feedback loop? Propose one control that makes recommendations easier for users to understand or correct.

> **YOUR ANSWER:**"""),
        markdown("""## Looking ahead

Neighborhood methods compare observed ratings directly. Latent-factor methods use low-rank structure to estimate preferences even when few ratings overlap. Neither determines what someone should watch: goals, safeguards, and feedback channels remain design choices."""),
    ],
}


# A second set of short sections provides enough room for discussion, group
# work, and an exit reflection in a 50–60 minute class period.
SECOND_EXTENSIONS = {
    "01_SupervisedLearning.ipynb": [
        markdown(r"""## Part 5 — One gradient-descent step

For simple regression, gradient descent updates each parameter using a learning rate $\eta$:

$$\beta_j \leftarrow \beta_j - \eta\frac{\partial MSE}{\partial \beta_j}.$$

A small $\eta$ makes slow progress; a large $\eta$ can overshoot the minimum. In practice, numerical libraries solve least squares efficiently, but the update rule explains how many modern learning algorithms improve a model."""),
        markdown("""### Small-group task

Sketch a curve of loss versus iteration for a learning rate that is too small, reasonable, and too large. What evidence would tell you to stop training?

> **YOUR ANSWER:**"""),
        markdown("""## Model-assessment checklist

Before trusting a supervised model, ask: Is the target measured reliably? Is the test set genuinely future or unseen data? Are key groups represented? Is the metric aligned with the decision? Could a feature leak information that would be unavailable at prediction time?

Choose one checklist question that matters for the diabetes example and explain why.

> **YOUR ANSWER:**"""),
        markdown("""## Lesson summary

Supervised learning connects labeled features to a target. Fitting is only the beginning: residuals, train/test splits, cross-validation, and context determine whether a model is credible."""),
    ],
    "02_NLP_TFIDF.ipynb": [
        markdown("""## Part 4 — Matrix shape and sparsity

The matrix has one row per document and one column per vocabulary term. Adding bigrams expands the number of possible columns faster than it expands the number of nonzero entries, so text matrices are usually sparse. Sparse storage lets us work with large corpora without recording every zero."""),
        markdown("""### Feature-design challenge

Would you use character n-grams, word unigrams, or word bigrams for detecting misspellings in product reviews? Make a choice, name one benefit, and name one tradeoff.

> **YOUR ANSWER:**"""),
        markdown("""## Limits of bag of words

Bag-of-words models treat `dog bites person` and `person bites dog` as nearly identical. They also struggle with long-range context and sarcasm. These limitations motivate embeddings and transformer models, but TF–IDF remains a transparent, fast baseline—and a useful feature representation for this course."""),
    ],
    "03_NLP_SentimentAnalysis.ipynb": [
        markdown("""## Part 4 — Read model coefficients carefully

Positive and negative coefficient lists show correlations in this small training sample. A high coefficient does not mean a word is always positive or negative; it may reflect topic, author style, or a coincidental pattern. Inspecting coefficients is a helpful diagnostic, not a causal explanation."""),
        markdown("""### Error-analysis protocol

For each mistake, record the review, true label, prediction, confidence, and a likely cause. Then group errors into themes—negation, ambiguity, missing vocabulary, labeling disagreement, or something else. Which theme would you address first, and how?

> **YOUR ANSWER:**"""),
        markdown("""## Lesson summary

The full text-classification pipeline is: collect and label text, split data, transform text to features, train a classifier, evaluate against a baseline, and inspect errors. Every step can introduce limitations worth documenting."""),
    ],
    "04_DimensionReduction_PCA.ipynb": [
        markdown("""## Part 4 — Choosing a number of components

There is no universal explained-variance cutoff. Fewer components simplify visualization and can reduce noise; more components preserve more information. Choose the number using the goal: a plot may need two components, while a predictive pipeline should compare candidates with cross-validation."""),
        markdown("""### Think–pair–share

Suppose one feature is measured in dollars and another is a fraction between 0 and 1. Predict what happens without scaling before PCA. Then explain why fitting the scaler before the train/test split would be data leakage.

> **YOUR ANSWER:**"""),
        markdown("""## Lesson summary

PCA rotates data to orthogonal directions ordered by variation. It can visualize, compress, and pre-process data, but it does not know the response variable or replace evaluation of a downstream task."""),
    ],
    "05_Similarity_SVD.ipynb": [
        markdown("""## Part 4 — A metric is a choice

Cosine similarity ignores magnitude, which is often desirable for documents but not always for ratings or purchases. Euclidean distance can be meaningful when a common scale and absolute differences matter. Select a metric based on the question, then test whether its neighbors make sense."""),
        markdown("""### Practice: choose a representation

For each case—song playlists, student exam scores, and short text messages—choose a possible vector representation and a similarity metric. Explain one limitation of your choice.

> **YOUR ANSWER:**"""),
        markdown("""## Lesson summary

Similarity measures direct relationships between encoded objects. SVD finds a compact set of directions that approximately reconstructs a matrix. Both depend on the data representation and can amplify its omissions."""),
    ],
    "06_RecommendationSystems.ipynb": [
        markdown("""## Part 4 — Content-based and collaborative signals

Content-based recommendation uses item attributes such as genres, descriptions, or TF–IDF features. Collaborative filtering uses patterns across users and items. Hybrid systems combine them: content can help a new item, while collaborative signals can discover connections not listed in metadata."""),
        markdown("""### Scenario discussion

A new user rates two science-fiction films highly. Which approach can recommend immediately: content-based, collaborative, or both? What additional information would make the first page of recommendations more useful?

> **YOUR ANSWER:**"""),
        markdown("""## Lesson summary

Recommendation is a prediction-and-ranking problem with sparse, incomplete feedback. Useful systems validate on held-out data and consider diversity, privacy, cold starts, popularity bias, and user control alongside numerical performance."""),
    ],
}


TOPIC_MODELING = [
    markdown(r"""## Part 5 — Topic modeling with dimension reduction

The TF–IDF notebook represented each document as a sparse row in a document–term matrix. We can reduce that high-dimensional matrix to a few latent directions. For sparse TF–IDF data, `TruncatedSVD` is usually preferred over ordinary PCA because it works directly with sparse matrices and does not require centering every zero. This approach is often called **latent semantic analysis (LSA)**.

Each latent direction is a possible *topic*: a weighted collection of terms that tend to occur together. Topics are discovered from word co-occurrence, so they need not be perfect labels or represent a single coherent idea."""),
    code("""from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

topic_documents = [
    "The telescope observed a distant galaxy and bright stars.",
    "Astronauts prepare a rocket for an orbit around Earth.",
    "A satellite sends images of planets and space.",
    "Cells use proteins to carry genetic information.",
    "The microscope shows bacteria in a tissue sample.",
    "Scientists study genes, cells, and evolution.",
    "The recipe uses tomatoes, herbs, and olive oil.",
    "Bake the bread until the crust is golden brown.",
    "The chef prepared a warm soup with vegetables.",
]

topic_vectorizer = TfidfVectorizer(stop_words="english")
document_term = topic_vectorizer.fit_transform(topic_documents)
print("document–term matrix shape:", document_term.shape)
pd.DataFrame(document_term.toarray(), columns=topic_vectorizer.get_feature_names_out()).round(2)"""),
    markdown("""### Before running the model

Read the nine documents and propose three human-assigned topic labels. Which words do you expect to be most helpful for distinguishing those groups? Which words might be shared across groups or too general to help?

> **YOUR ANSWER:**"""),
    code("""lsa = TruncatedSVD(n_components=3, random_state=232)
document_topics = lsa.fit_transform(document_term)
terms = topic_vectorizer.get_feature_names_out()

for topic_number, component in enumerate(lsa.components_, start=1):
    top_terms = terms[component.argsort()[-6:]][::-1]
    print(f"Topic {topic_number}: {', '.join(top_terms)}")

print()
print("Explained variance ratio:", lsa.explained_variance_ratio_.round(3))
pd.DataFrame(document_topics, columns=["topic_1", "topic_2", "topic_3"]).round(2)"""),
    markdown("""## Interpreting latent topics

The output names the *terms*, not the topics. You supply a tentative label after inspecting the top terms and the documents with high values on that component. For example, a component whose large terms include `rocket`, `orbit`, and `satellite` may reasonably be called a space topic. A document can have a mixture of topic scores rather than belonging to exactly one topic."""),
    code("""labels = ["space", "space", "space", "biology", "biology", "biology", "food", "food", "food"]
plt.figure(figsize=(7, 5))
for label in sorted(set(labels)):
    mask = np.array(labels) == label
    plt.scatter(document_topics[mask, 0], document_topics[mask, 1], label=label, s=70)
for index, text in enumerate(topic_documents):
    plt.annotate(str(index + 1), (document_topics[index, 0], document_topics[index, 1]))
plt.xlabel("latent topic direction 1")
plt.ylabel("latent topic direction 2")
plt.legend(title="human label")
plt.show()"""),
    markdown("""### ✏️ Topic-modeling practice

1. Give each discovered component a tentative topic label, using its top terms.
2. Does the two-dimensional plot separate the three human-labeled groups? Identify one document that may be mixed or ambiguous.
3. Change `n_components` to 2 and then 4. What changes in the top terms and explained variance?
4. Why should these automatically discovered topics be reviewed by a person before using them to summarize or make decisions about a corpus?

> **YOUR ANSWER:**"""),
    markdown("""## PCA, SVD, and topic modeling

PCA and truncated SVD share the dimension-reduction idea: find a small number of directions that retain important structure. PCA is commonly applied to centered dense features; truncated SVD is convenient for a sparse document–term matrix. Neither method “understands” language. Results depend on the corpus, preprocessing, number of components, and how people interpret the resulting term lists."""),
]


def repair_latex(text: str) -> str:
    """Undo Python's accidental escapes in Markdown written by the first generator."""
    replacements = {"\x07": r"\a", "\x08": r"\b", "\x0b": r"\v", "\x0c": r"\f", "\r": r"\r", "\t": r"\t"}
    for accidental, intended in replacements.items():
        text = text.replace(accidental, intended)
    # ``str.splitlines`` in the initial notebook writer discarded form-feed
    # and carriage-return characters, so these two formulas need their missing
    # LaTeX command initials restored explicitly.
    text = text.replace(
        r"\log\!\left(" + "\n" + r"rac{N}{df(t)}" + "\n" + r"ight)",
        r"\log\!\left(\frac{N}{df(t)}\right)",
    )
    text = text.replace(r"\cos(\theta) = " + "\n" + r"rac", r"\cos(\theta) = \frac")
    return text


def reorder_nlp_lesson(notebook: dict) -> None:
    """Put document-term-matrix intuition before any vectorizer or TF–IDF use."""
    if notebook["metadata"].get("nlp_opening_reordered"):
        return

    cells = notebook["cells"]

    def find(prefix: str) -> dict:
        return next(cell for cell in cells if "".join(cell["source"]).startswith(prefix))

    badge, title = cells[0], cells[1]
    manual = find("## Part 0 — Build a bag-of-words matrix by hand")
    manual_code = next(cell for cell in cells if "tiny_vectorizer = CountVectorizer()" in "".join(cell["source"]))
    manual_practice = find("### Pair practice — predict before running")
    introduction = find("## From documents to vectors")
    count_code = next(cell for cell in cells if "count_vectorizer = CountVectorizer" in "".join(cell["source"]))
    raw_counts = find("## Part 1 — TF–IDF in practice")
    tfidf_formula = find("## Part 2 — Read the TF–IDF formula")
    tfidf_code = next(cell for cell in cells if "tfidf_vectorizer = TfidfVectorizer" in "".join(cell["source"]))
    tfidf_response = find("### ✏️ Written response 1")
    preprocessing = find("## Part 2 — Choices are modeling decisions")
    bigram_code = next(cell for cell in cells if "bigram_vectorizer = TfidfVectorizer" in "".join(cell["source"]))
    practice = find("### Practice")
    sparsity = find("## Part 4 — Matrix shape and sparsity")
    feature_challenge = find("### Feature-design challenge")
    limits = find("## Limits of bag of words")
    summary = find("## Key ideas")
    vocabulary = find("## Vocabulary checkpoint")

    manual["source"] = source("""## Part 1 — Build a bag-of-words matrix by hand

Start with two short **documents**: A = `cats chase mice` and B = `dogs chase cats`. First choose a **vocabulary**, an ordered list of all words we will count: `[cats, chase, dogs, mice]`.

For each document, write a 1 when a vocabulary word occurs and 0 when it does not. Document A becomes $[1,1,0,1]$; B becomes $[1,1,1,0]$. Stacking the rows creates a **document–term matrix**: rows are documents, columns are vocabulary terms, and entries are counts. This is the bag-of-words model—it keeps which words occur and how often, but not word order, grammar, or most meaning.""")
    introduction["source"] = source("""## Part 2 — Build a document–term matrix with code

The hand-built table is small enough to inspect, but real collections can contain thousands of documents and terms. `CountVectorizer` automates the same three steps:

1. split each document into tokens (words);
2. create a shared vocabulary from the tokens;
3. count each vocabulary term in each document.

The code below uses four slightly longer documents. It removes common English stop words such as `the` and `and` so that the table emphasizes potentially informative words. Read the resulting table as you read the hand-built one: each row is a document and each column is a term.""")
    raw_counts["source"] = source("""## Part 3 — Why raw counts are not always enough

Raw counts treat every occurrence equally. But a word found in nearly every document, such as `excellent` in this toy corpus, may be less useful for telling documents apart than a word found in only one. TF–IDF changes the weights so that locally frequent but globally rare terms receive more emphasis.""")
    tfidf_formula["source"] = source(r"""## Part 4 — TF–IDF: downweight common words

Term frequency measures local importance in one document; inverse document frequency downweights terms that occur throughout the corpus. In a simplified form,

$$\operatorname{tfidf}(t,d)=\operatorname{tf}(t,d)\log\left(\frac{N}{df(t)}\right).$$

Here $N$ is the number of documents and $df(t)$ counts the documents containing $t$. Library implementations may apply smoothing and normalization, so their numerical values can differ from a hand calculation. The name **TF–IDF** combines term frequency and inverse document frequency.""")
    preprocessing["source"] = source("""## Part 5 — Preprocessing is a modeling choice

Lowercasing merges `Excellent` and `excellent`. Stop-word removal can save space, but removing `not` can harm sentiment analysis. Bigrams preserve short phrases such as `not good`, while producing a larger, sparser vocabulary.""")
    sparsity["source"] = source("""## Part 6 — Matrix shape and sparsity

The matrix has one row per document and one column per vocabulary term. Adding bigrams expands the number of possible columns faster than it expands the number of nonzero entries, so text matrices are usually sparse. Sparse storage lets us work with large corpora without recording every zero.""")

    notebook["cells"] = [
        badge, title, manual, manual_code, manual_practice, introduction, count_code,
        raw_counts, tfidf_formula, tfidf_code, tfidf_response, preprocessing, bigram_code,
        practice, vocabulary, sparsity, feature_challenge, limits, summary,
    ]
    notebook["metadata"]["nlp_opening_reordered"] = True


def main() -> None:
    for filename, extension_cells in EXTENSIONS.items():
        path = MODULE / filename
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "markdown":
                cell["source"] = source(repair_latex("".join(cell["source"])))
        if filename == "02_NLP_TFIDF.ipynb":
            reorder_nlp_lesson(notebook)
        if not any(cell.get("metadata", {}).get("module2_extension") for cell in notebook["cells"]):
            notebook["cells"].extend(extension_cells)
        if not any(cell.get("metadata", {}).get("module2_extension_round2") for cell in notebook["cells"]):
            for cell in SECOND_EXTENSIONS[filename]:
                cell["metadata"]["module2_extension_round2"] = True
            notebook["cells"].extend(SECOND_EXTENSIONS[filename])
        if filename == "04_DimensionReduction_PCA.ipynb" and not any(
            cell.get("metadata", {}).get("module2_topic_modeling") for cell in notebook["cells"]
        ):
            for cell in TOPIC_MODELING:
                cell["metadata"]["module2_topic_modeling"] = True
            notebook["cells"].extend(TOPIC_MODELING)
        path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
