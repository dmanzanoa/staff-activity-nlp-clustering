"""
activity_clustering.py
======================

This module provides a small library and command‑line interface for
cluster analysis of text‑based staff activity descriptions.  It was
inspired by a data science project that leveraged natural language
processing (NLP) and unsupervised learning to analyse and cluster
records of duties performed by hourly paid employees.  The resulting
insights informed a policy change that transitioned a large proportion
of casual staff to full‑time contracts.

The code is deliberately data‑agnostic: it does not contain any
hard‑coded file names or database identifiers.  To use the module you
must provide your own dataset of free‑text descriptions (for example
loaded from a CSV or Excel file).  Each description should be a single
string of Spanish text.

Example usage:

    from activity_clustering import load_texts, cluster_texts

    # Load your own data from a spreadsheet or CSV
    texts = load_texts('path/to/your/data.xlsx', column='DESC_FUNC')

    # Perform clustering and plot diagnostic charts
    labels, model = cluster_texts(texts, n_clusters=5, plot=True)

    # The `labels` array gives a cluster assignment for each row in
    # `texts`.  You can save this to disk or merge it back into your
    # original dataset as needed.

The module exposes a thin API for loading data, tokenising Spanish
text, computing TF‑IDF vectors, fitting a KMeans model, and
visualising the results.  It uses scikit‑learn for the machine
learning components and NLTK for tokenisation and stop word removal.

"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import stopwords  # type: ignore
from nltk.stem import SnowballStemmer  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud  # type: ignore


@dataclass
class SpanishTextPreprocessor:
    """Utility class for tokenising and normalising Spanish text.

    This class encapsulates the stop word list, punctuation and
    optional stemming.  Call the instance as a function to process
    individual strings.
    """

    stem: bool = False

    def __post_init__(self) -> None:
        self.stop_words = set(stopwords.words("spanish"))
        self.stop_words.update(["de la", "en la"])
        self.stemmer = SnowballStemmer("spanish")
        # Build a set of characters to remove (punctuation, digits, etc.)
        self.non_words = set(string.punctuation)
        self.non_words.update([
            "¿",
            "¡",
            "–",
            "‘",
            "’",
            "“",
            "”",
            "•",
            "°",
            "®",
            "(",
            ")",
            "[",
            "]",
            "@",
            "º",
        ])
        self.non_words.update(str(d) for d in range(10))

    def __call__(self, text: str) -> List[str]:
        """Tokenise and optionally stem a piece of Spanish text.

        Parameters
        ----------
        text : str
            The input string to process.

        Returns
        -------
        List[str]
            A list of processed tokens (lower‑cased, punctuation and
            digits removed, stop words filtered).  If `stem` is True
            then each token is reduced to its Snowball stem.
        """
        # Lower case and strip unwanted characters
        text = text.lower()
        text = ''.join(c for c in text if c not in self.non_words)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        if self.stem:
            return [self.stemmer.stem(t) for t in tokens]
        return tokens


def load_texts(path: str, column: str) -> List[str]:
    """Load a column of text from a CSV or Excel file.

    The function infers the file type from the extension and delegates
    to either :func:`pandas.read_excel` or :func:`pandas.read_csv`.

    Parameters
    ----------
    path : str
        Path to the spreadsheet or CSV file.
    column : str
        Name of the column containing free‑text descriptions.

    Returns
    -------
    List[str]
        A list of strings containing the activity descriptions.
    """
    ext = path.rsplit('.', 1)[-1].lower()
    if ext in {'xls', 'xlsx'}:
        df = pd.read_excel(path, usecols=[column])
    elif ext in {'csv', 'tsv'}:
        sep = ',' if ext == 'csv' else '\t'
        df = pd.read_csv(path, usecols=[column], sep=sep)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return df[column].astype(str).tolist()


def find_optimal_clusters(matrix: np.ndarray, k_range: Sequence[int]) -> Tuple[int, List[float]]:
    """Find the optimal number of clusters using the elbow and silhouette methods.

    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features)
        TF‑IDF feature matrix.
    k_range : sequence of int
        A range of candidate values for the number of clusters.

    Returns
    -------
    best_k : int
        The number of clusters that maximises the silhouette score.
    inertias : list of float
        Inertia values (sum of squared distances to cluster centres) for
        each k in `k_range`.  Useful for plotting an elbow diagram.
    """
    inertias = []
    silhouettes = []
    for k in k_range:
        model = KMeans(n_clusters=k, max_iter=200, n_init=10, random_state=42)
        labels = model.fit_predict(matrix)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(matrix, labels))
    best_k = k_range[int(np.argmax(silhouettes))]
    return best_k, inertias


def cluster_texts(
    texts: Sequence[str],
    n_clusters: Optional[int] = None,
    k_range: Sequence[int] = range(2, 10),
    plot: bool = False,
    stem: bool = False,
) -> Tuple[np.ndarray, KMeans]:
    """Vectorise a collection of texts and fit a KMeans clustering model.

    Parameters
    ----------
    texts : sequence of str
        The activity descriptions to cluster.
    n_clusters : int, optional
        Number of clusters to form.  If omitted, the optimal number
        will be estimated using the silhouette score over `k_range`.
    k_range : sequence of int, default range(2, 10)
        Candidate values of k to consider when estimating the optimal
        number of clusters.  Ignored if `n_clusters` is provided.
    plot : bool, default False
        If True, generate diagnostic plots (elbow chart, cluster
        visualisation in 2D, and word clouds for each cluster).
    stem : bool, default False
        Whether to apply stemming during tokenisation.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster assignments for each input text.
    model : KMeans
        The fitted KMeans model (with TF‑IDF vectors accessible via
        the ``vectoriser_`` attribute).
    """
    preprocessor = SpanishTextPreprocessor(stem=stem)
    vectoriser = TfidfVectorizer(tokenizer=preprocessor, stop_words=preprocessor.stop_words)
    tfidf_matrix = vectoriser.fit_transform(texts)
    # Optionally standardise features to unit variance
    scaler = StandardScaler(with_mean=False)
    matrix_std = scaler.fit_transform(tfidf_matrix)
    # Determine optimal number of clusters if not supplied
    if n_clusters is None:
        n_clusters, inertias = find_optimal_clusters(matrix_std, k_range)
        if plot:
            plt.figure()
            plt.plot(list(k_range), inertias, marker='x')
            plt.xlabel('k')
            plt.ylabel('Inertia (within‑cluster sum of squares)')
            plt.title('Elbow Method for Choosing k')
            plt.show()
    # Fit final model
    model = KMeans(n_clusters=n_clusters, max_iter=200, n_init=10, random_state=42)
    labels = model.fit_predict(matrix_std)
    # Attach vectoriser to model for later use
    model.vectoriser_ = vectoriser  # type: ignore[attr-defined]
    model.scaler_ = scaler  # type: ignore[attr-defined]
    if plot:
        _plot_clusters(matrix_std, labels)
        _plot_wordclouds(texts, labels, preprocessor)
    return labels, model


def _plot_clusters(matrix: np.ndarray, labels: np.ndarray) -> None:
    """Project TF‑IDF vectors into 2D and plot coloured clusters."""
    svd = TruncatedSVD(n_components=2, random_state=42)
    points = svd.fit_transform(matrix)
    plt.figure()
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    for k in unique_labels:
        class_points = points[labels == k]
        plt.scatter(class_points[:, 0], class_points[:, 1],
                    s=10, color=colors(k), label=f'Cluster {k}')
    plt.title('Cluster visualisation (TruncatedSVD)')
    plt.legend()
    plt.show()


def _plot_wordclouds(texts: Sequence[str], labels: np.ndarray, preprocessor: SpanishTextPreprocessor) -> None:
    """Generate and display a word cloud for each cluster."""
    n_clusters = len(np.unique(labels))
    for k in range(n_clusters):
        cluster_text = ' '.join([texts[i] for i in range(len(texts)) if labels[i] == k])
        # Remove punctuation and digits again for word cloud generation
        cluster_text = cluster_text.lower()
        cluster_text = ''.join(c for c in cluster_text if c not in preprocessor.non_words)
        tokens = [t for t in word_tokenize(cluster_text) if t not in preprocessor.stop_words]
        if preprocessor.stem:
            tokens = [preprocessor.stemmer.stem(t) for t in tokens]
        processed_text = ' '.join(tokens)
        wordcloud = WordCloud(max_words=100, background_color='white').generate(processed_text)
        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Cluster {k}')
        plt.show()


def save_cluster_results(texts: Sequence[str], labels: Sequence[int], path: str) -> None:
    """Save cluster assignments alongside the original texts to an Excel file.

    Parameters
    ----------
    texts : sequence of str
        Original activity descriptions.
    labels : sequence of int
        Cluster assignments corresponding to each description.
    path : str
        File name for the resulting Excel file.
    """
    df = pd.DataFrame({'text': texts, 'cluster': labels})
    df.to_excel(path, index=False)


if __name__ == '__main__':  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(
        description='Cluster Spanish text descriptions using TF‑IDF and KMeans.'
    )
    parser.add_argument('input_path', help='Path to a CSV or Excel file containing text data.')
    parser.add_argument('--column', default='DESC_FUNC',
                        help='Column name in the input file that contains text (default: DESC_FUNC).')
    parser.add_argument('--clusters', type=int, default=None,
                        help='Number of clusters to form.  If omitted the value will be determined automatically.')
    parser.add_argument('--plot', action='store_true', help='Generate diagnostic plots.')
    parser.add_argument('--stem', action='store_true', help='Apply stemming during tokenisation.')
    parser.add_argument('--out', default=None,
                        help='Optional path to save a spreadsheet with cluster labels.')
    args = parser.parse_args()

    # Load and cluster the texts
    descriptions = load_texts(args.input_path, args.column)
    labels, _ = cluster_texts(
        descriptions,
        n_clusters=args.clusters,
        k_range=range(2, 10),
        plot=args.plot,
        stem=args.stem,
    )
    if args.out:
        save_cluster_results(descriptions, labels, args.out)
        print(f'Cluster assignments saved to {args.out}')
    else:
        # Print a brief summary
        summary = pd.Series(labels).value_counts().sort_index()
        print('Cluster counts:')
        for k, count in summary.items():
            print(f'  Cluster {k}: {count} texts')
