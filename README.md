# Staff Activity NLP Clustering

This project contains a Python module and command-line tool for analysing and clustering text-based descriptions of staff activities. It was inspired by a data science initiative that leveraged natural language processing (NLP) and unsupervised learning to group duty descriptions from long-term hourly contracts. The insights helped inform policy changes that transitioned nearly 80% of 800 identified staff to full-time positions, improving employment conditions across the university.

## Overview

The module processes free-text descriptions, tokenises Spanish text, computes TF‑IDF vectors, and uses KMeans clustering to discover groups of similar activities. It also includes tools to determine an optimal number of clusters, visualise results via an elbow plot and 2D projection, and generate word clouds for each cluster. The code is data‑agnostic and does not include any hard‑coded database names or secrets.

## Repository structure

| File | Purpose |
| --- | --- |
| `activity_clustering.py` | Library and CLI for loading text data, preprocessing, vectorisation, clustering, and visualisation. |
| `requirements.txt` | List of Python dependencies needed to run the project. |

## Installation

1. Clone this repository.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

The script requires NLTK stopword lists. If you run the CLI for the first time, you may need to download the Spanish stopwords:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

You can use the module in your own Python code or run it as a standalone command-line tool.

**As a module:**

```python
from activity_clustering import load_texts, cluster_texts, save_cluster_results

# Load your own CSV or Excel file
texts = load_texts('your_data.csv', column='DESC_FUNC')

# Cluster the texts (let the algorithm choose the optimal k)
labels, model = cluster_texts(texts, plot=True)

# Optionally, save the results
save_cluster_results(texts, labels, 'clusters.xlsx')
```

**Command-line:**

```bash
python activity_clustering.py your_data.csv --column DESC_FUNC --clusters 5 --plot --out clusters.xlsx
```

This will read the specified column, cluster the descriptions into five groups, display diagnostic plots, and write the results to an Excel file.

## Ethical considerations

This project demonstrates how unsupervised learning applied to staff activity data can support evidence-based policy decisions. When implementing these techniques in real-world contexts, it is essential to protect individual privacy, secure sensitive information, and ensure that clustering results are interpreted and used responsibly. The analysis was conducted using confidential staff data from the University of Chile.
