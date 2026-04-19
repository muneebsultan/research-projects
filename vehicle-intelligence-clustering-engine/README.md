# Unsupervised Pattern Recognition: Car Clustering & Summarisation

[![BERT](https://img.shields.io/badge/model-BERT-yellow.svg)]()
[![HDBSCAN](https://img.shields.io/badge/clustering-HDBSCAN-blueviolet.svg)]()
[![Scale](https://img.shields.io/badge/scale-100K%2B%20articles%2Fday-success.svg)]()

## Overview

A high-throughput pipeline for identifying **semantic patterns in intelligence car engine streams** using transformer-based embeddings and density-based clustering. The system deduplicates near-identical stories, groups topically related articles, and surfaces emerging narratives in real time.

---

## Research Questions

1. How **robust are transformer-based embeddings** (BERT) to noise in financial text data?
2. Which clustering strategies best balance **accuracy with computational efficiency** at scale?
3. Can unsupervised methods reliably identify meaningful semantic patterns **without labelled data**?

---

## Methods

| Component | Choice |
|-----------|--------|
| Sentence embeddings | **BERT** (HuggingFace Transformers) |
| Clustering algorithm | **HDBSCAN** (density-based, handles varying cluster sizes) |
| Evaluation | Silhouette Score, Davies–Bouldin Index |
| Operating scale | **100,000+ articles per day** |

---

## Results

| Metric | Value |
|--------|-------|
| Article deduplication rate | **30%** reduction in duplicate / near-duplicate articles |
| Clustering quality (Silhouette Score, validation) | **0.68** |
| Throughput | **500 articles / sec** |
| Per-article embedding latency | **< 100 ms** |

---

## Key Insights

- **BERT embeddings capture semantic similarity well**, even for domain-specific financial language and entity-heavy text.
- **HDBSCAN outperforms K-Means** because financial cars data naturally produces clusters of varying density — many small breaking-cars clusters sit alongside a few large, long-running narratives.
- **Core trade-off:** Balancing false positives (missing real clusters) against false negatives (over-clustering). Density thresholds were tuned on a labelled validation slice.

---

## Related Work

- Sentence-BERT (SBERT) for semantic similarity
- Density-based clustering methods (DBSCAN, OPTICS, HDBSCAN)
- Unsupervised representation learning in NLP

---

## Technologies

`BERT` · `HDBSCAN` · `scikit-learn` · `Python` · `pandas` · `Production ML`

---

## Future Directions

- **Online / incremental clustering** for streaming data (avoid full re-fit on every batch)
- Comparison with newer embedding models (OpenAI embeddings, FinBERT, domain-specific transformers)
- Qualitative study on what makes a cluster *meaningful to a domain expert* — bridging unsupervised metrics and human judgment

---

## Repository Structure

```
vehicle-intelligence-clustering-engine/
├── README.md
├── requirements.txt
├── notebooks/      # embedding experiments, cluster quality analysis
├── src/            # ingestion, embedding, clustering modules
```

---

## References & Citations

- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2019)
- Reimers & Gurevych, *Sentence-BERT* (2019)
- McInnes et al., *HDBSCAN: Hierarchical Density-Based Clustering* (2017)
