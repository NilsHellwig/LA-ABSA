# LA-ABSA: LLM-as-an-Annotator for Aspect-Based Sentiment Analysis

<div align="center">

**LLM-as-an-Annotator: Training Lightweight Models with LLM-Annotated Examples for Aspect Sentiment Tuple Prediction**

Accepted at **LREC 2026** (15th edition) · Palma, Mallorca (Spain)

[![Paper](https://img.shields.io/badge/Paper-LREC%202026-blue?style=for-the-badge&logo=googlescholar)](TBA)
[![Model](https://img.shields.io/badge/Gemma--3--27B-FFAB00?style=for-the-badge&logo=googlegemini)](https://huggingface.co/google/gemma-3-27b-it)

---

**Nils Constantin Hellwig · Jakob Fehle · Udo Kruschwitz · Christian Wolff**

University of Regensburg, Germany

---

</div>

> **Abstract:** Training models for Aspect-Based Sentiment Analysis (ABSA) tasks requires manually annotated data, which is expensive and time-consuming to obtain. This paper introduces LA-ABSA, a novel approach that leverages Large Language Model (LLM)-generated annotations to fine-tune lightweight models for complex ABSA tasks. We evaluate our approach on five datasets for Target Aspect Sentiment Detection (TASD) and Aspect Sentiment Quad Prediction (ASQP). Our approach outperformed previously reported augmentation strategies and achieved competitive performance with LLM-prompting in low-resource scenarios, while providing substantial energy efficiency benefits. E.g. using 50 annotated examples for in-context learning (ICL) to guide the annotation of unlabelled data, LA-ABSA achieved an F1 score of 49.85 for ASQP on the SemEval Rest16 dataset, closely matching the performance of ICL prompting with Gemma-3-27B (51.10) with significantly lower computational requirements.

---

## 🚀 Overview

This repository contains the official implementation of **LA-ABSA**, an approach for training lightweight student models using LLM-generated synthetic annotations for Aspect-Based Sentiment Analysis.

### Key Features
- **LLM-driven Annotation**: Scripts to generate synthetic training data using state-of-the-art LLMs (e.g., Gemma-3).
- **Multi-task Support**: Implementation for Target Aspect Sentiment Detection (TASD) and Aspect Sentiment Quad Prediction (ASQP).
- **Augmentation Comparison**: Tools for comparing traditional augmentation (EDA, Back-translation) with LLM-based strategies.
- **Evaluation**: Comprehensive scripts for performance and energy efficiency analysis.

## 📁 Repository Structure

- `01_create_annotated_examples/`: Scripts for LLM-based annotation and data augmentation.
- `02_trainer/`: Training pipelines for the lightweight models.
- `plots/`: Data analysis notebooks and visualization tools.
- `_out_fine_tunings/`: Output directory for model checkpoints and results.

## 🛠️ Setup & Usage

*(TBA: Add installation instructions and execution examples here)*

## 📜 Citation (TBA)

```bibtex
tba
```
