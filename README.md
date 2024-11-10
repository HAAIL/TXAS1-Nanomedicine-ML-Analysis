# TXAS1-Biomarker-Analysis

This repository contains the analysis code and data for the study titled "**Thromboxane-A Synthase 1 as a Potential Dual Biomarker for Metastatic Prostate Cancer and Atherosclerosis: Insights from a Nanomedicine Protein Corona, Machine Learning, and Actual Causality Approaches**." The study leverages a combination of nanomedicine, machine learning, and causality analysis to identify Thromboxane-A Synthase 1 (TBXAS1) as a potential biomarker for metastatic prostate cancer (mPC) and atherosclerotic cardiovascular disease (ASCVD).

## Project Structure

- **`Analysis.py`**: The main analysis script, which performs the following:
  - Loads and preprocesses the dataset.
  - Constructs target variables to distinguish patient groups.
  - Trains an ElasticNet-regularized logistic regression model for classification.
  - Generates and saves model coefficients for each class.
  - Outputs a confusion matrix plot to visualize model performance.
  - Computes model accuracy.

- **`requirements.txt`**: Lists the dependencies needed to run the analysis. Install with `pip install -r requirements.txt`.

- **`LICENSE`**: Specifies the license under which this repository is made available.

- **`README.md`**: Overview and instructions for using this repository.

## Data Access

The dataset (`MM-Data_BUP_35Samples[3].xlsx`) used in this study contains proteomic information from 35 plasma samples. **To request access to the data**, please contact Morteza Mahmoudi at <mahmou22@msu.edu>. 

## Requirements

Install the dependencies listed in `requirements.txt` using:

```bash
pip install -r requirements.txt
