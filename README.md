# Stellar Classification: A Machine Learning Project with SDSS Data

## Project Overview

This project utilizes machine learning to classify celestial objects based on data from the Sloan Digital Sky Survey (SDSS) Data Release 17. The primary goal is to build and evaluate models capable of distinguishing between stars, galaxies, and quasars (QSOs) with high accuracy.

The workflow begins with data cleaning, feature engineering, and extensive exploratory data analysis (EDA). A baseline model using a Random Forest Classifier is established and then compared against several neural network architectures to test various optimization techniques, including class weighting and feature engineering. The project concludes with a comparative analysis of all models, demonstrating a complete end-to-end machine learning pipeline.

---

## Dataset

The dataset used is a curated subset of the **Sloan Digital Sky Survey (SDSS) Data Release 17**, sourced from Kaggle. It consists of **100,000 observations**, each with 18 initial features.

* **Target Variable:** `class` (GALAXY, STAR, QSO)
* **Key Predictive Features:**
    * **Photometric Features:** `u`, `g`, `r`, `i`, `z` (magnitudes in different light filters)
    * **Spectroscopic Features:** `redshift` (a measure of an object's recessional velocity)
    * **Positional Features:** `alpha`, `delta` (celestial coordinates)

---

## Project Workflow

1.  **Data Cleaning & Preprocessing:** Non-predictive identifier columns were dropped and placeholder error values (`-9999`) were handled.
2.  **Feature Engineering:** New "color" features (e.g., `u-g`, `g-r`) were created from the raw photometric features to provide more predictive information to the models.
3.  **Exploratory Data Analysis (EDA):** The cleaned data was visualized to understand feature distributions and the dataset's class imbalance.
4.  **Machine Learning Preparation:** The data was prepared for modeling using `scikit-learn` for label encoding, stratified train-test splitting, and feature scaling.
5.  **Comparative Modeling:** A Random Forest model was trained to set a high-performance baseline. It was then compared against multiple neural network configurations to test different optimization strategies, including deeper architectures, feature engineering, and class weighting.

---

## Key Findings & Visualizations

### Class Distribution
The dataset is imbalanced, with Quasars (QSO) being the minority class. This was accounted for during model evaluation.

*![A bar chart showing distribution of celestial objects](images/eda1.png)*

### Redshift as a Key Predictor
The `redshift` values for Stars, Galaxies, and QSOs occupy highly distinct ranges, making it the most powerful single feature for classification.

*![A box plot showing redshift values by class](images/eda2.png)*

### Feature Importance
The Random Forest model confirmed our EDA findings, ranking `redshift` as the most important feature by a large margin.

*![A bar chart showing feature importance](images/feature_importance.png)*

---

## Model Performance Comparison

The central outcome of this project is the nuanced comparison between the baseline model and the various neural network experiments.

| Metric | Random Forest (Baseline) | NN (with Feature Engineering) | NN (with Class Weights) |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | **97.95%** | 96.93% | 96.65% |
| **F1-Score (QSO)** | **0.95** | **0.95** | **0.95** |
| **Recall (QSO)** | 0.93 | 0.92 | **0.94** |

### Conclusions on Model Performance

1.  **Best Overall Model:** The **Random Forest Classifier** provided the best **overall accuracy**, proving to be an extremely effective and efficient model for this structured dataset.
2.  **Best for the Specific Problem:** For the specific challenge of finding the highest number of rare Quasars, the **Neural Network with Class Weights** was the superior model, achieving the highest **recall (94%)** for the QSO class.
3.  **The Value of Experimentation:** This highlights a critical concept in data science: the "best" model depends on the specific goal. While Random Forest was best overall, targeted techniques like class weighting were necessary to optimize for the minority class.

---

## Repository Contents

* **`01_Initial_Data_Exploration.ipynb`**: Details the data loading, cleaning, feature engineering, and exploratory data analysis process.
* **`02_Modeling.ipynb`**: Contains the ML data preparation, baseline model (Random Forest), and all neural network experiments and their evaluations.
* **`cleaned_sdss_data.pkl`**: The processed and cleaned dataset, used as the input for the modeling notebook.

## Tools & Libraries Used

* **Python 3**
* **Pandas & NumPy**
* **Scikit-learn** (RandomForestClassifier, StandardScaler, train_test_split, class_weight)
* **TensorFlow (Keras)**
* **Matplotlib & Seaborn**
