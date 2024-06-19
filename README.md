# IMDB-MOVIE-REVIEW-Data-Science-
This repository contains the Jupyter Notebook for predicting the sentiment of IMDB movie reviews. The project focuses on natural language processing (NLP) techniques and machine learning models to classify movie reviews as positive or negative.


## Project Description

### Objective

The objective of this project is to build a model that can accurately predict the sentiment of movie reviews from the IMDB dataset. The key tasks include:

1. **Data Preprocessing**: Cleaning and preparing the text data for analysis.
2. **Feature Extraction**: Transforming the text data into numerical features using techniques like TF-IDF.
3. **Modeling**: Building and evaluating machine learning models for sentiment classification.
4. **Evaluation**: Assessing the performance of the models using appropriate metrics.

## File Description

- **IMDB MOVIE REVIEW Prediction.ipynb**: This is the main Jupyter Notebook file containing all the code, visualizations, and explanations for the project tasks.

## Requirements

To run the notebook, you will need the following Python packages:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `nltk`
- `jupyter`

The best performing model appears to be XGBoost. It has a higher testing accuracy than the Random Forest model and a higher training accuracy than the Logistic Regression model (assuming that there is a Logistic Regression model as well). Additionally, it has the best hyperparameters among the two models, which were determined through a grid search. Therefore, XGBoost is the recommended model to use for this classification task
