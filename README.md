# Real Estate Price Predictor 🏠

This is a beginner-friendly data science project where I built a machine learning model to predict housing prices. I used Python to clean the data, explore trends, and train a regression model that gives a price estimate based on home features.

## What it does
- Loads a real estate dataset (from Kaggle)
- Cleans up missing or inconsistent data
- Visualizes key patterns (like price vs. living area)
- Trains a model to predict house prices using XGBoost
- Shows which features matter most when estimating value

## Why I made this
I'm a finance major learning data science, and I wanted a hands-on project that connects both. Real estate is something I’m interested in, and this helped me understand how to apply Python and machine learning to real-world financial data.

## Tools & Libraries
- Python (pandas, numpy)
- Seaborn & Matplotlib (data viz)
- scikit-learn & XGBoost (machine learning)
- VS Code
- Git & GitHub

## How to run it
1. Download the dataset from Kaggle (House Prices: Advanced Regression Techniques)
2. Place the CSV in the `data/` folder
3. Run these scripts from your terminal:

```bash
python clean_and_visualize.py
python train_model.py
