# 🌸 Iris Flower Classification - CodeAlpha Task 1

## 📌 Project Overview
This project implements a machine learning model to classify Iris flowers into three species (Setosa, Versicolor, Virginica) based on their sepal and petal measurements. The model achieves **100% accuracy** on test data!

## 🎯 Objective
- Build a classification model using the Random Forest algorithm
- Achieve high accuracy in predicting Iris species
- Understand feature importance and model evaluation

## 📂 Dataset
- **Source:** Scikit-learn's built-in Iris dataset
- **Samples:** 150 flowers
- **Features:** 4 measurements
  - Sepal Length
  - Sepal Width  
  - Petal Length
  - Petal Width
- **Target Classes:** 3 species (Setosa, Versicolor, Virginica)

## 🛠️ Technologies Used
- **Python 3.13** - Programming language
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib & Seaborn** - Data visualization
- **PyCharm** - Development environment

## 📊 Results
| Metric | Score |
|--------|-------|
| **Accuracy** | **100.00%** |
| **Model** | Random Forest Classifier |
| **Best Features** | Petal Length (44%), Petal Width (42%) |

### Classification Report

## 🔍 Key Insights
- **Petal length** and **petal width** are the most important features for classification
- The model can predict species with **100% confidence** for the test cases
- This demonstrates how machine learning can effectively classify biological species

## 🚀 How to Run This Project

### Prerequisites
```bash
pip install pandas scikit-learn matplotlib seaborn