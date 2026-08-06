# Task 1: Iris Flower Classification
# CodeAlpha Internship - Data Science Track

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("🌸 IRIS FLOWER CLASSIFICATION")
print("=" * 60)

# 1. LOAD DATA
print("\n📂 Loading Iris dataset...")
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

print(f"✅ Dataset loaded! Shape: {df.shape}")
print("\n📊 First 5 rows:")
print(df.head())

# 2. EXPLORE DATA
print("\n📈 Statistical summary:")
print(df.describe())

print("\n🔍 Checking for missing values:")
print(df.isnull().sum())

# 3. VISUALIZE DATA
print("\n📊 Creating visualizations...")
sns.pairplot(df, hue='species', diag_kind='hist')
plt.suptitle('Iris Flower Feature Relationships', y=1.02)
plt.savefig('iris_pairplot.png')  # Save the plot as an image
plt.show()

# 4. PREPARE DATA FOR ML
print("\n🎯 Preparing data for machine learning...")
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# 5. TRAIN MODEL
print("\n🤖 Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model training complete!")

# 6. EVALUATE MODEL
print("\n🔮 Making predictions on test data...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 7. FEATURE IMPORTANCE
print("\n🔑 Feature Importance (which measurements matter most):")
feature_importance = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance)

# 8. MAKE A PREDICTION
print("\n🌺 Making a prediction for a new flower...")
new_flower = [[5.8, 3.0, 4.5, 1.5]]
prediction = model.predict(new_flower)
prediction_proba = model.predict_proba(new_flower)

print(f"📏 Measurements: {new_flower[0]}")
print(f"🔮 Predicted species: {iris.target_names[prediction[0]]}")
print("\n🎲 Confidence:")
for i, species in enumerate(iris.target_names):
    print(f"   {species}: {prediction_proba[0][i] * 100:.2f}%")

print("\n" + "=" * 60)
print("✅ TASK 1 COMPLETED SUCCESSFULLY!")
print("=" * 60)