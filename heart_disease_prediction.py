import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold , cross_val_score
import numpy as np

# Step 1: Load dataset
data = pd.read_csv('heart.csv') 
# Step 2: Explore dataset
print(data.head())
print(data.info())
print(data.describe())

from sklearn.preprocessing import StandardScaler

# Step 3: Prepare features and target
X = data.drop('target', axis=1)
y = data['target']

# Step 3.1: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split dataset into training and testing sets (use scaled data)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Step 5: Initialize models
log_reg = LogisticRegression(max_iter=1000)
svc = SVC()
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()

# Step 6: Train models
log_reg.fit(X_train, y_train)
svc.fit(X_train, y_train)
rf.fit(X_train, y_train)
dt.fit(X_train, y_train)

# Step 7: Predict and evaluate
for model, name in zip([log_reg, svc, rf], ['Logistic Regression', 'SVC', 'Random Forest']):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'\n{name} Accuracy: {acc:.4f}')
    print(f'{name} F1 Score: {f1:.4f}')
    print(classification_report(y_test, y_pred))
 


# For Decision Tree
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
print('\nDecision Tree Accuracy:', round(acc_dt, 4))
print('Decision Tree F1 Score:', round(f1_dt, 4))
print(classification_report(y_test, y_pred_dt))



import joblib  

# Save models
joblib.dump(log_reg, 'logistic_model.pkl')
joblib.dump(svc, 'svc_model.pkl')
joblib.dump(rf, 'random_forest_model.pkl')
joblib.dump(dt, 'decision_tree_model.pkl')

joblib.dump(scaler, 'scaler.pkl')

print(" Models and scaler saved successfully!")


#plotting decision tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Set the size of the figure
plt.figure(figsize=(20, 10))

# Plot the decision tree
plot_tree(dt, 
          filled=True, 
          feature_names=X.columns, 
          class_names=['No Disease', 'Disease'], 
          rounded=True, 
          fontsize=10)

# Show the plot
plt.title("Decision Tree for Heart Disease Prediction")
plt.show()
dt = DecisionTreeClassifier(max_depth=3)



# Create K-Fold object (e.g., 5 folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Define your model
model = SVC(kernel='linear')

# Run cross-validation
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# Print results
print("Fold Accuracies:", scores)
print("Mean Accuracy:", np.mean(scores))
print("Std Deviation:", np.std(scores))



# Define models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVC": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

# Setup K-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Model Performance with 5-Fold Cross-Validation:\n")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    print(f"{name}:")
    print(f"  Mean Accuracy: {scores.mean():.4f}")
    print(f"  Std Deviation: {scores.std():.4f}\n")