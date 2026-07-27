# Adult Census Income - Classification Project

Predicting whether a person earns **<=50K** or **>50K** per year using the Adult Census Income dataset (1994 US Census data).

## Files in this project

| File | Description |
|---|---|
| `Adult_Income_Project.ipynb` | Full Jupyter notebook: EDA, feature selection, model training, comparison, tuning, and evaluation |
| `app.py` | Streamlit web app for making predictions with the trained model |
| `requirements.txt` | Python packages needed to run the notebook and the app |
| `best_model.pkl` | Final trained and tuned model (saved with joblib) |
| `scaler.pkl` | StandardScaler fitted on the training data |
| `selected_features.pkl` | List of the features used by the final model |
| `scaled_flag.pkl` | Whether the final model needs scaled input data |
| `model_accuracy.pkl` | Test accuracy of the final model |
| `Adult Census Income.csv` | The original dataset |

## Dataset

The dataset contains demographic information (age, education, occupation, marital status, hours worked per week, etc.) and the target column `income`, which is either `<=50K` or `>50K`.

## Steps followed in the notebook

1. Load the dataset and take a first look at it
2. Explore rows, columns, data types, and unique values
3. Handle missing values (the dataset uses `?` for missing data)
4. Remove duplicate rows
5. Exploratory Data Analysis (histograms, countplots, boxplots, scatter plots, pairplot, correlation heatmap)
6. One-Hot Encode the categorical columns
7. Feature Selection using SelectKBest, Mutual Information, Decision Tree importance, and Random Forest importance
8. Train-test split (80/20)
9. Feature scaling (only for models that need it)
10. Train 9 classification models: Logistic Regression, KNN, Naive Bayes, SVM, Decision Tree, Bagging, AdaBoost, Gradient Boosting, Random Forest
11. Evaluate every model (Accuracy, Precision, Recall, F1-score, ROC AUC, Confusion Matrix, ROC Curve, Precision-Recall Curve)
12. Compare all models in a single table and pick the best one automatically
13. Tune the best model using GridSearchCV and RandomizedSearchCV
14. Evaluate the final tuned model and look at its feature importance
15. Conclusion

## How to run the notebook

```bash
pip install -r requirements.txt
jupyter notebook Adult_Income_Project.ipynb
```

Run all the cells from top to bottom. Running the whole notebook will re-create `best_model.pkl`, `scaler.pkl`, `selected_features.pkl`, and `scaled_flag.pkl` used by the Streamlit app (a `joblib.dump(...)` cell for these is at the end of the feature-importance section of the notebook — add it there if you re-run and want to refresh the app's files).

## How to run the Streamlit app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app lets you:
- Enter details manually and get a prediction with probability
- Upload a CSV file of multiple people and get predictions for all of them
- See the model's test accuracy in the sidebar

## Result

The best model found was a **Gradient Boosting Classifier**, tuned with GridSearchCV, reaching about **86% accuracy** and **0.91 ROC AUC** on the test set. The most important features were marital status (married or not), capital gain, years of education, age, and hours worked per week.

## Possible Future Improvements

- Try more advanced boosting libraries like XGBoost or LightGBM
- Handle class imbalance with techniques like SMOTE
- Add more feature engineering (e.g. combining capital gain and capital loss)
- Use a larger hyperparameter search space
