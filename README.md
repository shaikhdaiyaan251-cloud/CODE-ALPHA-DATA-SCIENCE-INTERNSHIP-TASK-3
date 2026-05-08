# CODE-ALPHA-DATA-SCIENCE-INTERNSHIP-TASK-3
```markdown
# 🚗 Car Price Prediction – Machine Learning Regression

> **CodeAlpha Data Science Internship – Task 3**  
> A complete machine learning pipeline to predict car prices based on features like engine size, horsepower, fuel type, transmission, and car age.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Made with Colab](https://img.shields.io/badge/Made%20with-Colab-orange)](https://colab.research.google.com/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-blue)](https://scikit-learn.org/)

---

## 📖 Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Dataset Information](#dataset-information)
- [Methodology & Workflow](#methodology--workflow)
- [Model Performance](#model-performance)
- [Feature Importance](#feature-importance)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## 📌 Project Description

This project builds a **regression model** to predict the selling price of used cars using a variety of features. The pipeline includes:

- Data cleaning and handling missing values
- Exploratory Data Analysis (EDA) with visualisations
- Feature engineering (car age, power‑per‑engine ratio)
- Preprocessing (scaling numerical features, one‑hot encoding categorical variables)
- Training and comparing multiple regression algorithms
- Hyperparameter tuning (RandomizedSearchCV, GridSearchCV)
- Evaluation using MAE, RMSE, R², residual analysis, and learning curves

The entire workflow is implemented in **Google Colab** and is easily reproducible.

---

## ✨ Features

- **End‑to‑end ML pipeline** – from raw CSV to saved model.
- **Comparison of 7+ models**: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, SVR.
- **Hyperparameter tuning** for the best model (Random Forest).
- **Feature engineering** – creates `car_age` and `power_per_engine` from existing columns (if available).
- **Visualisations**:
  - Correlation heatmap
  - Pairplot of top features
  - Boxplots of price per categorical variable
  - Scatter plots with regression lines
  - Interactive 3D plot (Plotly)
  - Actual vs predicted scatter plot
  - Residual plots
  - Learning curves
- **Model evaluation** – cross‑validation, RMSE, R², MAE, residual normality check.
- **Feature importance** (built‑in and permutation importance).
- **Save & export** – model, predictions, metrics, and plots.

---

## 🛠 Tech Stack

| Category             | Tools / Libraries                                                                 |
| -------------------- | --------------------------------------------------------------------------------- |
| **Language**         | Python 3.8+                                                                       |
| **Environment**      | Google Colab / Jupyter Notebook                                                   |
| **Data Processing**  | pandas, numpy                                                                     |
| **Visualisation**    | matplotlib, seaborn, plotly                                                       |
| **Machine Learning** | scikit‑learn (preprocessing, models, metrics, tuning)                             |
| **Model persistence** | joblib                                                                          |
| **Utilities**        | os, json, datetime, zipfile                                                       |

---

## 📁 Project Structure

```
CodeAlpha_Car_Price_Prediction/
├── Car_Price_Prediction.ipynb       # Main Colab notebook
├── README.md                        # This file
├── requirements.txt                 # Dependencies
├── best_tuned_car_price_model.pkl   # Saved best model (after tuning)
├── car_price_results/               # Output folder
│   ├── test_predictions.csv
│   ├── evaluation_metrics.json
│   ├── actual_vs_predicted.png
│   ├── residuals_histogram.png
│   ├── feature_importance.png
│   ├── learning_curves.png
│   └── ...
├── car_data.csv                     # Dataset (not uploaded to GitHub)
└── .gitignore
```

---

## 🚀 Installation & Setup

### Option 1: Google Colab (Recommended – no installation)

1. Open the notebook in Colab:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaikhdaiyaan251-cloud/CODE-ALPHA-DATA-SCIENCE-INTERNSHIP-TASK-3/blob/main/Car_Price_Prediction.ipynb)

2. Upload your car dataset CSV when prompted (or use the provided sample).

3. Run all cells (`Runtime` → `Run all`).

### Option 2: Local Jupyter Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/shaikhdaiyaan251-cloud/CODE-ALPHA-DATA-SCIENCE-INTERNSHIP-TASK-3.git
   cd CODE-ALPHA-DATA-SCIENCE-INTERNSHIP-TASK-3
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

---

## 💻 Usage

1. Open the notebook and run the first cell (environment setup).
2. The second cell loads the car dataset – you can upload the CSV or specify a path.
3. Run the remaining cells sequentially. Each major step is clearly marked (EDA, preprocessing, model training, tuning, evaluation).
4. After execution, you will find:
   - The best model saved as `best_tuned_car_price_model.pkl`
   - Predictions on the test set in `car_price_results/test_predictions.csv`
   - Performance metrics in `car_price_results/evaluation_metrics.json`
   - All plots as PNG files.

### Making predictions with the saved model

```python
import joblib
import numpy as np

# Load the trained model
model = joblib.load('best_tuned_car_price_model.pkl')

# Assume you have preprocessed new data (same scaling/encoding as training)
new_sample = np.array([[...]])  # shape: (1, n_features)
predicted_price = model.predict(new_sample)
print(f"Predicted car price: ${predicted_price[0]:,.2f}")
```

---

## 📊 Dataset Information

The dataset used in this project is a typical **car listing dataset** containing information such as:

| Feature             | Type       | Description                                    |
| ------------------- | ---------- | ---------------------------------------------- |
| `brand` / `model`   | Categorical| Car manufacturer and model name                |
| `year`              | Numerical  | Manufacturing year                             |
| `engine_size`       | Numerical  | Engine displacement (liters or cc)            |
| `horsepower`        | Numerical  | Engine power (HP)                              |
| `mileage`           | Numerical  | Kilometres driven (km)                         |
| `fuel_type`         | Categorical| Petrol, Diesel, Electric, Hybrid               |
| `transmission`      | Categorical| Manual, Automatic                              |
| `seller_type`       | Categorical| Individual, Dealer                             |
| `owner`             | Categorical| First owner, Second owner, etc.                |
| `price`             | Numerical  | Target variable (selling price)                |

> *The exact column names may vary. The notebook automatically detects the price column and common categorical features.*

**Preprocessing steps**:
- Missing values imputed (median for numerical, mode for categorical).
- Created `car_age` = current year (2025) – `year`.
- Created `power_per_engine` = horsepower / engine_size (if both exist).
- One‑hot encoding for categorical variables.
- Standard scaling for numerical features.

---

## 📐 Methodology & Workflow

1. **Data Loading & EDA**  
   - Visualised distribution of price (target).
   - Correlation heatmap, pairplots, boxplots.
   - Identified key features influencing price.

2. **Preprocessing & Feature Engineering**  
   - Handled missing values.
   - Created `car_age` and `power_per_engine`.
   - Separated features (X) and target (y).
   - Split into training (80%) and test (20%) sets.
   - Built `ColumnTransformer` with pipelines for scaling and encoding.

3. **Model Training & Comparison**  
   - Trained 7 regression models.
   - Compared performance using RMSE, R², and cross‑validation.

4. **Hyperparameter Tuning**  
   - Used `RandomizedSearchCV` (30 iterations, 3‑fold CV) and `GridSearchCV` on a reduced grid.
   - Best model: Random Forest with optimised hyperparameters.

5. **Evaluation**  
   - Test set metrics (MAE, RMSE, R²).
   - Residual analysis (histogram, Q‑Q plot).
   - Learning curves to check for overfitting.
   - Feature importance (built‑in and permutation).

6. **Model Saving & Export**  
   - Saved the best tuned model using `joblib`.
   - Exported predictions, metrics, and visualisations.

---

## 📈 Model Performance

After hyperparameter tuning, the **Random Forest Regressor** achieved the following results:

| Metric                    | Value (on test set) |
| ------------------------- | ------------------- |
| **Mean Absolute Error**   | TODO (e.g., $1,250) |
| **Root Mean Squared Error** | TODO (e.g., $2,100) |
| **R² Score**               | TODO (e.g., 0.92)   |

*Note: Fill in the actual numbers after running the notebook.*

### Actual vs Predicted (Example)

![Actual vs Predicted](car_price_results/actual_vs_predicted.png)

### Residual Distribution

![Residuals](car_price_results/residuals_histogram.png)

### Learning Curves

![Learning Curves](car_price_results/learning_curves.png)

---

## 🔍 Feature Importance

The tuned Random Forest model identified the following features as most important (example):

![Feature Importance](car_price_results/feature_importance.png)

> *Typical top features: `car_age`, `engine_size`, `horsepower`, `brand`, `fuel_type`. Exact order may vary.*

---

## 🔮 Future Improvements

- **Incorporate more features** – such as accident history, number of owners, or location‑based pricing.
- **Use advanced models** – XGBoost, LightGBM, or CatBoost.
- **Deploy as a web app** – using Streamlit or FastAPI.
- **Add SHAP explanations** for model interpretability.
- **Implement automated outlier handling** (e.g., winsorization).
- **Experiment with target transformation** (log or Box‑Cox) to handle price skewness.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas to improve the model or the pipeline, please:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add some amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 🙏 Acknowledgements

- **CodeAlpha** – for the internship opportunity and project guidelines.
- **Scikit‑learn** – for providing robust machine learning tools.
- The open‑source community for libraries that made this analysis possible.

---

## 📬 Contact

**DAIYAAN SHAIKH** – [LinkedIn](https://www.linkedin.com/in/daiyaan-shaikh-159909377?utm_source=share_via&utm_content=profile&utm_medium=member_android) – [GitHub](https://github.com/shaikhdaiyaan251-cloud)  
Project Link: [[https://github.com/shaikhdaiyaan251-cloud/CODE-ALPHA-DATA-SCIENCE-INTERNSHIP-TASK-3](https://github.com/shaikhdaiyaan251-cloud/CODE-ALPHA-DATA-SCIENCE-INTERNSHIP-TASK-3)]


---

⭐ **If you found this project useful, please give it a star!** ⭐
```
