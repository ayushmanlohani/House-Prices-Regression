# House-Prices-Regression

This project focuses on **predicting house prices** using regression models. We analyze the **House Prices (Ames) dataset** to train models that estimate the sale price of a house based on its features. We evaluated multiple models (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVR) and selected the best one. The final trained model is a **Lasso regression** pipeline (with preprocessing and feature selection) that can be used to predict house prices for new data.

## Workflow
The goal of this project is to build a predictive model that can accurately forecast house prices based on various  factors. This helps in:

  1. **Data Preprocessing**: Handle missing values, scale numerical features, and one-hot encode categorical features using pipelines.

  2. **Feature Engineering**: Create new features (**BsmtFinTotal** = **BsmtFinSF1** + **BsmtFinSF2**, and **TotalLivingArea**= **1stFlrSF** + **2ndFlrSF** + **LowQualFinSF)**, then drop redundant columns.

  3. **Feature Selection**: Use SelectFromModel with a LassoCV estimator to keep only the most important features.

  4. **Model Training & Evaluation**: Train and evaluate several models (Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVR) using a pipeline that includes preprocessing and feature selection. Compare models using R², RMSE, and MAE on a held-out validation set.

  5. **Hyperparameter Tuning**: Perform hyperparameter tuning (GridSearchCV) for the chosen model (Lasso) to find the best alpha value.

  6. **Save Model Pipeline**: Fit the final pipeline with the best hyperparameters on all training data, and save it using joblib for later use.

  7. **Visualizing Results**: Made these graphs to present the result graphically :-

          *Actual vs Predicted Scatter Plot* -> Plots the predicted values against the true house prices to visually check how close predictions are to the 45° line.

          *Residual Distribution Plot* -> Shows the distribution of residuals (errors) to verify if they are centered around zero.                                      

          *Residuals vs Predicted Plot* -> Checks for patterns in residuals — helps validate assumptions like homoscedasticity.


The project implements and compares multiple machine learning algorithms to find the best performing model for bike rental prediction.

# Dataset Description
### Source
The dataset contains house prices (Ames) data with necessary feautures, and it is obatained from kaggle.
Kaggle Link - https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques


### Dataset
The dataset is the Ames Housing Dataset (Kaggle "House Prices - Advanced Regression Techniques"). It consists of a training set (train.csv) with features and the target SalePrice, and a test set (test.csv) with features only. Each row represents a house with ~70 features including:

* **OverallQual** (overall material and finish quality)
* **GrLivArea** (above-ground living area in sq ft)
* **GarageCars** (garage capacity)
* **YearBuilt** (year built)
* …and many others covering lot, basement, exterior, and interior attributes.

### File Description
 **1_Train_HousePrice.ipynb**
   * Performs EDA (exploratory data analysis, correlation heatmaps, feature distributions)  
   * Engineers new features (`BsmtFinTotal`, `TotalLivingArea`) and drops unused columns  
   * Builds preprocessing pipelines for:  
     * Numerical data → impute missing values, scale  
     * Categorical data → impute missing values, one-hot encode  
   * Selects features using **LassoCV** (via `SelectFromModel`)  
   * Trains and compares multiple models:  
     * Linear Regression  
     * Ridge Regression  
     * Lasso Regression  
     * Decision Tree Regressor  
     * Random Forest Regressor  
     * Gradient Boosting Regressor  
     * XGBoost Regressor  
     * Support Vector Regressor (SVR)  
   * Identifies **Lasso Regression** as the best model based on cross-validation and validation scores  
   * Tunes Lasso with **GridSearchCV** for best `alpha`  
   * Saves:  
     * Final model pipeline → `best_Lasso_pipeline.joblib`  
     * Preprocessor → `preprocessor.joblib` 

## Steps Performed

1. **Data Preprocessing**
   * Handled missing values in both numerical and categorical features  
   * Scaled numerical features for uniformity  
   * Applied one-hot encoding to categorical variables using pipelines  

2. **Feature Engineering**
   * Created new features:  
     * `BsmtFinTotal` = `BsmtFinSF1` + `BsmtFinSF2`  
     * `TotalLivingArea` = `1stFlrSF` + `2ndFlrSF` + `LowQualFinSF`  
   * Dropped redundant or less useful columns  

3. **Feature Selection**
   * Used `LassoCV` with `SelectFromModel` to identify and keep important predictors  

4. **Model Training**
   * Implemented multiple ML models:  
     * Linear Regression  
     * Ridge Regression  
     * Lasso Regression  
     * Decision Tree Regressor  
     * Random Forest Regressor  
     * Gradient Boosting Regressor  
     * XGBoost Regressor  
     * Support Vector Regressor (SVR)  
   * Evaluated performance using train-test split  

5. **Hyperparameter Tuning**
   * Optimized models with `RandomizedSearchCV`  
   * Tuned Lasso model further with cross-validation to find the best `alpha`  

6. **Model Evaluation**
   * Compared models using:  
     * Root Mean Squared Error (RMSE)  
     * Mean Absolute Error (MAE)  
     * R² Score  
   * Selected **Lasso Regression** as the final model based on balanced performance  

7. **Model Saving**
   * Saved final trained pipeline and preprocessing steps with `joblib` for future predictions

 ## 📈 Evaluation Metrics
* **R² (R-squared)**  
  * Proportion of variance explained by the model (closer to 1 is better)  
  * Final Lasso Model R² ≈ **0.91**  

* **RMSE (Root Mean Squared Error)**  
  * Measures the prediction error magnitude (lower is better, penalizes large errors)  
  * Final Lasso Model RMSE ≈ **23,450**  

* **MAE (Mean Absolute Error)**  
  * Average absolute difference between predicted and actual values (lower is better)  
  * Final Lasso Model MAE ≈ **16,800**  

These metrics were used to evaluate and compare models during training and validation.  
The **Lasso Regression model** achieved the best balance across all metrics and was selected as the final model.


## Required Packages
* *pandas*  
* *numpy*  
* *matplotlib*  
* *seaborn*  
* *scikit-learn*  
* *xgboost*

## Notes

The final selected model is *Lasso Regression* with optimized hyperparameters. Other models were tested but Lasso gave strong cross-validation performance with good generalization.

All preprocessing, feature engineering, and feature selection steps are encapsulated in the *pipeline*, making the model production-ready.

The trained pipeline is saved with *joblib*, so it can be easily loaded and used to predict new data without retraining.


