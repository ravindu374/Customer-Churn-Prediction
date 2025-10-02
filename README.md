# 📊 Customer Churn Prediction

## 📌 Project Overview
This project predicts customer churn using two approaches:  
1. **Machine Learning models** (Logistic Regression, SVM, KNN, Decision Tree, Random Forest, etc.)  
2. **Deep Learning (ANN)** with multiple hidden layers.  

The goal is to identify customers likely to leave, helping businesses take proactive actions.

---

## 🛠️ Data Preprocessing
- Removed unnecessary columns.  
- Converted mixed/invalid data types into a common type.  
- Filled **missing values** using mean (numeric) and mode (categorical).  
- Detected **class imbalance** in target variable and applied **oversampling** only on the training set (to keep test set realistic).  
- Scaled features with **StandardScaler** for both train and test sets.  
- Split dataset into **training and validation** parts.  

---

## 🤖 Machine Learning Models
Tested multiple algorithms:  
- Logistic Regression  
- Support Vector Classifier (SVC)  
- Kernel SVM  
- K-Nearest Neighbors (KNN)  
- Gaussian Naïve Bayes  
- Decision Tree Classifier  
- Random Forest  

### 🔍 Model Selection
- Many models overfit (high training accuracy, much lower test accuracy).  
- Logistic Regression had **consistent performance** on both sets.  
- Applied **k-Fold Cross Validation** → ~0.77 accuracy.  
- Tuned hyperparameters using **GridSearchCV**: chose **L2 regularization** and `C=1`.  

### 📑 Final Logistic Regression Results
| Model               | Accuracy | Precision | Recall  | F1 Score | F2 Score |
|----------------------|----------|-----------|---------|----------|----------|
| Logistic Regression  | 0.735    | 0.50      | 0.789   | 0.612    | 0.707    |

---

## 🧠 Deep Learning Model (ANN)
- Built an **ANN** with 3 hidden layers.  
- Each layer included **Dense + Dropout + Batch Normalization**.  
- Trained for **100 epochs** with **EarlyStopping** to avoid overfitting.  

### 📑 ANN Results
<img width="709" height="57" alt="image" src="https://github.com/user-attachments/assets/3f935f0c-a316-43a8-b47e-560e8b75a670" />


---

## 📊 Model Comparison
- **Logistic Regression (ML)** → Accuracy ~0.73, strong recall but lower precision, simple and interpretable.  
- **ANN (DL)** → Accuracy ~0.81, better overall performance and generalization.  

**Conclusion:** ANN outperformed Logistic Regression in predictive accuracy, but Logistic Regression remains useful as a lightweight, interpretable baseline.  

---

## 🚀 Future Work
- Explore advanced resampling (SMOTE, ADASYN).  
- Test gradient boosting methods (XGBoost, LightGBM, CatBoost).  
- Tune ANN hyperparameters (optimizers, learning rates, activation functions).  
- Deploy as a web app for real-time churn detection.  

---

## 📂 Repository Structure
<img width="535" height="170" alt="image" src="https://github.com/user-attachments/assets/d082588f-0bd6-4cc6-9bf4-f09a9ae086fa" />


---

## 👤 Author
Developed by *Ravindu Lakshan*  

