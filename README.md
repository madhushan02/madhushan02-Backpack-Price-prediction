# madhushan02-Backpack-Price-prediction
# Backpack Price Prediction System

## Overview
This project aims to build a **Backpack Price Prediction System** using **machine learning** techniques, specifically **Random Forest Regression** and **Deep Neural Networks (DNNs)**. The model predicts backpack prices based on various product attributes, helping businesses optimize pricing strategies.

## Features
- **Exploratory Data Analysis (EDA)**: Visualization and statistical insights into the dataset.
- **Data Preprocessing**: Handling missing values, feature scaling, and encoding categorical variables.
- **Model Training & Evaluation**:
  - **Random Forest Regression**: A robust baseline model.
  - **Deep Neural Network (DNN)**: Optimized with batch normalization, dropout, and L2 regularization.
- **Performance Metrics**:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
- **Predictions & Visualizations**:
  - True vs. Predicted Price comparison
  - Distribution analysis of predicted prices
- **Final Submission File**: CSV output for real-world applicability.

## Project Structure
```
├── dataset/                # Data files (train/test sets)
├── models/                 # Trained models and weights
├── notebooks/              # Jupyter Notebooks for EDA, training, and evaluation
├── src/                    # Scripts for data processing and model training
├── submission/             # Generated prediction outputs
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/backpack-price-prediction.git
cd backpack-price-prediction
pip install -r requirements.txt
```

## Usage
Run the Jupyter Notebook to train the model and generate predictions:
```bash
jupyter notebook notebooks/Backpack_Price_Prediction.ipynb
```
Alternatively, execute the Python script for training:
```bash
python src/train_model.py
```

## Results
- **Best RMSE Achieved**: 38.66
- **Key Observations**:
  - The DNN model outperformed the Random Forest model in capturing complex relationships.
  - Feature selection and hyperparameter tuning significantly improved predictions.

## Future Improvements
- Incorporate additional features like brand reputation and customer reviews.
- Fine-tune hyperparameters using Bayesian Optimization.
- Deploy the model as a real-time pricing API.

## Contributing
Contributions are welcome! Feel free to fork this repo and submit a pull request with improvements.

## License
This project is licensed under the MIT License.

---
**Author:** Amirthalingam Madhushan
**GitHub:** madhushan02 | https://github.com/madhushan02 

