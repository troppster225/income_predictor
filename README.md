# Income Predictor
In this project I created an income predicition app deployed using Streamlit. The initial model is created in the IncomePrediction_Working.ipynb file and was trained using the adult.csv dataset. The model predicts an employee's income category (Either less than or equal to $50K or more than $50K) using the ADA Boost Classifier algorithm. The model has an ROC AUC score of .78 and an accuracy of .87 displaying strong predictive capabilities.

## Steps for Running the Code Locally:
1. Clone the repository:
```bash
   git clone git@github.com:troppster225/income_predictor.git
```
2. Install the following dependencies if not already installed (check requirements.txt for exact versions):
* python 3.11.5
* numpy
* sklearn
* pandas
3. Open and run the IncomePrediction_Working.ipynb file if you wish to see the model making process
4. At the bottom of the IncomePrediction_Working.ipynb file run the cell with the !streamlit run app.py code
5. Input the information and see what the model predicts!

## Running remotely
If you wish to run the deployed model without downloading anything locally, go to this url: 

## Why we need a .py and .ipynb file
The .ipynb format is perfect for experimentation and iterative development. You can easily visualize things as you go and execute code step by step, allowing for better debugging. The .py file is better for automation and reproducibility. This file can be run more efficiently then the .ipynb file and can be run without manual intervention. When retraining the model in a production setting, we could use this script to automate the process and ensure consistency.