# SMS-Spam-Detection-Using-ML
SMS spam detection system built using Python and Scikit-learn with text preprocessing and classification.

## Project Overview
This is a Machine Learning web application that detects whether an SMS message is Spam or Ham (Not Spam).
The model is built using Multinomial Naïve Bayes and achieves 96% accuracy on the SMS Spam dataset.

## Technologies Used
- Python
- Flask
- Scikit-learn
- HTML, CSS, JavaScript
- TF-IDF Vectorization
- NLP (Text Preprocessing)

## Project Files
- app.py : Flask backened application
- templates/index.html : Frontened UI
- model.pkl : Trained Multinomial Naive Bayes model
- vectorizer.pkl : TF-IDF vectorizer
- spam.csv : Dataset used
  
## Features
- Text cleaning and preprocessing
- Feature extraction using vectorization
- Spam/Ham classification
- Model evaluation

## Machine Learning Model
- Algorithm Used: Multinomial Naïve Bayes
- Accuracy: 96%

## How to Run
1. Install required libraries:
   pip install flask flask-cors scikit-learn numpy pandas

2. Run the application:
   python app.py

3. Open your browser and go to:
   http://127.0.0.1:5000/
