# Resume Classifier using NLP and Machine Learning
This project is a machine learning pipeline built to **classify resumes into job categories** using NLP techniques and a machine learning model. 
It classifies resumes into different job categories (like Data Science, HR, Software Development, etc.)
The goal is to help recruiters automatically identify resume domains from raw text.


# Project Highlights
Cleaned and preprocessed real resume data
Visualized resume category distribution
Extracted most frequent words from resumes
Used **TF-IDF Vectorization** for feature extractio
Applied NLP techniques: stopword removal, regex cleaning, lemmatization
Trained a **Random Forest Classifier**
Achieved over **98% accuracy** on test data!

# Dataset Overview
*Rows:** 962 resumes  
- **Columns:**  
  - `Resume`: Raw resume text  
  - `Category`: Labeled job category (25 unique classes)

# Model: Random Forest
 Used `RandomForestClassifier` from `sklearn`
- Trained on 80% data, tested on 20%
- Output: Classification report with **precision**, **recall**, and **f1-score** per category
