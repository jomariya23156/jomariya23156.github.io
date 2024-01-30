# Work Experiences
---
## Machine Learning Engineer at [Perceptra](https://perceptra.tech/)
**Jun 2022 - _Present_**  
Responsibilities
- Implement, adapt, and improve state-of-the-art neural networks / techniques from academic papers
- Deploy production-level deep learning models / services
- Optimize data pipeline and deployment processes
- Design / improve / maintain the full MLOps ecosystem
---
## ML Engineer Intern at [Perceptra](https://perceptra.tech/)
Feb 2022 - Apr 2022 (3 months)
- Performed transfer learning with EfficientNet v2 and further adapted network structure to do tuberculosis multi-label classification
- Performed hyperparameters tuning on the new tuberculosis classification model
- Migrated the MLOps platform from MLflow to ClearML for tuberculosis and CXR projects
- Wrote the new utility module for ClearML based projects
- Built and deployed the mammogram services (microservices) with multi-label ensemble models
- Optimized model inference process of the mammogram services to 3-4x faster
- Wrote well-detailed documentation and instructions about model deployment and MLOps
---
## AI Engineer Intern at [Obodroid](https://www.obodroid.com/)
Jun 2021 - Jul 2021 (2 months)
- Involved in the "Place Recognition" part of the Robot Navigation project
- Collected data from the real working environments
- Adapted SuperGlue model to do Visual Place Recognition task
- Implemented a lot of utility files for images/videos processing and running inference
- Built a complete pipeline from raw data input to place recognition result
- Built an end-to-end API to do Visual Place Recognition with SuperGlue and SuperPoints model using FastAPI ([GitHub](https://github.com/jomariya23156/SuperGlue-for-Visual-Place-Recognition))

# Projects
---
## Full-stack On-Premises MLOps system for Computer Vision

[![Static Badge](https://img.shields.io/badge/View_on_GitHub-blue?style=flat&logo=github&labelColor=grey)](https://github.com/jomariya23156/full-stack-on-prem-cv-mlops)
![GitHub Repo stars](https://img.shields.io/github/stars/jomariya23156/full-stack-on-prem-cv-mlops?style=flat&logo=github) 
[![Static Badge](https://img.shields.io/badge/View_on_YouTube-red?style=flat&logo=youtube&labelColor=grey&color=red)](https://youtu.be/NKil4uzmmQc)
![YouTube Video Views](https://img.shields.io/youtube/views/NKil4uzmmQc?style=flat&logo=youtube&link=https%3A%2F%2Fyoutu.be%2FNKil4uzmmQc)

Fully operating on-premises MLOps system tailored for Computer Vision tasks from Data versioning to Model monitoring and drift detection with the concept: **"1 config, 1 command from Jupyter Notebook to serve Millions of users"**. 

<center><img src="images/full-stack-on-prem-cv-mlops.png"/></center>

---
### Detect Non-negative Airline Tweets: BERT for Sentiment Analysis

[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1f32gj5IYIyFipoINiC8P3DvKat-WWLUK)

<div style="text-align: justify">The release of Google's BERT is described as the beginning of a new era in NLP. In this notebook I'll use the HuggingFace's transformers library to fine-tune pretrained BERT model for a classification task. Then I will compare BERT's performance with a baseline model, in which I use a TF-IDF vectorizer and a Naive Bayes classifier. The transformers library helps us quickly and efficiently fine-tune the state-of-the-art BERT model and yield an accuracy rate 10% higher than the baseline model.</div>

<center><img src="images/BERT-classification.png"/></center>

---
### Detect Food Trends from Facebook Posts: Co-occurence Matrix, Lift and PPMI

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/detect-food-trends-facebook.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/facebook-detect-food-trends)

<div style="text-align: justify">First I build co-occurence matrices of ingredients from Facebook posts from 2011 to 2015. Then, to identify interesting and rare ingredient combinations that occur more than by chance, I calculate Lift and PPMI metrics. Lastly, I plot time-series data of identified trends to validate my findings. Interesting food trends have emerged from this analysis.</div>
<br>
<center><img src="images/fb-food-trends.png"></center>
<br>

---
### Detect Spam Messages: TF-IDF and Naive Bayes Classifier

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/detect-spam-nlp.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/detect-spam-messages-nlp/blob/master/detect-spam-nlp.ipynb)

<div style="text-align: justify">In order to predict whether a message is spam, first I vectorized text messages into a format that machine learning algorithms can understand using Bag-of-Word and TF-IDF. Then I trained a machine learning model to learn to discriminate between normal and spam messages. Finally, with the trained model, I classified unlabel messages into normal or spam.</div>
<br>
<center><img src="images/detect-spam-nlp.png"/></center>
<br>

---
## Data Science

### Credit Risk Prediction Web App

[![Open Web App](https://img.shields.io/badge/Heroku-Open_Web_App-blue?logo=Heroku)](http://credit-risk.herokuapp.com/)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/chriskhanhtran/credit-risk-prediction/blob/master/documents/Notebook.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/credit-risk-prediction)

<div style="text-align: justify">After my team preprocessed a dataset of 10K credit applications and built machine learning models to predict credit default risk, I built an interactive user interface with Streamlit and hosted the web app on Heroku server.</div>
<br>
<center><img src="images/credit-risk-webapp.png"/></center>
<br>

---
### Kaggle Competition: Predict Ames House Price using Lasso, Ridge, XGBoost and LightGBM

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/ames-house-price.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/kaggle-house-price/blob/master/ames-house-price.ipynb)

<div style="text-align: justify">I performed comprehensive EDA to understand important variables, handled missing values, outliers, performed feature engineering, and ensembled machine learning models to predict house prices. My best model had Mean Absolute Error (MAE) of 12293.919, ranking <b>95/15502</b>, approximately <b>top 0.6%</b> in the Kaggle leaderboard.</div>
<br>
<center><img src="images/ames-house-price.jpg"/></center>
<br>

---
### Predict Breast Cancer with RF, PCA and SVM using Python

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/breast-cancer.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/predict-breast-cancer-with-rf-pca-svm/blob/master/breast-cancer.ipynb)

<div style="text-align: justify">In this project I am going to perform comprehensive EDA on the breast cancer dataset, then transform the data using Principal Components Analysis (PCA) and use Support Vector Machine (SVM) model to predict whether a patient has breast cancer.</div>
<br>
<center><img src="images/breast-cancer.png"/></center>
<br>

---
### Business Analytics Conference 2018: How is NYC's Government Using Money?

[![Open Research Poster](https://img.shields.io/badge/PDF-Open_Research_Poster-blue?logo=adobe-acrobat-reader&logoColor=white)](pdf/bac2018.pdf)

<div style="text-align: justify">In three-month research and a two-day hackathon, I led a team of four students to discover insights from 6 million records of NYC and Boston government spending data sets and won runner-up prize for the best research poster out of 18 participating colleges.</div>
<br>
<center><img src="images/bac2018.JPG"/></center>
<br>

---
<center>© 2024 Ariya Sontrapornpol. Powered by Jekyll and the Minimal Theme.</center>
