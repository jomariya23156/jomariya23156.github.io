# Work Experiences
---
## Machine Learning Engineer at [Perceptra](https://perceptra.tech/)
<b>Jun 2022 - _Present_</b>  
Responsibilities
- Implement, adapt, and improve state-of-the-art neural networks / techniques from academic papers
- Deploy production-level deep learning models / services
- Optimize data pipeline and deployment processes
- Design / improve / maintain the full MLOps ecosystem

---
## ML Engineer Intern at [Perceptra](https://perceptra.tech/)
<b>Feb 2022 - Apr 2022 (3 months)</b>
- Performed transfer learning with EfficientNet v2 and further adapted network structure to do tuberculosis multi-label classification
- Performed hyperparameters tuning on the new tuberculosis classification model
- Migrated the MLOps platform from MLflow to ClearML for tuberculosis and CXR projects
- Wrote the new utility module for ClearML based projects
- Built and deployed the mammogram services (microservices) with multi-label ensemble models
- Optimized model inference process of the mammogram services to 3-4x faster
- Wrote well-detailed documentation and instructions about model deployment and MLOps

---
## AI Engineer Intern at [Obodroid](https://www.obodroid.com/)
<b>Jun 2021 - Jul 2021 (2 months)</b>
- Involved in the "Place Recognition" part of the Robot Navigation project
- Collected data from the real working environments
- Adapted SuperGlue model to do Visual Place Recognition task
- Implemented a lot of utility files for images/videos processing and running inference
- Built a complete pipeline from raw data input to place recognition result
- Built an end-to-end API to do Visual Place Recognition with SuperGlue and SuperPoints model using FastAPI ([GitHub](https://github.com/jomariya23156/SuperGlue-for-Visual-Place-Recognition))

---
# Projects
---
## Full-stack On-Premises MLOps system for Computer Vision

[![Static Badge](https://img.shields.io/badge/View_on_GitHub-blue?style=flat&logo=github&labelColor=grey)](https://github.com/jomariya23156/full-stack-on-prem-cv-mlops)
![GitHub Repo stars](https://img.shields.io/github/stars/jomariya23156/full-stack-on-prem-cv-mlops?style=flat&logo=github) 
[![Static Badge](https://img.shields.io/badge/View_on_YouTube-red?style=flat&logo=youtube&labelColor=grey&color=red)](https://youtu.be/NKil4uzmmQc)
![YouTube Video Views](https://img.shields.io/youtube/views/NKil4uzmmQc?style=flat&logo=youtube&link=https%3A%2F%2Fyoutu.be%2FNKil4uzmmQc)

<div style="text-align: justify">
Fully operating on-premises MLOps system tailored for Computer Vision tasks from Data versioning to Model monitoring and drift detection with the concept: <b>1 config, 1 command from Jupyter Notebook to serve Millions of users"</b>. This system equips you with everything you need, from a development workspace in Jupyter Lab/Notebook to production-level services and it only takes "1 config and 1 command" to run the whole system from building the model to deployment! I've integrated numerous best practices to ensure scalability and reliability while maintaining flexibility. While my primary use case revolves around image classification, this project structure can easily adapt to a wide range of ML/DL developments, even transitioning from on-premises to cloud!
</div>

<b>Tool stack:</b>
- Platform: [Docker](https://www.docker.com/) 
- Workspace: [Jupyter Lab](https://jupyter.org/)
- Deep Learning framework: [TensorFlow](https://www.tensorflow.org/)
- Data versioning: [DvC](https://dvc.org/)
- Data validation: [DeepChecks](https://deepchecks.com/)
- Machine Learning platform / Experiment tracking: [MLflow](https://mlflow.org/)
- Pipeline orchestrator: [Prefect](https://www.prefect.io/)
- Machine Learning service deployment: [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), [Gunicorn](https://gunicorn.org/), [Nginx](https://www.nginx.com/) (+ HTML, CSS, JS for a simple UI)
- Databases: [PostgreSQL](https://www.postgresql.org/) (SQL), [Prometheus](https://prometheus.io/) (Time-series)
- Machine Learning model monitoring & drift detection: [Evidently](https://www.evidentlyai.com/)
- Overall system monitoring & dashboard: [Grafana](https://grafana.com/)

<center><img src="images/full-stack-on-prem-cv-mlops.png"/></center>

---
## Real-time Webcam Background Replacement Web Application

[![Static Badge](https://img.shields.io/badge/View_on_GitHub-blue?style=flat&logo=github&labelColor=grey)](https://github.com/jomariya23156/realtime-webcam-bg-replace-add-filters)
![GitHub Repo stars](https://img.shields.io/github/stars/jomariya23156/realtime-webcam-bg-replace-add-filters?style=flat&logo=github) 
[![Static Badge](https://img.shields.io/badge/View_on_YouTube-red?style=flat&logo=youtube&labelColor=grey&color=red)](https://youtu.be/00FC_3qZmZc)
![YouTube Video Views](https://img.shields.io/youtube/views/00FC_3qZmZc?style=flat&logo=youtube&link=https%3A%2F%2Fyoutu.be%2F00FC_3qZmZc)

<div style="text-align: justify">
A web application with the Zoom-like feature: Real-time webcam background replacement with a Web UI + Cartoonification + Image filters built with <b>FastAPI</b> using <b>WebSocket</b> (Also, utilizes JavaScript for frontend functionalities).  
Here are the main features implemented in this app:
</div>

- Replace the webcam background with a selected prepopulated image or one uploaded by the user.
- Two available models for background segmentation: <b>Mediapipe</b> (default) and <b>'apple/deeplabv3-mobilevit-xx-small' from Hugging Face</b>.
- Cartoonify webcam stream with two options: <b>OpenCV</b> (Sequence of image processings) and <b>CartoonGAN</b> (Deep learning model).
- Apply filters to the webcam stream. Available filters include Grayscale, Saturation, Brightness, Contrast.
- Supports concurrent connections.
- The app is <b>dockerized</b>.

<center><img src="images/realtime-webcam-bg-replace-add-filters.gif"/></center>

---
## Face Recognition with Liveness Detection Login on Flask Web application

[![Static Badge](https://img.shields.io/badge/View_on_GitHub-blue?style=flat&logo=github&labelColor=grey)](https://github.com/jomariya23156/face-recognition-with-liveness-web-login)
![GitHub Repo stars](https://img.shields.io/github/stars/jomariya23156/face-recognition-with-liveness-web-login?style=flat&logo=github) 
[![Static Badge](https://img.shields.io/badge/View_on_YouTube-red?style=flat&logo=youtube&labelColor=grey&color=red)](https://youtu.be/2S-HmiPNViU)
![YouTube Video Views](https://img.shields.io/youtube/views/2S-HmiPNViU?style=flat&logo=youtube&link=https%3A%2F%2Fyoutu.be%2F2S-HmiPNViU)

<div style="text-align: justify">
A web application login page including <b>face verification</b> (1-to-1 to verify whether the person who is logging in is really that person), for security purpose, with <b>liveness detection mechanism</b> (to check whether the person detected on the camera is a <b>REAL</b> person or <b>FAKE</b> (eg. image, video, etc. of that person)) for Anti-Spoofting (Others pretending to be the person). After the login page, a webpage placeholder is also provided for future use.
</div>

- Implemented the face liveness detection method proposed in a research paper ([link](https://arxiv.org/pdf/1405.2227.pdf)).
- Collected data for real and fake images for binary classification.
- Built and trained the CNN-based liveness model from scratch with <b>TensorFlow</b>.
- Implemented the web application with <b>Flask</b> framework.
- Used <b>dlib</b> for face recognition.

<center><img src="images/face-login-short-demo.gif"/></center>

---
## EzFit: Startup's MVP

<b>Pain point:</b> People are bored of exercise and lack motivation.  
<b>Solution:</b> Gamification and Multi-user Empowered by AI.  
<b>Responsibilities in Business Aspect:</b> 
- Led and envisioned the team.
- Built a full startup plan from scratch.
- Built a Financial Model.
- Built a Marketing Plan.
- Built a pitch deck and pitched to many entrepreneurs and investors.
- Did market validation.

<b>Responsibilities in AI / App Development Aspect:</b>  
- Led the dev team.
- Analyzed and drew insights with data analytics from a market validation survey.
- Collected and preprocessed data for exercise repetition counting.
- Adapted pre-trained Pose Estimation model and trained Exercise Classification models for classifying and counting the repetition of 7 exercises.
- Achieved high accuracy across all exercises: <b>99%</b> for push-up, <b>96%</b> for jumping-jack, <b>92%</b> for squat, <b>91%</b> for leg-raise, <b>84%</b> for lunge, <b>84%</b> bicycle-crunch, <b>72%</b> for mountain-climber.
- Optimized and deployed real-time ML models on the mobile device using <b>TensorFlow</b> ecosystems. 
- Built a cross-platform mobile application using <b>React Native</b>.
- Implemented and took care of everything in the main gameplay part.

<b>Awards:</b>
- Received a grant and advanced to demo day at Startup Thailand League 2021, one of the largest startup competitions in Thailand.
- Finalist at INNO for Change 2021

<b>Main tools:</b> React Native, Expo, Firebase, TensorFlow, Tensorflow Lite, TensorFlowJS, Mediapipe

<center><img src="images/ezfit-short-demo-small.gif"/></center>

---
# Competitions
There are a lot of competitions I have attended, but here are the ones I've learned the most from and I'm allowed to open-source the work.

---
## Thailand Machine Learning for Chemistry Competition (TMLCC)

[![Static Badge](https://img.shields.io/badge/View_on_GitHub-blue?style=flat&logo=github&labelColor=grey)](https://github.com/jomariya23156/MOFs_CO2_prediction_TMLCC)
[![Static Badge](https://img.shields.io/badge/View_on_Devpost-blue?style=flat&logo=devpost&labelColor=grey)](https://devpost.com/software/project-9zuw48p5gokl?ref_content=my-projects-tab&ref_feature=my_projects)
[![Static Badge](https://img.shields.io/badge/Open_Jupyter_Notebook-blue?logo=Jupyter&labelColor=grey)](notebook_htmls/tmlcc_main_notebook.html)

<div style="text-align: justify">
TMLCC is a new type of Data Science competition in which competitors need to use Machine Learning techniques in order to build and develop mathematic models with the ability to learn and understand the relationship between structural properties of Metal-Organic frameworks (MOFs) and their Working Capacity. The developed models need to be able to predict this property of other MOFs accurately and precisely.   
Briefly given a number of features of chemistry properties and molecule files, the goal is to predict the CO2 Working Capacity (how much the MOFs can absorb CO2) in mL/g. Hence this is the regression task.
</div>
<br>

We placed in <b>6th place from over 200+ teams nationwide</b>. Here are the main techniques we applied that made us stand out:
- Create many new features from feature engineering with domain expertise in Chemistry subject.
- Fill missing values in some features with Multivariate Imputation by Chained Equation (MICE) technique and Chemistry formula.
- Select features using backward elimination along with SHAP values.
- Do hyperparameters tuning with Optuna.
- Use [DeepInsight](https://github.com/alok-ai-lab/DeepInsight) to transform tabular data into images and train a downstream CNN model.
- Use the trained CNN as a feature extractor to extract more features and append them to train along with other original features with LightGBM. This gave us the best standalone model! 
- Ensemble models with a weighted average strategy by giving higher weight to the results from more accurate models.

<center><img src="images/tmlcc_preview.png"/></center>

---
<center>© 2024 Ariya Sontrapornpol. Powered by Jekyll and the Minimal Theme.</center>