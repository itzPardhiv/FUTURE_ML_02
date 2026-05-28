# 🚀 Ticket Desk — Support Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-AI%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Ticket%20Automation-6A5ACD?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Business%20Visuals-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## Project Status

This project is actively maintained and improved as part of my machine learning, NLP, and intelligent automation portfolio.

**Status:** Professional Internship Simulation ✅
**Focus Area:** NLP, Support Automation, Ticket Classification, Priority Prediction, Streamlit Dashboard
**Current Version:** 1.0.0
**Last Updated:** May 2026

---

## 📊 Overview

**Ticket Desk — Support Intelligence Platform** is a professional NLP-based support automation dashboard designed to simulate an enterprise IT helpdesk intelligence system.

The platform automates first-level support ticket triage by classifying incoming tickets, predicting priority, recommending SLA guidance, routing tickets to the appropriate support team, and generating professional first-response drafts.

This project upgrades a machine learning internship task into a complete business-facing product simulation that combines **NLP, machine learning, dashboard design, workflow automation, and stakeholder-friendly insights**.

**Built for:** IT support teams, helpdesk managers, business analysts, recruiters, and ML project reviewers
**Use case:** Ticket classification, priority prediction, SLA guidance, support routing, and response automation

---

## 🎯 Business Problem

Support teams receive large volumes of unstructured tickets every day. These tickets often need to be manually read, categorized, prioritized, assigned, and responded to.

This manual process can create several problems:

* Slow first response times
* Inconsistent ticket prioritization
* Incorrect team routing
* Higher SLA risk
* More repetitive work for support agents
* Limited visibility into ticket distribution and support workload

**Ticket Desk** addresses this problem by using machine learning to assist the first stage of ticket triage and convert raw support requests into structured, actionable support decisions.

---

## 🧠 Project Objective

The objective of this project is to build an intelligent support triage system that can:

* Understand raw support ticket descriptions
* Predict the most likely ticket category
* Estimate the ticket priority level
* Recommend SLA guidance
* Suggest the correct support team
* Generate a professional first-response draft
* Present ticket and model insights through an interactive dashboard

The goal is not just to train a model, but to demonstrate how machine learning can be converted into a practical business workflow.

---

## ✨ Core Features

### 🎫 Intelligent Ticket Classification

* Classifies support tickets into relevant issue categories.
* Uses a trained **LinearSVC** model for category prediction.
* Converts ticket text into numerical features using **TF-IDF Vectorization**.
* Handles real-world unstructured support ticket descriptions.
* Supports saved model loading for interactive dashboard inference.

---

### 🚦 Priority Prediction

* Predicts ticket urgency as **Low**, **Medium**, or **High**.
* Uses a **Logistic Regression** model for priority classification.
* Supports rule-based fallback priority generation when source priority labels are unavailable.
* Helps support teams identify tickets that may require faster attention.
* Provides a foundation for SLA-aware support workflows.

---

### 🧭 SLA Guidance

* Maps predicted priority to practical SLA recommendations.
* Helps determine how urgently a ticket should be handled.
* Gives business-friendly explanations for support decision-making.
* Makes the dashboard useful for both technical and non-technical users.

---

### 🏢 Support Team Routing

* Suggests the most suitable support team based on predicted category.
* Reduces misrouting and repeated manual review.
* Simulates real helpdesk assignment logic.
* Connects machine learning output with business process automation.

---

### 💬 First-Response Draft Generation

* Generates a professional first-response message for support agents.
* Uses predicted category and priority to create a contextual reply.
* Helps reduce repetitive communication work.
* Demonstrates how the system can assist support communication without fully replacing human agents.

---

### 📈 Interactive Streamlit Dashboard

The dashboard includes multiple business-focused pages:

* **Dashboard Overview**
* **Ticket Triage**
* **Model Performance**
* **Dataset Explorer**
* **About Project**

Each page is designed to present ML results in a way that recruiters, managers, and non-technical stakeholders can understand.

---

## 📸 Dashboard Preview

Screenshots can be added inside the `visuals/` folder.

```text
┌───────────────────────────────────────────────────────────────┐
│        Ticket Desk — Support Intelligence Platform          │
│                                                               │
│  📊 Dashboard Overview                                        │
│  ├─ Total Tickets: 47,837                                     │
│  ├─ Category Accuracy: ~85%                                   │
│  ├─ Priority Accuracy: ~96%                                   │
│  └─ TF-IDF Features: 10,000                                   │
│                                                               │
│  🎫 Ticket Triage                                             │
│  ├─ User enters support ticket description                    │
│  ├─ System predicts issue category                            │
│  ├─ System predicts ticket priority                           │
│  ├─ SLA guidance is generated                                 │
│  ├─ Support team is recommended                               │
│  └─ First-response draft is created                           │
│                                                               │
│  📈 Model Performance                                         │
│  ├─ Accuracy metrics                                          │
│  ├─ Classification reports                                    │
│  └─ Training summary                                          │
│                                                               │
│  🔍 Dataset Explorer                                          │
│  ├─ Sample ticket records                                     │
│  ├─ Category distribution                                     │
│  └─ Priority distribution                                     │
│                                                               │
│  💼 Business Intelligence                                     │
│  └─ Faster triage, better routing, SLA-aware decisions         │
└───────────────────────────────────────────────────────────────┘
```

---

## 🖼️ Screenshots

### Dashboard Overview

```text
visuals/dashboard_overview.png
```

The dashboard overview highlights the dataset size, model accuracy, TF-IDF feature count, business value, category distribution, and priority distribution.

---

### Ticket Triage

```text
visuals/ticket_triage.png
```

The ticket triage page allows users to enter a support ticket and receive the predicted category, priority level, SLA guidance, suggested support team, urgency explanation, and first-response draft.

---

### Model Performance

```text
visuals/model_performance.png
```

The model performance page summarizes category accuracy, priority accuracy, dataset details, TF-IDF feature configuration, and classification reports.

---

### Dataset Explorer

```text
visuals/dataset_explorer.png
```

The dataset explorer provides sample records, dataset shape, column details, category counts, and priority distribution insights.

---

### About Project

```text
visuals/about_project.png
```

The about page explains the business problem, machine learning workflow, technology stack, and expected business impact of the support automation system.

---

## 🏗️ Architecture

### NLP Support Automation Pipeline

```text
Support Ticket Dataset
        ↓
   Data Loading & Validation
   ├─ Load CSV dataset
   ├─ Detect text column
   ├─ Detect category column
   └─ Validate required fields
        ↓
   Text Preprocessing
   ├─ Clean raw ticket text
   ├─ Normalize ticket descriptions
   └─ Prepare labels
        ↓
   Feature Engineering
   ├─ TF-IDF Vectorization
   ├─ 10,000 configured features
   └─ Train/test split
        ↓
   Model Training
   ├─ LinearSVC for category classification
   ├─ Logistic Regression for priority prediction
   └─ Rule-based priority fallback
        ↓
   Model Evaluation
   ├─ Accuracy scores
   ├─ Classification reports
   └─ JSON training report
        ↓
   Model Persistence
   ├─ category_model.pkl
   ├─ priority_model.pkl
   ├─ tfidf_vectorizer.pkl
   └─ model_report.json
        ↓
   Streamlit Dashboard
   ├─ Dashboard overview
   ├─ Ticket triage
   ├─ SLA guidance
   ├─ Team routing
   ├─ Response generation
   └─ Dataset/model insights
```

---

## Module Structure

| Module                  | Purpose                                             |
| ----------------------- | --------------------------------------------------- |
| Data Loading            | Loads and validates the support ticket dataset      |
| Text Cleaning           | Prepares raw ticket descriptions for NLP processing |
| Feature Engineering     | Converts ticket text into TF-IDF features           |
| Category Classification | Predicts ticket category using LinearSVC            |
| Priority Prediction     | Predicts urgency level using Logistic Regression    |
| SLA Guidance            | Converts priority into response-time guidance       |
| Team Routing            | Recommends the most suitable support team           |
| Response Generator      | Creates professional first-response drafts          |
| Dashboard               | Provides interactive Streamlit pages for analysis   |
| Model Storage           | Saves and loads ML artifacts using Joblib           |

---

## 🛠️ Tech Stack

| Layer               | Technology          | Purpose                                  |
| ------------------- | ------------------- | ---------------------------------------- |
| Frontend            | Streamlit           | Interactive dashboard and user interface |
| Visualization       | Plotly              | Business charts and distribution visuals |
| NLP Features        | TF-IDF Vectorizer   | Text-to-feature transformation           |
| Category Model      | LinearSVC           | Support ticket category classification   |
| Priority Model      | Logistic Regression | Ticket urgency prediction                |
| Data Processing     | Pandas              | Dataset loading, cleaning, and analysis  |
| Numerical Computing | NumPy               | Array operations and ML support          |
| Model Storage       | Joblib              | Saving and loading trained artifacts     |
| Backend             | Python 3.11+        | Core runtime and ML pipeline             |

---

## 📁 Project Structure

```text
Ticket-Desk-Support-Intelligence-Platform/
├── 📄 README.md
├── 📋 requirements.txt
├── 📄 main.py
│
├── 📂 app/
│   └── app.py
│
├── 📂 data/
│   └── all_tickets_processed_improved_v3.csv
│
├── 📂 models/
│   ├── category_model.pkl
│   ├── priority_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_report.json
│
├── 📂 notebooks/
│   └── 01_ticket_classification_model.ipynb
│
├── 📂 scripts/
│   └── train_models.py
│
└── 📂 visuals/
    ├── dashboard_overview.png
    ├── ticket_triage.png
    ├── model_performance.png
    ├── dataset_explorer.png
    └── about_project.png
```

---

## ⚙️ Installation

### Prerequisites

* Python 3.11 or higher
* Windows, macOS, or Linux
* pip package manager
* Terminal or PowerShell access

---

## Quick Start

### 1. Clone or download the repository

```bash
cd Ticket-Desk-Support-Intelligence-Platform
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the machine learning models

```bash
python scripts/train_models.py
```

### 6. Run the Streamlit dashboard

```bash
streamlit run app/app.py
```

### 7. Open the app in your browser

```text
Local URL: http://localhost:8501
```

---

## 📊 Usage Guide

### Dashboard Pages

| Page               | Description                                                                             |
| ------------------ | --------------------------------------------------------------------------------------- |
| Dashboard Overview | Displays KPIs, dataset size, model summary, and distributions                           |
| Ticket Triage      | Accepts a ticket description and returns category, priority, SLA, team, and reply draft |
| Model Performance  | Shows accuracy metrics and classification reports                                       |
| Dataset Explorer   | Allows users to inspect records, columns, and label distributions                       |
| About Project      | Explains the business problem, ML workflow, stack, and expected impact                  |

---

## 🎫 Ticket Triage Workflow

1. Open the Streamlit dashboard.
2. Go to the **Ticket Triage** page.
3. Enter a support ticket description.
4. Click **Analyze Ticket**.
5. Review the predicted category.
6. Review the predicted priority.
7. Check the SLA guidance.
8. Check the suggested support team.
9. Read the urgency explanation.
10. Use or edit the generated first-response draft.

---

## 🧠 Machine Learning Workflow

### Category Classification

The category model uses:

* TF-IDF Vectorizer
* LinearSVC classifier
* Train/test split evaluation
* Classification report output
* Saved model artifact for dashboard inference

### Priority Prediction

The priority model uses:

* Logistic Regression
* Dataset priority labels when available
* Rule-based keyword priority generation when priority labels are unavailable
* Saved model artifact for dashboard inference

---

## 📈 Model Performance

The saved local training report shows strong internship-level NLP triage performance.

| Metric            | Result                      |
| ----------------- | --------------------------- |
| Category Accuracy | Approximately 84.5% to 85%+ |
| Priority Accuracy | Approximately 96% to 97%    |
| TF-IDF Features   | 10,000                      |
| Dataset Size      | Around 47,837 rows          |
| Category Model    | LinearSVC                   |
| Priority Model    | Logistic Regression         |

> These results make the project suitable as a realistic machine learning internship simulation rather than a production deployment claim.

---

## 💼 Business Impact

### Faster Ticket Triage

* Automatically classifies incoming support tickets.
* Reduces manual review for first-line support teams.
* Helps agents understand the issue type faster.

### Better Priority Handling

* Predicts urgency levels to support SLA-aware decisions.
* Helps identify high-priority tickets earlier.
* Improves consistency in ticket prioritization.

### Smarter Team Routing

* Recommends appropriate support teams based on the predicted issue.
* Reduces misrouted tickets and repeated manual assignment.
* Simulates real-world helpdesk assignment logic.

### Improved Support Communication

* Generates professional first-response drafts.
* Helps support agents respond faster.
* Maintains consistent communication tone.

### Recruiter-Ready ML Project

* Shows end-to-end machine learning implementation.
* Combines NLP, classification, dashboarding, and business logic.
* Demonstrates product thinking beyond basic model training.

---

## 🚀 Advanced Features

### Model Persistence

* Category model saves to `models/category_model.pkl`
* Priority model saves to `models/priority_model.pkl`
* TF-IDF vectorizer saves to `models/tfidf_vectorizer.pkl`
* Training report saves to `models/model_report.json`

### Health Checker

The project includes a beginner-friendly `main.py` launcher and health checker to verify:

* Required files
* Dataset availability
* Model artifacts
* Project folder structure

### Dataset Explorer

The dashboard includes dataset-level review features such as:

* Sample ticket records
* Dataset shape
* Column information
* Category distribution
* Priority distribution

---

## 🔮 Future Roadmap

### Phase 2: Model Explainability

* Add feature importance for ticket predictions
* Add LIME or SHAP-style explanations
* Show top words influencing category and priority predictions

### Phase 3: Workflow Automation

* Add batch ticket upload
* Add CSV export for analyzed tickets
* Add support queue simulation
* Add ticket assignment history

### Phase 4: Advanced NLP

* Experiment with transformer-based models such as BERT
* Add semantic similarity for duplicate ticket detection
* Add intent detection and sentiment analysis
* Add multilingual ticket support

### Phase 5: Enterprise Features

* Role-based views for agents, managers, and admins
* SLA breach alerts
* Team workload analytics
* Resolution-time prediction
* Cloud deployment using Streamlit Community Cloud or another hosting platform

---

## 📈 Performance & Benchmarks

| Metric            | Value                 |
| ----------------- | --------------------- |
| Dataset Size      | Around 47,837 tickets |
| Category Accuracy | ~84.5% to 85%+        |
| Priority Accuracy | ~96% to 97%           |
| TF-IDF Features   | 10,000                |
| Category Model    | LinearSVC             |
| Priority Model    | Logistic Regression   |

---

## 🐛 Troubleshooting

### Streamlit App Not Found

Issue:

```bash
Error: Invalid value: File does not exist: app/app.py
```

Solution:

Make sure you are inside the project root folder.

```bash
cd Ticket-Desk-Support-Intelligence-Platform
streamlit run app/app.py
```

---

### Models Not Found

Issue:

```text
category_model.pkl not found
```

Solution:

Train the models first.

```bash
python scripts/train_models.py
```

---

### Virtual Environment Activation Error

Issue:

```text
running scripts is disabled on this system
```

Solution for Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

---

### Missing Dependencies

Issue:

```text
ModuleNotFoundError
```

Solution:

```bash
pip install -r requirements.txt
```

---

## 📚 Learning Resources

* Streamlit Documentation: https://docs.streamlit.io/
* Scikit-Learn Documentation: https://scikit-learn.org/stable/
* Pandas Documentation: https://pandas.pydata.org/docs/
* Plotly Python Documentation: https://plotly.com/python/
* TF-IDF Vectorizer Guide: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

---

## 📧 Support & Contributing

### Questions?

Open an issue on GitHub or contact the project maintainer.

### Want to Contribute?

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/your-feature
```

3. Commit your changes.

```bash
git commit -m "Add your feature"
```

4. Push to the branch.

```bash
git push origin feature/your-feature
```

5. Open a Pull Request.

---

## 👨‍💼 About the Author

**A.J. Pardhiv**

AI & Data Science Student
Google Certified Data Analyst
Full-Stack Developer
Python & Machine Learning Enthusiast
Interested in AI, Data Science, NLP, and intelligent automation systems

### Connect With Me

GitHub: [@itzPardhiv](https://github.com/itzPardhiv)
LinkedIn: [A.J. Pardhiv](https://www.linkedin.com/in/aj-pardhiv-406a40333)

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

* Scikit-Learn — Machine learning algorithms and TF-IDF vectorization
* Streamlit — Interactive dashboard framework
* Pandas — Data processing and analysis
* Plotly — Professional dashboard visualizations
* Joblib — Model serialization and loading
* Future Interns — Internship task inspiration and project structure

---

## ⭐ Show Your Support

If you find this project useful, please consider:

* ⭐ Starring this repository
* 🔗 Sharing it with your network
* 📝 Suggesting improvements
* 💬 Giving feedback for future enhancements

---

**Last Updated:** May 2026
**Version:** 1.0.0
**Status:** Professional Internship Simulation ✅
