# 🏡 RealVisor — AI-Powered Real Estate Platform

**RealVisor** is an interactive web app built with **Streamlit** that helps users make smarter real estate decisions.  
It provides accurate **price predictions**, **property recommendations**, **market analytics**, and **comparative insights** using AI models trained on housing data.

---

## 🚀 Features

- 💸 **Price Prediction** – Predict apartment or house prices using advanced ML models.  
- 📊 **Market Analysis** – Explore data-driven insights with charts and trends.  
- 🏢 **Property Recommendations** – Get personalized property suggestions based on your preferences.  
- ⚖️ **Comparative Analytics** – Compare multiple properties side by side.  
- 📌 **Insights & Charts** – Discover key metrics like price distribution and top-performing localities.
- & Much More 

---

## 🧠 Tech Stack

- **Python**
- **Streamlit** (for the interactive UI)
- **Scikit-learn**
- **Pandas / NumPy**
- **Matplotlib / Seaborn / Plotly**
- **Pickle / Joblib** (for model saving)
- **Google Drive** (for hosting large `.pkl` model files)

---

## 📁 Project Structure

RealVisor/
├── Home.py
├── pages/
│ ├── 1_Price_Prediction.py
│ ├── 2_Analysis_Tool.py
│ ├── 3_Recommend_Apartments.py
│ ├── 4_Compare_Properties.py
│ └── 5_Insights.py
├── datasets/
├── requirements.txt
├── compress_pipeline.py
└── pipeline_compressed.pkl (download separately)


---

## 💾 Download the Model File

The trained ML model (`pipeline_compressed.pkl`) is large, so it’s hosted externally.

👉 **Download it here:**
https://drive.google.com/file/d/15Q3eb4q0NmR5YzBbVShSRFvHvMkdAs9p/view?usp=sharing

After downloading, place it in the main `RealVisor` folder before running the app.

---

## ⚙️ Run Locally

1. Clone or download the repo.  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

streamlit run Home.py

👨‍💻 Author

Created by: Latchan Chhetri



