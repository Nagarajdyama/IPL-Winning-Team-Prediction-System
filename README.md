 ---

## 🏏 IPL Winning Team Prediction System  

### 🚀 **Project Overview**  
Predict the **winner of IPL matches** using **advanced machine learning models, real-time team statistics, player performance data, and venue impact analysis**.  

This **AI-powered prediction system** integrates **data science, sports analytics**, and **deep learning algorithms** to **forecast match results accurately**, providing an interactive and immersive user experience.  

By combining **historical IPL match data** and **real-time team stats**, the system enhances predictions and offers **high-accuracy insights** into match outcomes. 🔥  

---

## 🎯 **Key Features**
✅ **AI-Powered Predictions** – Utilizes **Random Forest, XGBoost, LightGBM, CatBoost**, and **Neural Networks** to predict IPL match winners.  
✅ **Advanced Data Processing** – Aggregates team-level **batting & bowling statistics** from historical IPL matches and deliveries data.  
✅ **Interactive UI** – A **Streamlit-based interface** that allows users to select teams, toss winners, and visualize predictions dynamically.  
✅ **Real-Time Match Analysis** – Uses AI to **analyze match-winning trends, venue conditions, and team performance in real time**.  
✅ **Balloon Animation & Dynamic Winner Reveal** – Predicts the winner with **balloons celebrating victory**, followed by an **animated reveal effect** that gradually **shrinks the winner's name dynamically** on the UI.  
✅ **Hyperparameter Optimization** – Implements **Optuna-based hyperparameter tuning** to optimize prediction accuracy.  

---

## 📐 **Project Architecture**
This system follows a **modular pipeline structure**:  

1️⃣ **Data Collection & Preprocessing**  
   - Loads historical IPL match data & deliveries dataset  
   - Cleans, filters, and aggregates key statistical features  
   - Encodes categorical variables (teams, players, venues)  

2️⃣ **Feature Engineering & Model Selection**  
   - Computes **batting strike rates, bowling economy**, and **team head-to-head records**  
   - Selects best ML models (**XGBoost, LightGBM, CatBoost, Random Forest**)  
   - Applies **Optuna-based hyperparameter tuning** for optimization  

3️⃣ **Interactive UI & Prediction Processing**  
   - Streamlit UI allows team selection, toss winner input, and real-time analysis  
   - **Machine learning model processes the user-input match scenario**  
   - Outputs the predicted **winning team** with **animated display effects**  

4️⃣ **Visualization & User Experience**  
   - Launches **balloons animation on winner selection**  
   - **Smooth animated text scaling effect** reveals the predicted winner dynamically  

---

## 🔗 **Tech Stack & Dependencies**
🚀 **Core Libraries:**  
- **Machine Learning:** Scikit-Learn, XGBoost, LightGBM, CatBoost  
- **Data Processing:** Pandas, NumPy  
- **Visualization & UI:** Matplotlib, Seaborn, Plotly, Streamlit  
- **Optimizations:** Optuna for hyperparameter tuning  

📦 **Install dependencies using:**
```bash
pip install -r requirements.txt
```

---

## ⚙️ **Installation Guide**

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/IPL-Prediction-System.git
cd IPL-Prediction-System
```

### **Step 2: Set Up the Virtual Environment**
Create a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate
```

### **Step 3: Install Required Libraries**
```bash
pip install -r requirements.txt
```

### **Step 4: Run the Application**
```bash
streamlit run app.py
```
The UI will launch in your browser, allowing you to input match details and see AI-generated predictions.

---

## 📝 **Usage Instructions**
1️⃣ **Upload IPL match datasets** (CSV/Excel) via the UI or place them in the project folder.  
2️⃣ **Select Team 1 & Team 2**, as well as the **toss winner** from the dropdown options.  
3️⃣ Click **"Predict Winner"** and wait for the AI analysis.  
4️⃣ **Enjoy the dynamic prediction experience with balloon animations & a stylish winning team reveal effect!**  

---

## 📊 **Model Performance & Results**
The model is **trained & tested on historical IPL data**, achieving **high prediction accuracy** using ensemble learning techniques.  
It analyzes features such as:  
- **Batting Strike Rate** 🏏  
- **Bowling Economy Rate** 🎯  
- **Venue Performance History** 🏟️  
- **Head-to-Head Team Statistics** 📊  

---

## 🚀 **Future Enhancements**
🔹 **Live Match Data Integration** – Implement API connections for **real-time IPL match tracking**.  
🔹 **Deep Learning Models** – Experiment with LSTM-based **sequence learning** for match predictions.  
🔹 **Custom Animations** – Introduce **more interactive UI effects** for an immersive prediction experience.  
🔹 **Player-Level Analysis** – Incorporate **individual player performance tracking** alongside team-level predictions.  

---

## 🏆 **Credits & Internship**
🔹 **Developed as part of my Internship at SystemTron**  
🔹 **Week 4 Internship Task: Enhancing AI-Powered IPL Predictions**  
🔹 Designed to integrate **machine learning, sports analytics**, and **engaging UI animations**  

---

## 🔗 **Contributing**
Contributions are welcome! If you'd like to enhance the UI, model, or data pipeline:  
1. **Fork the repository**  
2. **Create a feature branch**  
3. **Commit & push changes**  
4. **Submit a pull request**  

Let's collaborate and make **sports prediction AI even more powerful!** 🤖🏏  

---

## 📧 **Contact**
🔹 **Developer:** Nagaraj Dyama  
🔹 **GitHub:** https://github.com/Nagarajdyama  
🔹 **LinkedIn:** https://www.linkedin.com/in/nagaraj-dyama-9236a7244  
🔹 **Email:** nagarajdyama@gmail.com 

---


