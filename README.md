# 💰 Tips Dashboard — Streamlit App

A data engineering project built with **Streamlit** that explores tipping patterns from the classic `tips` dataset.

---

## 📁 Project Structure

```
streamlit-tips-dashboard/
│
├── app.py                    # Main Streamlit entry point
├── requirements.txt          # Project dependencies
├── .gitignore
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Load & cache dataset
│   ├── filters.py            # Sidebar filter logic
│   └── charts.py             # Plotly chart functions
│
└── tests/
    ├── __init__.py
    ├── conftest.py           # Shared pytest fixtures
    ├── test_data_loader.py
    ├── test_filters.py
    └── test_charts.py
```

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/streamlit-tips-dashboard.git
cd streamlit-tips-dashboard
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the app
```bash
streamlit run app.py
```

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

---

## 📊 App Features

| Section | Description |
|---|---|
| **Overview** | Dataset preview, shape, column types, missing values |
| **Statistics** | Descriptive statistics on filtered data |
| **Visualizations** | 4 interactive Plotly charts |
| **Insights** | Key metrics: avg tip, best day, highest tip |

### Charts
- **Histogram** — distribution of tip amounts
- **Scatter plot** — total bill vs tip, colored by gender
- **Bar chart** — average tip by day of the week
- **Box plot** — tip distribution by meal time (Lunch vs Dinner)

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo and set `app.py` as the entry point
4. Click **Deploy** — Streamlit Cloud reads `requirements.txt` automatically

---

## 📦 Dependencies

| Package | Role |
|---|---|
| `streamlit` | Web app framework |
| `seaborn` | Dataset source |
| `pandas` | Data manipulation |
| `plotly` | Interactive charts |
| `pytest` | Unit testing |
