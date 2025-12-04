# Weyland-Yutani Mining Operations
Realistic synthetic data generator + full-featured analytics dashboard for daily resource extraction across Weyland-Yutani Corporation mining colonies.

Two independent Streamlit web apps:
- **Data Generator** – creates realistic time series with trends, weekly seasonality, inter-mine correlation, custom events (spikes/drops)
- **Analytics Dashboard** – anomaly detection, interactive charts, polynomial trendlines and one-click professional PDF report generation


### Live Apps
- Data Generator → https://[your-name]-generator.streamlit.app *(fill after deploy)*
- Analytics Dashboard → https://[your-name].streamlit.app *(fill after deploy)*

### Features
- Realistic data (not white noise): correlations, day-of-week patterns, overall trend, random events
- Anomaly detection using 3 methods (IQR, Z-score, moving-average deviation) with adjustable thresholds
- Charts: line / bar / stacked area + polynomial trendlines (degree 0–4)
- Anomalies highlighted with big red X markers
- One-click beautiful PDF report (statistics + charts)

### Tech Stack
- Python + Streamlit
- Pandas, NumPy, Plotly
- ReportLab (PDF generation – works everywhere)

### Local run
```bash
pip install -r requirements.txt

streamlit run generator/main_generator.py   # Generator
streamlit run dashboard/app.py              # Dashboard