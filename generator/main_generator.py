import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import os

class MiningDataGenerator:
    def __init__(self, mines, start_date, days, means, stds, correlation,
                 daily_growth, dow_factors, events, seed=None):
        self.mines = mines
        self.start_date = pd.to_datetime(start_date)
        self.days = int(days)
        self.means = means
        self.stds = stds
        self.correlation = correlation
        self.daily_growth = daily_growth / 100
        self.dow_factors = dow_factors
        self.events = events
        if seed is not None:
            np.random.seed(seed)

    def generate(self) -> pd.DataFrame:
        dates = pd.date_range(self.start_date, periods=self.days, freq='D')
        df = pd.DataFrame(index=dates)
        df.index.name = "Date"

        n = len(self.mines)
        cov = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    cov[i, j] = self.stds[i] ** 2
                else:
                    cov[i, j] = self.correlation * self.stds[i] * self.stds[j]

        base = np.random.multivariate_normal(self.means, cov, size=self.days)

        for i, mine in enumerate(self.mines):
            series = base[:, i]
            trend = (1 + self.daily_growth) ** np.arange(self.days)
            dow_effect = np.array([self.dow_factors.get(d.weekday(), 1.0) for d in dates])
            df[mine] = series * trend * dow_effect

        for ev in self.events:
            if np.random.random() > ev["prob"]:
                continue
            center = pd.to_datetime(ev["date"])
            mask = (df.index >= center) & (df.index < center + timedelta(days=ev["duration"]))
            if not mask.any():
                continue
            idx = np.flatnonzero(mask)
            center_idx = len(idx) // 2
            dist = np.abs(np.arange(len(idx)) - center_idx)
            bell = np.exp(-dist**2 / (2 * (ev["duration"]/3)**2))
            df.loc[mask] *= 1 + (ev["factor"] - 1) * bell

        return df.round(2)


st.set_page_config(page_title="Weyland-Yutani Mining Generator", layout="wide")
st.title("Weyland-Yutani Corporation")
st.markdown("### Mining Operations Data Generator")

with st.sidebar:
    st.header("Settings")

    mines_input = st.text_input("Mine names (comma-separated)", "LV-426, Origae-6, Fiorina 161")
    mines = [m.strip() for m in mines_input.split(",") if m.strip()]

    c1, c2 = st.columns(2)
    start_date = c1.date_input("Start date", datetime(2099, 11, 2))
    days = c2.number_input("Days", 10, 365, 40)

    means, stds = [], []
    for mine in mines:
        col1, col2 = st.columns(2)
        means.append(col1.number_input(f"{mine} — mean", 0.0, 1000.0, 50.0, key=f"mean_{mine}"))
        stds.append(col2.number_input(f"{mine} — std", 0.0, 200.0, 20.0, key=f"std_{mine}"))

    correlation = st.slider("Mine correlation", -0.9, 0.9, 0.2, 0.05)
    daily_growth = st.slider("Daily growth %", -10.0, 10.0, 2.0, 0.1)

    st.subheader("Day-of-week effect")
    dow_days = st.multiselect("Reduced output days", 
                              ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
                              default=["Sunday"])
    dow_factor = st.slider("Multiplier", 0.1, 1.0, 0.6, 0.05)
    dow_dict = {["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(d): dow_factor for d in dow_days}

    st.subheader("Events (max 6)")
    events = []
    factor_options = [0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0]
    for i in range(6):
        with st.expander(f"Event {i+1}"):
            col1, col2 = st.columns(2)
            date = col1.date_input("Date", datetime(2099, 11, 10), key=f"date{i}")
            duration = col2.number_input("Duration", 1, 30, 3, key=f"dur{i}")
            factor = st.select_slider("Factor", options=factor_options, value=0.7, key=f"fac{i}")
            prob = st.slider("Probability", 0.0, 1.0, 1.0, 0.05, key=f"prob{i}")
            if prob > 0:
                events.append({"date": date, "duration": duration, "factor": factor, "prob": prob})

    seed = st.number_input("Seed", 0, 999999, 42)

if st.sidebar.button("GENERATE DATA", type="primary"):
    gen = MiningDataGenerator(mines, start_date, days, means, stds,
                              correlation, daily_growth, dow_dict, events, seed)
    df = gen.generate()
    os.makedirs("../data", exist_ok=True)
    df.to_excel("../data/mining_data_latest.xlsx")
    st.session_state.df = df

if "df" in st.session_state:
    df = st.session_state.df
    fig = px.line(df.reset_index().melt(id_vars="Date", var_name="Mine", value_name="Output"),
                  x="Date", y="Output", color="Mine", title="Daily Mining Output")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([1,3])
    with col1:
        st.metric("Records", len(df))
        st.metric("Total output", f"{df.sum().sum():,.0f} tons")
        st.download_button("Download Excel", 
                           df.to_excel(index=True).encode(),
                           f"mining_{datetime.now():%Y%m%d_%H%M}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.success("Data saved → data/mining_data_latest.xlsx")
else:
    st.info("Set parameters and click GENERATE DATA")