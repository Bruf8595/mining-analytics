import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import os

st.set_page_config(page_title="Weyland-Yutani Data Generator", layout="wide")
st.title("Weyland-Yutani Corporation")
st.markdown("### Realistic Mining Data Generator")


with st.sidebar:
    st.header("Generation Parameters")
    
    start_date = st.date_input("Start date", datetime(2025, 1, 1))
    end_date = st.date_input("End date", datetime(2026, 12, 31))
    
    mines_input = st.text_input("Mine names (comma-separated)", "LV-426, Prometheus, Calpamos, Achilles")
    mines_list = [m.strip() for m in mines_input.split(",") if m.strip()]
    
    base_output = st.slider("Base daily output (tons)", 100, 5000, 2500, 100)
    growth_rate = st.slider("Annual growth %", -20.0, 50.0, 8.0, 0.5)
    
    corr_level = st.slider("Inter-mine correlation", 0.0, 0.95, 0.7, 0.05)
    
    st.markdown("### Day-of-week factors")
    day_factors = {
        "Monday": st.slider("Monday", 0.7, 1.3, 1.1, 0.05),
        "Tuesday": st.slider("Tuesday", 0.8, 1.3, 1.05, 0.05),
        "Wednesday": st.slider("Wednesday", 0.8, 1.3, 1.0, 0.05),
        "Thursday": st.slider("Thursday", 0.8, 1.3, 1.05, 0.05),
        "Friday": st.slider("Friday", 0.8, 1.3, 0.95, 0.05),
        "Saturday": st.slider("Saturday", 0.5, 1.2, 0.7, 0.05),
        "Sunday": st.slider("Sunday", 0.3, 1.0, 0.5, 0.05),
    }
    
    st.markdown("### Events (spikes/drops)")
    events = st.text_area("Events (one per line: YYYY-MM-DD, duration_days, multiplier)", 
                          "2025-06-15, 7, 2.5\n2025-11-20, 5, 0.3")


def generate_mining_data(start, end, mines, base, growth, corr, events_list, day_factors):
    dates = pd.date_range(start, end)
    n_days = len(dates)
    n_mines = len(mines)
    
    
    trend = np.linspace(0, growth/100 * n_days, n_days)
    noise = np.random.multivariate_normal(
        mean=np.zeros(n_mines),
        cov=np.full((n_mines, n_mines), corr) + np.eye(n_mines)*(1-corr),
        size=n_days
    )
    
    df = pd.DataFrame(noise * base * 0.15, index=dates, columns=mines)
    df = df + base * (1 + trend[:, np.newaxis])
    
    
    dow_factor = dates.day_name().map(day_factors)
    df = df.multiply(dow_factor.values[:, np.newaxis])
    
    
    for event in events_list:
        try:
            date, duration, mult = event.split(",")
            date = pd.to_datetime(date.strip())
            duration = int(duration)
            mult = float(mult)
            mask = (dates >= date) & (dates < date + timedelta(days=duration))
            df.loc[mask] *= mult
        except:
            continue
    
    df = df.round(0).clip(lower=0)
    return df


events_list = [line for line in events.split("\n") if line.strip()]


if st.button("GENERATE DATA", type="primary"):
    with st.spinner("Building better worlds..."):
        df = generate_mining_data(
            start=start_date,
            end=end_date,
            mines=mines_list,
            base=base_output,
            growth=growth_rate,
            corr=corr_level,
            events_list=events_list,
            day_factors=day_factors
        )
        
        
        os.makedirs("data", exist_ok=True)
        df.to_excel("data/mining_data_latest.xlsx", index_label="Date")
        
       
        buffer = BytesIO()
        df.to_excel(buffer, index_label="Date")
        buffer.seek(0)
        
        st.success(f"Generated {len(df):,} days × {len(df.columns)} mines")
        st.download_button(
            label="Download mining_data_latest.xlsx",
            data=buffer.getvalue(),
            file_name="mining_data_latest.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.line_chart(df.tail(30))
        st.balloons()

st.info("After generation — open the Analytics Dashboard to see anomalies and PDF report")
st.caption("© 2025 Weyland-Yutani Corp. Building Better Worlds.")