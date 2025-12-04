import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet


st.set_page_config(page_title="Weyland-Yutani Mining", layout="wide")
st.title("Weyland-Yutani Corporation")
st.markdown("### Mining Operations Analytics Dashboard")

st.info("Generate data in the Generator → download the file → drag it here")

uploaded_file = st.file_uploader("mining_data_latest.xlsx", type=["xlsx"])
if not uploaded_file:
    st.stop()

df = pd.read_excel(uploaded_file, index_col="Date")
df.index = pd.to_datetime(df.index)
st.success(f"Data loaded: {len(df)} days × {len(df.columns)} mines")


def get_stats(df):
    stats = []
    for col in df.columns:
        s = df[col]
        q1, q3 = s.quantile([0.25, 0.75])
        stats.append({
            "Mine": col,
            "Mean": round(s.mean(), 1),
            "Std": round(s.std(), 1),
            "Median": round(s.median(), 1),
            "Q1": round(q1, 1),
            "Q3": round(q3, 1),
            "IQR": round(q3 - q1, 1)
        })
    return pd.DataFrame(stats)


def find_anomalies(df, iqr_k=1.5, z_thr=3.0, ma_win=7, ma_pct=30):
    out = []
    ma = df.rolling(ma_win).mean()
    for mine in df.columns:
        s = df[mine]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - iqr_k * iqr, q3 + iqr_k * iqr

        for date, val in s[(s < low) | (s > high)].items():
            out.append({"Date": date.date(), "Mine": mine, "Value": round(val,1), "Method": "IQR"})
        z = np.abs((s - s.mean()) / s.std())
        for date, val in s[z > z_thr].items():
            out.append({"Date": date.date(), "Mine": mine, "Value": round(val,1), "Method": "Z-score"})
        pct = np.abs(s - ma[mine]) / ma[mine] * 100
        for date, val in s[pct > ma_pct].items():
            if not np.isnan(val):
                out.append({"Date": date.date(), "Mine": mine, "Value": round(val,1), "Method": "MA %"})
    return pd.DataFrame(out)


st.sidebar.header("Anomaly Detection")
iqr_k  = st.sidebar.slider("IQR multiplier", 1.0, 5.0, 1.5, 0.1)
z_thr  = st.sidebar.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.1)
ma_win = st.sidebar.slider("MA window", 3, 30, 7)
ma_pct = st.sidebar.slider("MA deviation %", 10, 100, 30, 5)

stats_df = get_stats(df)
anomalies_df = find_anomalies(df, iqr_k, z_thr, ma_win, ma_pct)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Days", len(df))
c2.metric("Total output", f"{df.sum().sum():,.0f} t")
c3.metric("Mines", len(df.columns))
c4.metric("Anomalies", len(anomalies_df))

st.markdown("### Statistics")
st.dataframe(stats_df)

st.markdown("### Last 30 Days (Interactive)")
st.line_chart(df.tail(30))

if not anomalies_df.empty:
    st.markdown("### Detected Anomalies")
    st.dataframe(anomalies_df.sort_values("Date"))


if st.button("Generate PDF Report", type="primary"):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=60)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Weyland-Yutani Corporation", styles["Title"]))
    story.append(Paragraph(f"Mining Operations Report<br/>{datetime.now():%Y-%m-%d %H:%M}", styles["Heading2"]))
    story.append(Spacer(1, 30))

    story.append(Paragraph("Summary Statistics", styles["Heading2"]))
    data = [stats_df.columns.tolist()] + stats_df.round(1).values.tolist()
    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#0b5394")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey),
        ('BACKGROUND',(0,1),(-1,-1),colors.HexColor("#f8f8f8"))
    ]))
    story.append(t)
    story.append(Spacer(1, 30))

    if not anomalies_df.empty:
        story.append(Paragraph(f"Anomalies Detected ({len(anomalies_df)})", styles["Heading2"]))
        anom_data = [anomalies_df.columns.tolist()] + anomalies_df.values.tolist()
        t2 = Table(anom_data)
        t2.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.red),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('GRID',(0,0),(-1,-1),0.5,colors.grey)
        ]))
        story.append(t2)

    story.append(Spacer(1, 50))
    story.append(Paragraph(" 2025 Weyland-Yutani Corp. Building Better Worlds.", styles["Normal"]))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()

    st.download_button(
        "Download the PDF report",
        data=pdf,
        file_name=f"WeylandYutani_Report_{datetime.now():%Y%m%d_%H%M}.pdf",
        mime="application/pdf"
    )
    st.success("Complete")
    st.balloons()

st.balloons()
st.caption("Complete")