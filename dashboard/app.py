import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Weyland-Yutani Mining", layout="wide")
st.title("Weyland-Yutani Corporation")
st.markdown("### Mining Operations Analytics Dashboard")

st.info("Step 1 → Generate data in the **Generator** app → download the file  \nStep 2 → Drag & drop it here")

uploaded_file = st.file_uploader(
    "Drop **mining_data_latest.xlsx** here (generated in the Generator app)",
    type=["xlsx"]
)

if uploaded_file is None:
    st.stop()


df = pd.read_excel(uploaded_file, index_col="Date")
df.index = pd.to_datetime(df.index)
st.success(f"Data loaded successfully: {len(df)} days × {len(df.columns)} mines")


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

def make_chart(df, kind, poly_deg, anomalies):
    fig = go.Figure()
    for mine in df.columns:
        if kind == "line":
            fig.add_trace(go.Scatter(x=df.index, y=df[mine], name=mine, mode="lines"))
        elif kind == "bar":
            fig.add_trace(go.Bar(x=df.index, y=df[mine], name=mine))
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df[mine], name=mine, fill="tonexty", stackgroup="one"))

        if poly_deg > 0:
            x_num = np.arange(len(df))
            coeffs = np.polyfit(x_num, df[mine], poly_deg)
            trend = np.polyval(coeffs, x_num)
            fig.add_trace(go.Scatter(x=df.index, y=trend, name=f"{mine} trend", line=dict(dash="dot", width=2)))

    if not anomalies.empty:
        anom = anomalies.copy()
        anom["Date"] = pd.to_datetime(anom["Date"])
        fig.add_trace(go.Scatter(x=anom["Date"], y=anom["Value"], mode="markers",
                                 marker=dict(color="red", size=16, symbol="x", line=dict(width=3, color="darkred")),
                                 name="Anomaly"))

    fig.update_layout(height=700, title="Daily Mining Output", legend=dict(orientation="h"))
    return fig


st.sidebar.header("Anomaly Detection")
iqr_k   = st.sidebar.slider("IQR multiplier", 1.0, 5.0, 1.5, 0.1)
z_thr   = st.sidebar.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.1)
ma_win  = st.sidebar.slider("MA window", 3, 30, 7)
ma_pct  = st.sidebar.slider("MA deviation %", 10, 100, 30, 5)
chart_kind = st.sidebar.selectbox("Chart type", ["line", "bar", "area"])
poly_deg   = st.sidebar.selectbox("Trendline degree", [0,1,2,3,4], index=2)


stats_df = get_stats(df)
anomalies_df = find_anomalies(df, iqr_k, z_thr, ma_win, ma_pct)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Days", len(df))
c2.metric("Total output", f"{df.sum().sum():,.0f} tons")
c3.metric("Mines", len(df.columns))
c4.metric("Anomalies", len(anomalies_df))

st.markdown("### Statistics")
st.dataframe(stats_df)

st.markdown("### Daily Output")
fig = make_chart(df, chart_kind, poly_deg, anomalies_df)
st.plotly_chart(fig, use_container_width=True)

if not anomalies_df.empty:
    st.markdown("### Detected Anomalies")
    st.dataframe(anomalies_df.sort_values("Date").reset_index(drop=True))


if st.button("Generate PDF Report", type="primary"):
    with st.spinner("Creating PDF..."):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Weyland-Yutani Corporation", styles["Title"]))
        story.append(Paragraph(f"Mining Report – {datetime.now():%Y-%m-%d %H:%M}", styles["Heading2"]))
        story.append(Spacer(1, 20))

       
        data = [stats_df.columns.tolist()] + stats_df.round(1).astype(str).values.tolist()
        t = Table(data)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor("#0b5394")),
                               ('TEXTCOLOR',(0,0),(-1,0),colors.white),
                               ('GRID',(0,0),(-1,-1),1,colors.black)]))
        story.append(t)
        story.append(Spacer(1, 30))

      
        story.append(Image(BytesIO(make_chart(df, chart_kind, poly_deg, pd.DataFrame()).to_image(format="png")), 
                          width=7*inch, height=4*inch))
        if not anomalies_df.empty:
            story.append(Spacer(1, 20))
            story.append(Paragraph("Anomalies", styles["Heading2"]))
            story.append(Table([anomalies_df.columns.tolist()] + anomalies_df.astype(str).values.tolist()))
            story.append(Image(BytesIO(make_chart(df, chart_kind, 1, anomalies_df).to_image(format="png")), 
                              width=7*inch, height=4*inch))

        doc.build(story)
        pdf = buffer.getvalue()
        buffer.close()

    st.download_button("Download PDF Report", pdf, 
                       file_name=f"WeylandYutani_Report_{datetime.now():%Y%m%d_%H%M}.pdf", 
                       mime="application/pdf")
    st.success("PDF ready!")

st.balloons()