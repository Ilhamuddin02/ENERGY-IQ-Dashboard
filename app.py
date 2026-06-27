"""
EnergyIQ — Smart Building Energy Dashboard
============================================
Dashboard interaktif untuk eksplorasi data & prediksi konsumsi energi
gedung pintar, menggunakan model Random Forest Regressor.

Jalankan lokal   : streamlit run app.py
Deploy           : Streamlit Community Cloud (lihat README.md)
"""

import json
import math
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EnergyIQ — Smart Building Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

IDR_PER_KWH = 1699.53
BASELINE_KWH = 320.0

COLORS = {
    "primary": "#6366f1",
    "secondary": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "bg": "#0f172a",
    "card": "#1e293b",
    "border": "#334155",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
}

TYPE_COLORS = {
    "Commercial": "#6366f1",
    "Industrial": "#ef4444",
    "Educational": "#10b981",
    "Residential": "#f59e0b",
}

OCC_COLORS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}


# ──────────────────────────────────────────────────────────────────────────
# CSS KUSTOM
# ──────────────────────────────────────────────────────────────────────────
def load_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"]  { font-family: 'Manrope', sans-serif; }

        .stApp { background-color: #0f172a; }

        /* Hero header */
        .eiq-hero {
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 55%, #052e2b 100%);
            border: 1px solid #334155;
            border-radius: 18px;
            padding: 2rem 2.2rem;
            margin-bottom: 1.6rem;
        }
        .eiq-hero h1 {
            font-size: 2.1rem;
            font-weight: 800;
            color: #f8fafc;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .eiq-hero p {
            color: #94a3b8;
            font-size: 0.98rem;
            margin-top: 0.4rem;
            max-width: 700px;
        }
        .eiq-badge {
            display: inline-block;
            background: rgba(99,102,241,0.15);
            color: #a5b4fc;
            border: 1px solid rgba(99,102,241,0.35);
            padding: 0.18rem 0.7rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 1rem 1.2rem 0.8rem 1.2rem;
        }
        div[data-testid="stMetric"] label {
            color: #94a3b8 !important;
            font-weight: 600;
            font-size: 0.78rem !important;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        div[data-testid="stMetricValue"] {
            color: #f8fafc !important;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Section title */
        .eiq-section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #f1f5f9;
            margin: 0.2rem 0 0.9rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .eiq-section-title .tag {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: #6366f1;
            background: rgba(99,102,241,0.12);
            border-radius: 6px;
            padding: 0.1rem 0.45rem;
        }

        /* Prediction result card */
        .eiq-result-card {
            border-radius: 16px;
            padding: 1.6rem 1.8rem;
            border: 1px solid #334155;
            background: linear-gradient(135deg, #1e293b 0%, #172033 100%);
        }
        .eiq-result-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.6rem;
            font-weight: 700;
            color: #f8fafc;
            line-height: 1.1;
        }
        .eiq-result-label {
            color: #94a3b8;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.3rem;
        }
        .eiq-pill {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .pill-high   { background: rgba(239,68,68,0.16);  color: #fca5a5; border: 1px solid rgba(239,68,68,0.35); }
        .pill-normal { background: rgba(245,158,11,0.16); color: #fcd34d; border: 1px solid rgba(245,158,11,0.35); }
        .pill-low    { background: rgba(16,185,129,0.16); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.35); }

        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}

        section[data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #334155;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────
# LOAD DATA & ARTEFAK MODEL (cached)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("processed_data.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


@st.cache_resource
def load_artifacts():
    model = joblib.load("rf_model.pkl")
    le_building = joblib.load("le_building.pkl")
    le_occupancy = joblib.load("le_occupancy.pkl")
    le_type = joblib.load("le_type.pkl")
    with open("model_metadata.json") as f:
        meta = json.load(f)
    return model, le_building, le_occupancy, le_type, meta


try:
    df = load_data()
    rf_model, le_building, le_occupancy, le_type, meta = load_artifacts()
    DATA_OK = True
except Exception as e:
    DATA_OK = False
    LOAD_ERROR = str(e)


def predict_energy(hour, day_of_week, temperature, humidity,
                    occupancy_label, building_type_label, building_id_label,
                    month=1):
    """Replika persis fungsi prediksi dari notebook pelatihan."""
    is_weekend = 1 if day_of_week >= 5 else 0
    is_work_hour = 1 if (8 <= hour <= 18 and not is_weekend) else 0
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    day_sin = math.sin(2 * math.pi * day_of_week / 7)
    day_cos = math.cos(2 * math.pi * day_of_week / 7)
    occ_enc = int(le_occupancy.transform([occupancy_label])[0])
    type_enc = int(le_type.transform([building_type_label])[0])
    bid_enc = int(le_building.transform([building_id_label])[0])

    X_pred = pd.DataFrame([{
        "Hour": hour, "DayOfWeek": day_of_week, "IsWeekend": is_weekend,
        "IsWorkingHour": is_work_hour, "Month": month,
        "Hour_sin": hour_sin, "Hour_cos": hour_cos,
        "Day_sin": day_sin, "Day_cos": day_cos,
        "Temperature (°C)": temperature, "Humidity (%)": humidity,
        "Occupancy_enc": occ_enc, "BuildingType_enc": type_enc,
        "Building_ID_enc": bid_enc,
    }])
    pred = float(rf_model.predict(X_pred)[0])
    return max(50.0, pred)


# ──────────────────────────────────────────────────────────────────────────
# PLOTLY THEME HELPER
# ──────────────────────────────────────────────────────────────────────────
def style_fig(fig, height=400):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color="#e2e8f0", size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
        hoverlabel=dict(bgcolor="#1e293b", font_size=12, font_family="Manrope"),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGASI
# ──────────────────────────────────────────────────────────────────────────
def sidebar_nav():
    st.sidebar.markdown(
        """
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.3rem;">
            <span style="font-size:1.7rem;">⚡</span>
            <span style="font-size:1.25rem;font-weight:800;color:#f8fafc;">EnergyIQ</span>
        </div>
        <p style="color:#64748b;font-size:0.78rem;margin-top:-0.3rem;margin-bottom:1.2rem;">
            Smart Building Energy Intelligence
        </p>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Navigasi",
        [
            "🏠  Ringkasan",
            "📊  Eksplorasi Data",
            "🔮  Prediksi Energi",
            "🧠  Performa Model",
        ],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    return page


# ──────────────────────────────────────────────────────────────────────────
# HALAMAN 1 — RINGKASAN
# ──────────────────────────────────────────────────────────────────────────
def page_overview(df):
    st.markdown(
        """
        <div class="eiq-hero">
            <div class="eiq-badge">⚡ Random Forest · R² 0.77</div>
            <h1>EnergyIQ — Smart Building Energy Dashboard</h1>
            <p>Pantau, eksplorasi, dan prediksi konsumsi energi 20 gedung pintar
            berdasarkan waktu, cuaca, dan tingkat hunian — didukung model
            Machine Learning yang dilatih dari data historis per jam.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_energy = df["Energy_Usage (kWh)"].sum()
    avg_energy = df["Energy_Usage (kWh)"].mean()
    n_buildings = df["Building_ID"].nunique()
    date_min, date_max = df["Timestamp"].min(), df["Timestamp"].max()
    n_days = (date_max - date_min).days + 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Konsumsi", f"{total_energy/1000:,.1f} MWh", help="Total seluruh gedung & periode data")
    c2.metric("Rata-rata per Jam", f"{avg_energy:,.1f} kWh")
    c3.metric("Jumlah Gedung", f"{n_buildings}", "B001–B020")
    c4.metric("Periode Data", f"{n_days} hari", f"{date_min.date()} → {date_max.date()}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown('<div class="eiq-section-title">📈 Tren Konsumsi Energi Harian (Rata-rata Seluruh Gedung)</div>', unsafe_allow_html=True)
        daily = df.groupby(df["Timestamp"].dt.date)["Energy_Usage (kWh)"].mean().reset_index()
        daily.columns = ["Tanggal", "Rata-rata kWh"]
        fig = px.area(daily, x="Tanggal", y="Rata-rata kWh")
        fig.update_traces(line_color=COLORS["primary"], fillcolor="rgba(99,102,241,0.18)")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    with col_right:
        st.markdown('<div class="eiq-section-title">🏢 Distribusi Tipe Gedung</div>', unsafe_allow_html=True)
        type_count = df.groupby("Building_Type")["Building_ID"].nunique().reset_index()
        type_count.columns = ["Tipe", "Jumlah Gedung"]
        fig = px.pie(
            type_count, names="Tipe", values="Jumlah Gedung", hole=0.55,
            color="Tipe", color_discrete_map=TYPE_COLORS,
        )
        fig.update_traces(textinfo="label+value", textfont_size=12)
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="eiq-section-title">🏆 Peringkat Konsumsi Energi per Gedung</div>', unsafe_allow_html=True)
    rank = (
        df.groupby(["Building_ID", "Building_Type"])["Energy_Usage (kWh)"]
        .mean()
        .reset_index()
        .sort_values("Energy_Usage (kWh)", ascending=False)
    )
    fig = px.bar(
        rank, x="Building_ID", y="Energy_Usage (kWh)", color="Building_Type",
        color_discrete_map=TYPE_COLORS,
        labels={"Energy_Usage (kWh)": "Rata-rata kWh", "Building_ID": "Gedung"},
    )
    fig.update_layout(legend_title_text="Tipe Gedung")
    st.plotly_chart(style_fig(fig, 380), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────
# HALAMAN 2 — EKSPLORASI DATA
# ──────────────────────────────────────────────────────────────────────────
def page_explore(df):
    st.markdown('<div class="eiq-section-title">📊 <span>Eksplorasi Data</span> <span class="tag">7,200 baris</span></div>', unsafe_allow_html=True)

    with st.expander("🔎 Filter Data", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sel_buildings = st.multiselect(
                "Gedung", sorted(df["Building_ID"].unique()),
                default=sorted(df["Building_ID"].unique())[:5],
            )
        with fc2:
            sel_types = st.multiselect(
                "Tipe Gedung", sorted(df["Building_Type"].unique()),
                default=sorted(df["Building_Type"].unique()),
            )
        with fc3:
            date_range = st.date_input(
                "Rentang Tanggal",
                value=(df["Timestamp"].min().date(), df["Timestamp"].max().date()),
                min_value=df["Timestamp"].min().date(),
                max_value=df["Timestamp"].max().date(),
            )

    fdf = df[df["Building_Type"].isin(sel_types)]
    if sel_buildings:
        fdf = fdf[fdf["Building_ID"].isin(sel_buildings)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        fdf = fdf[(fdf["Timestamp"].dt.date >= start) & (fdf["Timestamp"].dt.date <= end)]

    if fdf.empty:
        st.warning("Tidak ada data untuk filter yang dipilih. Coba ubah filter di atas.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baris Terfilter", f"{len(fdf):,}")
    m2.metric("Rata-rata kWh", f"{fdf['Energy_Usage (kWh)'].mean():,.1f}")
    m3.metric("Suhu Rata-rata", f"{fdf['Temperature (°C)'].mean():,.1f} °C")
    m4.metric("Kelembapan Rata-rata", f"{fdf['Humidity (%)'].mean():,.1f} %")

    st.markdown("<br>", unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["📈 Time-Series", "⏰ Pola Waktu", "🌡️ Cuaca vs Energi", "🧩 Korelasi"])

    with t1:
        fig = px.line(
            fdf, x="Timestamp", y="Energy_Usage (kWh)", color="Building_ID",
            labels={"Energy_Usage (kWh)": "kWh"},
        )
        fig.update_traces(line_width=1.1, opacity=0.85)
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    with t2:
        pc1, pc2 = st.columns(2)
        with pc1:
            hourly = fdf.groupby("Hour")["Energy_Usage (kWh)"].mean().reset_index()
            colors = [COLORS["warning"] if 8 <= h <= 18 else COLORS["primary"] for h in hourly["Hour"]]
            fig = go.Figure(go.Bar(x=hourly["Hour"], y=hourly["Energy_Usage (kWh)"], marker_color=colors))
            fig.update_layout(title="Rata-rata Konsumsi per Jam", xaxis_title="Jam", yaxis_title="kWh")
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
        with pc2:
            dow_names = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
            daily = fdf.groupby("DayOfWeek")["Energy_Usage (kWh)"].mean().reset_index()
            daily["Hari"] = daily["DayOfWeek"].apply(lambda x: dow_names[x])
            colors_d = [COLORS["danger"] if d >= 5 else COLORS["secondary"] for d in daily["DayOfWeek"]]
            fig = go.Figure(go.Bar(x=daily["Hari"], y=daily["Energy_Usage (kWh)"], marker_color=colors_d))
            fig.update_layout(title="Rata-rata Konsumsi per Hari", xaxis_title="", yaxis_title="kWh")
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)

        bc1, bc2 = st.columns(2)
        with bc1:
            fig = px.box(
                fdf, x="Building_Type", y="Energy_Usage (kWh)", color="Building_Type",
                color_discrete_map=TYPE_COLORS,
                category_orders={"Building_Type": ["Commercial", "Industrial", "Educational", "Residential"]},
            )
            fig.update_layout(title="Konsumsi per Tipe Gedung", showlegend=False)
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
        with bc2:
            fig = px.box(
                fdf, x="Occupancy_Level", y="Energy_Usage (kWh)", color="Occupancy_Level",
                color_discrete_map=OCC_COLORS,
                category_orders={"Occupancy_Level": ["Low", "Medium", "High"]},
            )
            fig.update_layout(title="Konsumsi per Tingkat Hunian", showlegend=False)
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)

    with t3:
        sc1, sc2 = st.columns(2)
        with sc1:
            fig = px.scatter(
                fdf.sample(min(2000, len(fdf)), random_state=42),
                x="Temperature (°C)", y="Energy_Usage (kWh)", color="Building_Type",
                color_discrete_map=TYPE_COLORS, opacity=0.55,
            )
            fig.update_layout(title="Suhu vs Konsumsi Energi")
            st.plotly_chart(style_fig(fig, 380), use_container_width=True)
        with sc2:
            fig = px.scatter(
                fdf.sample(min(2000, len(fdf)), random_state=42),
                x="Humidity (%)", y="Energy_Usage (kWh)", color="Building_Type",
                color_discrete_map=TYPE_COLORS, opacity=0.55,
            )
            fig.update_layout(title="Kelembapan vs Konsumsi Energi")
            st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    with t4:
        num_cols = [
            "Energy_Usage (kWh)", "Temperature (°C)", "Humidity (%)", "Hour",
            "DayOfWeek", "IsWeekend", "IsWorkingHour", "Building_ID_enc",
            "Occupancy_enc", "BuildingType_enc",
        ]
        corr = fdf[num_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            aspect="auto",
        )
        fig.update_layout(title="Matriks Korelasi Fitur Numerik")
        st.plotly_chart(style_fig(fig, 520), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🗂️ Lihat Tabel Data Mentah"):
        st.dataframe(
            fdf[["Timestamp", "Building_ID", "Building_Type", "Occupancy_Level",
                 "Temperature (°C)", "Humidity (%)", "Energy_Usage (kWh)"]]
            .sort_values("Timestamp", ascending=False),
            use_container_width=True,
            height=320,
        )
        csv = fdf.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Unduh data terfilter (CSV)", csv, "energyiq_filtered.csv", "text/csv")


# ──────────────────────────────────────────────────────────────────────────
# HALAMAN 3 — PREDIKSI ENERGI
# ──────────────────────────────────────────────────────────────────────────
def page_predict(df, meta):
    st.markdown('<div class="eiq-section-title">🔮 <span>Simulasi & Prediksi Konsumsi Energi</span></div>', unsafe_allow_html=True)
    st.caption("Atur parameter waktu, cuaca, dan gedung di bawah ini untuk memprediksi konsumsi energi per jam menggunakan model Random Forest.")

    building_types = meta["type_classes"]
    occupancy_levels = meta["occupancy_classes"]
    building_ids = meta["building_classes"]

    building_type_map = df.groupby("Building_ID")["Building_Type"].first().to_dict()

    col_form, col_result = st.columns([1.1, 1])

    with col_form:
        st.markdown("##### 🏢 Gedung")
        fc1, fc2 = st.columns(2)
        with fc1:
            building_id = st.selectbox("ID Gedung", building_ids, index=0)
        with fc2:
            default_type = building_type_map.get(building_id, building_types[0])
            building_type = st.selectbox(
                "Tipe Gedung", building_types,
                index=building_types.index(default_type) if default_type in building_types else 0,
            )

        st.markdown("##### 🕐 Waktu")
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            hour = st.slider("Jam", 0, 23, 9)
        with tc2:
            day_name = st.selectbox(
                "Hari", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"],
            )
            day_of_week = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"].index(day_name)
        with tc3:
            month = st.selectbox("Bulan", list(range(1, 13)), index=0,
                                  format_func=lambda m: datetime(2025, m, 1).strftime("%B"))

        st.markdown("##### 🌦️ Kondisi & Hunian")
        wc1, wc2 = st.columns(2)
        with wc1:
            temperature = st.slider("Suhu (°C)", -10.0, 40.0, 28.0, 0.5)
        with wc2:
            humidity = st.slider("Kelembapan (%)", 20.0, 95.0, 60.0, 0.5)
        occupancy = st.select_slider("Tingkat Hunian", options=occupancy_levels, value="Medium")

        predict_clicked = st.button("⚡ Jalankan Prediksi", type="primary", use_container_width=True)

    with col_result:
        if predict_clicked or "last_pred" in st.session_state:
            if predict_clicked:
                pred_kwh = predict_energy(
                    hour, day_of_week, temperature, humidity,
                    occupancy, building_type, building_id, month,
                )
                st.session_state["last_pred"] = pred_kwh
                st.session_state["last_inputs"] = dict(
                    building_id=building_id, building_type=building_type,
                    hour=hour, day_name=day_name, temperature=temperature,
                    humidity=humidity, occupancy=occupancy,
                )
            pred_kwh = st.session_state["last_pred"]
            inp = st.session_state["last_inputs"]

            pct_vs_baseline = (pred_kwh - BASELINE_KWH) / BASELINE_KWH * 100
            cost_per_hour = pred_kwh * IDR_PER_KWH

            if pct_vs_baseline > 10:
                pill_class, pill_label = "pill-high", "🔴 Di atas normal"
            elif pct_vs_baseline < -5:
                pill_class, pill_label = "pill-low", "🟢 Hemat energi"
            else:
                pill_class, pill_label = "pill-normal", "🟡 Normal"

            st.markdown(
                f"""
                <div class="eiq-result-card">
                    <div class="eiq-result-label">Prediksi Konsumsi Energi</div>
                    <div class="eiq-result-value">{pred_kwh:,.1f} <span style="font-size:1.2rem;color:#94a3b8;">kWh / jam</span></div>
                    <div style="margin-top:0.8rem;">
                        <span class="eiq-pill {pill_class}">{pill_label}</span>
                        <span style="color:#94a3b8;font-size:0.85rem;margin-left:0.5rem;">
                            {pct_vs_baseline:+.1f}% vs. baseline ({BASELINE_KWH:.0f} kWh)
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            rc1, rc2 = st.columns(2)
            rc1.metric("Estimasi Biaya / Jam", f"Rp {cost_per_hour:,.0f}", help=f"Tarif: Rp {IDR_PER_KWH:,.2f}/kWh")
            rc2.metric("Estimasi Biaya / Hari", f"Rp {cost_per_hour*24:,.0f}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Konteks: gedung sejenis pada jam yang sama**")
            ctx = df[(df["Building_Type"] == inp["building_type"]) & (df["Hour"] == inp["hour"])]
            if not ctx.empty:
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=ctx["Energy_Usage (kWh)"], name=f"Historis · {inp['building_type']} · jam {inp['hour']}",
                    marker_color=COLORS["primary"], boxpoints="outliers",
                ))
                fig.add_trace(go.Scatter(
                    x=[f"Historis · {inp['building_type']} · jam {inp['hour']}"], y=[pred_kwh],
                    mode="markers", name="Prediksi Anda",
                    marker=dict(color=COLORS["secondary"], size=16, symbol="star"),
                ))
                fig.update_layout(showlegend=True, yaxis_title="kWh")
                st.plotly_chart(style_fig(fig, 300), use_container_width=True)
        else:
            st.info("👈 Atur parameter di sebelah kiri, lalu klik **Jalankan Prediksi** untuk melihat hasilnya di sini.")

    st.markdown("---")
    st.markdown('<div class="eiq-section-title">🧪 Skenario Cepat</div>', unsafe_allow_html=True)
    st.caption("Beberapa skenario contoh — klik untuk mengisi otomatis form di atas (lalu klik Jalankan Prediksi).")

    scenarios = [
        {"nama": "Senin pagi, panas, padat", "hour": 9, "dow": 0, "temp": 33.0, "hum": 65.0, "occ": "High", "type": "Commercial", "bid": "B001"},
        {"nama": "Rabu siang, sejuk, sedang", "hour": 14, "dow": 2, "temp": 22.0, "hum": 55.0, "occ": "Medium", "type": "Educational", "bid": "B005"},
        {"nama": "Sabtu malam, sepi", "hour": 22, "dow": 5, "temp": 27.0, "hum": 70.0, "occ": "Low", "type": "Residential", "bid": "B010"},
        {"nama": "Jumat sore, terik, padat", "hour": 17, "dow": 4, "temp": 35.0, "hum": 80.0, "occ": "High", "type": "Industrial", "bid": "B015"},
    ]
    cols = st.columns(len(scenarios))
    for col, sc in zip(cols, scenarios):
        with col:
            st.markdown(f"**{sc['nama']}**")
            kwh = predict_energy(sc["hour"], sc["dow"], sc["temp"], sc["hum"], sc["occ"], sc["type"], sc["bid"])
            st.metric("Prediksi", f"{kwh:,.1f} kWh")
            st.caption(f"{sc['bid']} · {sc['type']} · {sc['occ']} hunian")


# ──────────────────────────────────────────────────────────────────────────
# HALAMAN 4 — PERFORMA MODEL
# ──────────────────────────────────────────────────────────────────────────
def page_model(meta, df):
    st.markdown('<div class="eiq-section-title">🧠 <span>Performa & Detail Model</span></div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE", f"{meta['rmse']:.2f} kWh", help="Root Mean Squared Error")
    m2.metric("MAE", f"{meta['mae']:.2f} kWh", help="Mean Absolute Error")
    m3.metric("R²", f"{meta['r2']:.4f}", help="Koefisien determinasi")
    m4.metric("MAPE", f"{meta['mape']:.2f}%", help="Mean Absolute Percentage Error")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="eiq-section-title">📌 Feature Importance</div>', unsafe_allow_html=True)
        fi = pd.DataFrame(meta["feature_importance"]).sort_values("Importance", ascending=True)
        fig = go.Figure(go.Bar(
            x=fi["Importance"], y=fi["Feature"], orientation="h",
            marker_color=COLORS["primary"],
        ))
        fig.update_layout(xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(style_fig(fig, 460), use_container_width=True)

    with col2:
        st.markdown('<div class="eiq-section-title">⚙️ Spesifikasi Model</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background:#1e293b;border:1px solid #334155;border-radius:14px;padding:1.2rem 1.4rem;">
            <table style="width:100%;font-size:0.92rem;color:#e2e8f0;">
            <tr><td style="color:#94a3b8;padding:0.35rem 0;">Algoritma</td><td style="text-align:right;font-weight:600;">Random Forest Regressor</td></tr>
            <tr><td style="color:#94a3b8;padding:0.35rem 0;">n_estimators</td><td style="text-align:right;font-weight:600;">300</td></tr>
            <tr><td style="color:#94a3b8;padding:0.35rem 0;">max_depth</td><td style="text-align:right;font-weight:600;">15</td></tr>
            <tr><td style="color:#94a3b8;padding:0.35rem 0;">min_samples_split</td><td style="text-align:right;font-weight:600;">5</td></tr>
            <tr><td style="color:#94a3b8;padding:0.35rem 0;">min_samples_leaf</td><td style="text-align:right;font-weight:600;">2</td></tr>
            <tr><td style="color:#94a3b8;padding:0.35rem 0;">Jumlah fitur</td><td style="text-align:right;font-weight:600;">{len(meta['features'])}</td></tr>
            <tr><td style="color:#94a3b8;padding:0.35rem 0;">Strategi split</td><td style="text-align:right;font-weight:600;">Kronologis 80/20</td></tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="eiq-section-title">🏷️ Kelas Kategori</div>', unsafe_allow_html=True)
        st.markdown(f"**Tipe Gedung** · {', '.join(meta['type_classes'])}")
        st.markdown(f"**Tingkat Hunian** · {', '.join(meta['occupancy_classes'])}")
        st.markdown(f"**Total Gedung** · {len(meta['building_classes'])} ({meta['building_classes'][0]}–{meta['building_classes'][-1]})")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="eiq-section-title">📋 Daftar Fitur yang Digunakan Model</div>', unsafe_allow_html=True)
    feat_df = pd.DataFrame(meta["feature_importance"]).sort_values("Importance", ascending=False).reset_index(drop=True)
    feat_df.index += 1
    feat_df["Importance"] = feat_df["Importance"].round(4)
    st.dataframe(feat_df, use_container_width=True, height=280)


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────
def main():
    load_css()

    if not DATA_OK:
        st.error(
            "⚠️ Gagal memuat data atau model. Pastikan file berikut berada "
            "di folder yang sama dengan `app.py`:\n\n"
            "`processed_data.csv`, `rf_model.pkl`, `le_building.pkl`, "
            "`le_occupancy.pkl`, `le_type.pkl`, `model_metadata.json`"
        )
        st.code(LOAD_ERROR)
        st.stop()

    page = sidebar_nav()

    st.sidebar.markdown(
        f"""
        <div style="font-size:0.78rem;color:#64748b;line-height:1.6;">
        <b style="color:#94a3b8;">Dataset</b><br>
        {len(df):,} baris · {df['Building_ID'].nunique()} gedung<br>
        {df['Timestamp'].min().date()} → {df['Timestamp'].max().date()}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-size:0.72rem;color:#475569;">
        EnergyIQ · Proyek EAS Pembelajaran Mesin<br>
        Model: Random Forest Regressor
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page.startswith("🏠"):
        page_overview(df)
    elif page.startswith("📊"):
        page_explore(df)
    elif page.startswith("🔮"):
        page_predict(df, meta)
    elif page.startswith("🧠"):
        page_model(meta, df)


if __name__ == "__main__":
    main()
