import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="KEMUDI - Data Science Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

ASSET_DIR = Path("streamlit_assets")

# =========================
# GLOBAL CSS STYLING
# =========================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:       #f4f6fb;
    --surface:  #ffffff;
    --surface2: #eef1f8;
    --border:   #dce2f0;
    --accent:   #2563eb;
    --accent2:  #e67e22;
    --accent3:  #7c3aed;
    --text:     #1e293b;
    --muted:    #64748b;
    --success:  #16a34a;
}

.stApp {
    background: var(--bg) !important;
    font-family: 'Syne', sans-serif;
}

section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

p, li, div, span, label {
    color: var(--text) !important;
}

/* --- Sidebar Brand --- */
.sidebar-brand {
    padding: 20px 0 28px;
    text-align: center;
}
.sidebar-logo {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent) !important;
    letter-spacing: 3px;
}
.sidebar-tagline {
    font-size: 0.68rem;
    color: var(--muted) !important;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-top: 6px;
    line-height: 1.6;
}

/* --- Page Hero --- */
.page-hero {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 5px solid var(--accent);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 28px;
}
.page-hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    color: var(--text) !important;
    margin: 0 0 8px;
}
.page-hero-sub {
    font-size: 0.92rem;
    color: var(--muted) !important;
    line-height: 1.8;
    max-width: 900px;
    text-align: justify;
    text-justify: inter-word;
}

/* --- Section Header --- */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 14px;
}
.section-number {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--accent) !important;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 1px;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.12rem;
    font-weight: 700;
    color: var(--text) !important;
    margin: 0;
}

/* --- Metric Cards --- */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 14px;
    margin: 18px 0;
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 4px solid var(--accent);
    border-radius: 10px;
    padding: 18px 16px;
}
.metric-label {
    font-size: 0.67rem;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--accent) !important;
    line-height: 1;
}
.metric-sub {
    font-size: 0.72rem;
    color: var(--muted) !important;
    margin-top: 6px;
}

/* --- Info Boxes --- */
.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 14px 0;
    font-size: 0.88rem;
    color: #1d4ed8 !important;
    line-height: 1.65;
}
.warning-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 14px 0;
    font-size: 0.88rem;
    color: #92400e !important;
    line-height: 1.65;
}
.success-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 14px 0;
    font-size: 0.88rem;
    color: #166534 !important;
    line-height: 1.65;
}

/* --- BQ Cards --- */
.bq-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 12px;
}
.bq-num {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent2) !important;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.bq-text {
    font-size: 0.87rem;
    color: var(--text) !important;
    line-height: 1.55;
}

/* --- Feature Tags --- */
.feature-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 14px 0;
}
.feature-tag {
    background: #f5f3ff;
    border: 1px solid #ddd6fe;
    color: #5b21b6 !important;
    padding: 5px 13px;
    border-radius: 20px;
    font-size: 0.77rem;
    font-family: 'Space Mono', monospace;
}

/* --- Pipeline --- */
.pipeline {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    margin: 14px 0;
    padding: 18px 20px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
}
.pipeline-step {
    background: var(--surface2);
    border: 1px solid var(--border);
    padding: 7px 13px;
    border-radius: 7px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    color: var(--text) !important;
    white-space: nowrap;
}
.pipeline-arrow {
    color: var(--accent) !important;
    font-size: 1rem;
    font-weight: 700;
}

/* --- Insight List with dots (no divider line) --- */
.insight-list {
    list-style: none;
    padding: 0;
    margin: 14px 0;
}
.insight-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 8px 0;
}
.insight-dot {
    width: 7px;
    height: 7px;
    min-width: 7px;
    background: var(--accent);
    border-radius: 50%;
    margin-top: 7px;
}
.insight-text {
    font-size: 0.9rem;
    color: var(--text) !important;
    line-height: 1.65;
}

/* --- Rec Items --- */
.rec-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid var(--border);
}
.rec-bullet {
    width: 6px;
    height: 6px;
    min-width: 6px;
    background: var(--accent2);
    border-radius: 50%;
    margin-top: 8px;
}

/* Table */
.stDataFrame {
    border-radius: 10px !important;
    overflow: hidden !important;
}

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =========================
# MATPLOTLIB LIGHT THEME
# =========================

plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#ffffff",
    "axes.edgecolor":   "#dce2f0",
    "axes.labelcolor":  "#64748b",
    "axes.titlecolor":  "#1e293b",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "text.color":       "#1e293b",
    "grid.color":       "#e2e8f0",
    "grid.linewidth":   0.6,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "monospace",
    "axes.titlesize":   12,
    "axes.labelsize":   10,
})

PALETTE = ["#2563eb", "#e67e22", "#7c3aed", "#16a34a", "#dc2626", "#0891b2"]

# ROC-AUC dari ROC Curve pada validation set
# Nilai ini digunakan sebagai pendukung analisis threshold, bukan sebagai baris threshold baru.
VALIDATION_ROC_AUC = 0.9996

# =========================
# HELPERS
# =========================

@st.cache_data
def load_csv(filename):
    p = ASSET_DIR / filename
    return pd.read_csv(p) if p.exists() else None

def missing(filename):
    st.markdown(
        f'<div class="warning-box">File <code>{filename}</code> belum ditemukan '
        f'di folder <code>streamlit_assets/</code>.</div>',
        unsafe_allow_html=True
    )

def plot_bar(df, x_col, y_col, title, xlabel, ylabel, colors=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    bar_colors = colors or PALETTE[:len(df)]
    bars = ax.bar(df[x_col], df[y_col], color=bar_colors,
                  width=0.5, edgecolor="none", zorder=3)
    ax.yaxis.grid(True, zorder=0, alpha=0.5)
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0)
    for bar, val in zip(bars, df[y_col]):
        label_txt = f"{val:.4f}" if isinstance(val, float) and val < 10 else f"{int(val)}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(df[y_col]) * 0.012,
                label_txt, ha="center", va="bottom", fontsize=9.5, color="#1e293b")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def section_header(num, title):
    st.markdown(f"""
    <div class="section-header">
        <span class="section-number">{num}</span>
        <p class="section-title">{title}</p>
    </div>
    """, unsafe_allow_html=True)

def insight_list(items):
    html = '<ul class="insight-list">'
    for item in items:
        html += (
            f'<li class="insight-item">'
            f'<span class="insight-dot"></span>'
            f'<span class="insight-text">{item}</span>'
            f'</li>'
        )
    html += "</ul>"
    st.markdown(html, unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================

class_counts       = load_csv("class_counts.csv")
split_distribution = load_csv("split_distribution.csv")
feature_summary    = load_csv("feature_summary.csv")
normalized_feature_summary = load_csv("normalized_feature_summary.csv")
model_metrics      = load_csv("model_metrics.csv")
threshold_comp     = load_csv("threshold_comparison.csv")
ab_testing         = load_csv("ab_testing.csv")
latency_result     = load_csv("latency_result.csv")

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">KEMUDI</div>
        <div class="sidebar-tagline">Data Science Dashboard<br>Pemantauan Kantuk Pengemudi</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    menu = st.radio(
        "Navigasi",
        [
            "Project Overview",
            "Data Insight",
            "Feature Engineering",
            "Model Evaluation",
            "A/B Testing & Latency",
            "Conclusion"
        ]
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#94a3b8; line-height:1.9;">
    <strong style="color:#64748b;">Stack Analisis</strong><br>
    Python · TensorFlow · OpenCV<br>
    Scikit-learn · Streamlit<br><br>
    <strong style="color:#64748b;">Dataset</strong><br>
    Kaggle · Eye State Images<br>
    Kelas: Open · Closed
    </div>
    """, unsafe_allow_html=True)

# =========================
# PROJECT OVERVIEW
# =========================

if menu == "Project Overview":
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-title">KEMUDI: Sistem Pemantauan Kantuk Pengemudi</div>
        <div class="page-hero-sub">
            Proyek capstone ini mengangkat tema Healthy Lives & Well-being dan dikembangkan untuk membantu
            mendeteksi indikasi awal kantuk pengemudi berbasis computer vision. Dashboard ini merangkum
            proses dan hasil kerja learning path Data Scientist, mulai dari pengolahan dataset mata Open dan Closed,
            preprocessing, EDA, feature engineering, evaluasi model, hingga latency test.
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_header("01", "Pipeline Analisis")
    steps = ["Gathering Data", "Assessing", "Cleaning", "Preprocessing",
             "EDA", "Feature Eng.", "Model Eval.", "Threshold", "A/B Testing", "Latency"]
    pipeline_html = '<div class="pipeline">'
    for i, s in enumerate(steps):
        pipeline_html += f'<span class="pipeline-step">{s}</span>'
        if i < len(steps) - 1:
            pipeline_html += '<span class="pipeline-arrow">&#8594;</span>'
    pipeline_html += "</div>"
    st.markdown(pipeline_html, unsafe_allow_html=True)

    section_header("02", "Business Questions")
    bqs = [
        ("BQ 01", "Apakah kondisi mata terbuka dan tertutup dapat digunakan sebagai indikator awal dalam sistem deteksi kantuk pengemudi?"),
        ("BQ 02", "Bagaimana karakteristik dan keseimbangan dataset Open dan Closed yang digunakan dalam pengembangan model?"),
        ("BQ 03", "Apakah proses preprocessing mampu menghasilkan input gambar yang seragam dan siap digunakan untuk model klasifikasi?"),
        ("BQ 04", "Seberapa baik performa model dalam mengklasifikasikan kondisi mata berdasarkan accuracy, precision, recall, F1-score, dan ROC-AUC?"),
        ("BQ 05", "Threshold prediksi berapa yang paling optimal untuk membedakan kondisi mata Open dan Closed berdasarkan validation set?"),
        ("BQ 06", "Apakah model memiliki latency yang cukup rendah untuk mendukung sistem deteksi kantuk secara real-time?"),
    ]
    col1, col2 = st.columns(2)
    for i, (num, text) in enumerate(bqs):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"""
            <div class="bq-card">
                <div class="bq-num">{num}</div>
                <div class="bq-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top:20px;">
        Dashboard ini berfokus pada <strong>ringkasan hasil analisis yang dilakukan oleh learning path Data Scientist</strong>,
        seperti insight data, evaluasi model, A/B testing, latency, dan kesimpulan. Fitur live camera serta integrasi
        aplikasi utama dikembangkan secara terpisah oleh learning path Full-Stack Developer.
    </div>
    """, unsafe_allow_html=True)

# =========================
# DATA INSIGHT
# =========================

elif menu == "Data Insight":
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-title">Data Insight</div>
        <div class="page-hero-sub">
            Distribusi dataset dan pembagian data train, validation, serta test set digunakan untuk memastikan
            representasi kelas Open dan Closed tetap seimbang.
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_header("01", "Distribusi Dataset Open vs Closed")

    if class_counts is not None:
        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.dataframe(class_counts, use_container_width=True, hide_index=True)
            if "Count" in class_counts.columns and len(class_counts) == 2:
                total = class_counts["Count"].sum()
                ratio = class_counts["Count"].iloc[0] / class_counts["Count"].iloc[1]
                st.markdown(f"""
                <div class="metric-grid" style="grid-template-columns:1fr 1fr; margin-top:14px;">
                    <div class="metric-card">
                        <div class="metric-label">Total Gambar</div>
                        <div class="metric-value">{total:,}</div>
                    </div>
                    <div class="metric-card" style="border-top-color:#e67e22;">
                        <div class="metric-label">Rasio Open/Closed</div>
                        <div class="metric-value" style="color:#e67e22;">{ratio:.2f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        with c2:
            plot_bar(class_counts, "Class", "Count",
                     "Distribusi Kelas Dataset", "Class", "Jumlah Gambar",
                     colors=["#2563eb", "#e67e22"])

        # Dynamic insight if values are available, fallback to general wording.
        try:
            open_count = int(class_counts.loc[class_counts["Class"].str.lower() == "open", "Count"].values[0])
            closed_count = int(class_counts.loc[class_counts["Class"].str.lower() == "closed", "Count"].values[0])
            first_insight = f"Dataset memiliki jumlah data yang seimbang, yaitu {open_count} gambar Open dan {closed_count} gambar Closed."
        except Exception:
            first_insight = "Dataset memiliki distribusi kelas Open dan Closed yang relatif seimbang."

        insight_list([
            first_insight,
            "Distribusi yang seimbang membantu mengurangi risiko model terlalu condong pada salah satu kelas.",
            "Keseimbangan kelas ini menjadi dasar yang baik sebelum data digunakan untuk proses training dan evaluasi model.",
        ])
    else:
        missing("class_counts.csv")

    section_header("02", "Distribusi Train / Validation / Test Split")

    if split_distribution is not None:
        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.dataframe(split_distribution, use_container_width=True, hide_index=True)
        with c2:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            x = np.arange(len(split_distribution["Class"]))
            w = 0.25
            for i, (col, color) in enumerate(zip(["Train", "Validation", "Test"], PALETTE[:3])):
                ax.bar(x + i * w, split_distribution[col], w,
                       label=col, color=color, edgecolor="none", zorder=3)
            ax.set_xticks(x + w)
            ax.set_xticklabels(split_distribution["Class"])
            ax.set_title("Distribusi Label per Split")
            ax.set_ylabel("Jumlah Data")
            ax.yaxis.grid(True, zorder=0, alpha=0.5)
            ax.legend(framealpha=0.9)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        insight_list([
            "Dataset dibagi menjadi train, validation, dan test set menggunakan stratified split.",
            "Setiap subset memiliki proporsi kelas Open dan Closed yang sama, sehingga kedua kelas tetap terwakili pada semua tahap.",
            "Pembagian yang seimbang membantu proses evaluasi model menjadi lebih adil karena performa tidak hanya dinilai dari satu kelas tertentu.",
        ])
    else:
        missing("split_distribution.csv")

# =========================
# FEATURE ENGINEERING
# =========================

elif menu == "Feature Engineering":
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-title">Feature Engineering</div>
        <div class="page-hero-sub">
            Ekstraksi fitur numerik dari gambar mata dilakukan untuk memahami karakteristik visual kelas Open dan Closed
            secara lebih terukur.
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_header("01", "Fitur yang Diekstraksi")

    features = [
        ("brightness",        "Rata-rata tingkat kecerahan piksel pada gambar."),
        ("contrast",          "Variasi gelap-terang pada gambar berdasarkan sebaran nilai pixel."),
        ("pixel_range",       "Selisih antara nilai piksel paling gelap dan paling terang."),
        ("dark_pixel_ratio",  "Proporsi area gelap terhadap keseluruhan piksel gambar."),
        ("bright_pixel_ratio","Proporsi area terang terhadap keseluruhan piksel gambar."),
        ("edge_density",      "Proporsi tepi atau garis terdeteksi menggunakan Canny edge detection."),
    ]

    tags_html = '<div class="feature-tags">'
    for name, _ in features:
        tags_html += f'<span class="feature-tag">{name}</span>'
    tags_html += '</div>'
    st.markdown(tags_html, unsafe_allow_html=True)

    cols = st.columns(3)
    for i, (name, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="bq-card" style="margin-bottom:12px; min-height:88px;">
                <div class="bq-num" style="color:#7c3aed;">{name}</div>
                <div class="bq-text" style="font-size:0.82rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    section_header("02", "Rata-rata Fitur per Kelas")

    if feature_summary is not None:
        st.markdown("""
        <div class="info-box">
            Tabel berikut menampilkan nilai rata-rata fitur dalam skala asli. Nilai ini digunakan
            untuk melihat karakteristik numerik dari setiap kelas secara langsung.
        </div>
        """, unsafe_allow_html=True)

        class_col = "class" if "class" in feature_summary.columns else "Class"
        st.dataframe(feature_summary, use_container_width=True, hide_index=True)
    else:
        missing("feature_summary.csv")

    section_header("03", "Normalized Feature Comparison")

    if normalized_feature_summary is not None:
        st.markdown("""
        <div class="info-box">
            Grafik berikut menggunakan fitur yang sudah dinormalisasi ke rentang 0–1.
            Normalisasi ini hanya digunakan untuk visualisasi agar fitur dengan skala berbeda
            dapat dibandingkan dengan lebih mudah, bukan untuk mengubah hasil model utama.
        </div>
        """, unsafe_allow_html=True)

        class_col_norm = "class" if "class" in normalized_feature_summary.columns else "Class"
        st.dataframe(normalized_feature_summary, use_container_width=True, hide_index=True)

        feature_cols = [c for c in normalized_feature_summary.columns if c != class_col_norm]
        classes = normalized_feature_summary[class_col_norm].tolist()

        fig, ax = plt.subplots(figsize=(11, 4.5))
        x = np.arange(len(feature_cols))
        w = 0.35

        for i, (cls, color) in enumerate(zip(classes, PALETTE)):
            vals = normalized_feature_summary[
                normalized_feature_summary[class_col_norm] == cls
            ][feature_cols].values.flatten()

            ax.bar(
                x + i * w,
                vals,
                w,
                label=cls,
                color=color,
                edgecolor="none",
                zorder=3
            )

        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(feature_cols, rotation=15, ha="right")
        ax.set_title("Normalized Feature Comparison: Open vs Closed")
        ax.set_ylabel("Normalized Average Value")
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, zorder=0, alpha=0.5)
        ax.legend(framealpha=0.9)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        insight_list([
            "Tabel tetap menampilkan nilai asli setiap fitur, sedangkan grafik normalized digunakan agar perbandingan antarfitur lebih mudah dibaca.",
            "Normalisasi membantu fitur dengan skala kecil seperti dark_pixel_ratio, bright_pixel_ratio, dan edge_density terlihat lebih jelas pada grafik.",
            "Feature engineering pada dashboard ini digunakan sebagai analisis pendukung untuk memahami karakteristik data, bukan sebagai input utama model CNN.",
        ])
    else:
        missing("normalized_feature_summary.csv")

# =========================
# MODEL EVALUATION
# =========================

elif menu == "Model Evaluation":
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-title">Model Evaluation</div>
        <div class="page-hero-sub">
            Performa model klasifikasi mata Open dan Closed dievaluasi menggunakan accuracy, precision, recall, F1-score, serta analisis threshold. Nilai ROC-AUC pada validation set digunakan sebagai pendukung untuk melihat kemampuan model membedakan kedua kelas pada berbagai threshold.
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_header("01", "Metrik Evaluasi Model")

    if model_metrics is not None:
        metric_accent = {
            "Accuracy":  "#2563eb",
            "Precision": "#7c3aed",
            "Recall":    "#0891b2",
            "F1-score":  "#16a34a",
            "ROC-AUC":   "#e67e22",
        }
        cards_html = '<div class="metric-grid">'
        for _, row in model_metrics.iterrows():
            color = metric_accent.get(row["Metric"], "#2563eb")
            grade = "Excellent" if row["Value"] >= 0.9 else ("Good" if row["Value"] >= 0.8 else "Fair")
            cards_html += f"""
            <div class="metric-card" style="border-top-color:{color};">
                <div class="metric-label">{row['Metric']}</div>
                <div class="metric-value" style="color:{color};">{row['Value']:.4f}</div>
                <div class="metric-sub">{grade}</div>
            </div>"""
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.dataframe(model_metrics, use_container_width=True, hide_index=True)
        with c2:
            fig, ax = plt.subplots(figsize=(7, 4))
            bar_colors = [metric_accent.get(m, "#2563eb") for m in model_metrics["Metric"]]
            bars = ax.bar(model_metrics["Metric"], model_metrics["Value"],
                          color=bar_colors, edgecolor="none", width=0.5, zorder=3)
            ax.set_ylim(0, 1.12)
            ax.axhline(0.9, color="#16a34a", linewidth=1.2, linestyle="--",
                       alpha=0.7, label="Target 0.9")
            ax.yaxis.grid(True, zorder=0, alpha=0.5)
            ax.set_title("Skor per Metrik Evaluasi")
            ax.set_ylabel("Score")
            for bar, val in zip(bars, model_metrics["Value"]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{val:.4f}", ha="center", va="bottom",
                        fontsize=9, color="#1e293b")
            ax.legend(framealpha=0.9)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        insight_list([
            "Model menunjukkan performa yang sangat baik dalam mengklasifikasikan kondisi mata Open dan Closed, dengan accuracy sebesar 0.9908 dan F1-score sebesar 0.9907.",
            "Precision sebesar 1.0000 menunjukkan bahwa prediksi positif model sangat tepat pada data uji yang digunakan.",
            "Recall sebesar 0.9817 menunjukkan bahwa sebagian besar data pada kelas target berhasil dikenali, meskipun masih terdapat sedikit data yang belum terdeteksi dengan sempurna.",
            "Secara keseluruhan, hasil evaluasi menunjukkan model sudah layak digunakan untuk tahap integrasi awal, tetapi tetap perlu diuji kembali pada data real-time dari webcam.",
        ])
    else:
        missing("model_metrics.csv")

    section_header("02", "Threshold Analysis")

    st.markdown(f"""
    <div class="metric-grid" style="grid-template-columns:minmax(220px, 300px); margin-bottom:16px;">
        <div class="metric-card" style="border-top-color:#7c3aed;">
            <div class="metric-label">Validation ROC-AUC</div>
            <div class="metric-value" style="color:#7c3aed;">{VALIDATION_ROC_AUC:.4f}</div>
            <div class="metric-sub">ROC Curve pada validation set</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        Nilai ROC-AUC pada validation set sebesar <strong>{VALIDATION_ROC_AUC:.4f}</strong>
        menunjukkan bahwa model memiliki kemampuan yang sangat baik dalam membedakan kelas
        Open dan Closed pada berbagai pilihan threshold. Nilai ini digunakan sebagai dasar
        pendukung dalam proses pemilihan threshold optimal.
    </div>
    """, unsafe_allow_html=True)

    if threshold_comp is not None:
        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.dataframe(threshold_comp, use_container_width=True, hide_index=True)
        with c2:
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(
                threshold_comp["Threshold"].astype(str),
                threshold_comp["F1-score"],
                color=["#2563eb", "#e67e22"],
                width=0.4, edgecolor="none", zorder=3
            )
            ax.set_ylim(0, 1.12)
            ax.yaxis.grid(True, zorder=0, alpha=0.5)
            ax.set_title("Perbandingan F1-score: Default vs Optimal Threshold")
            ax.set_xlabel("Threshold")
            ax.set_ylabel("F1-score")
            for bar, val in zip(bars, threshold_comp["F1-score"]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{val:.4f}", ha="center", va="bottom",
                        fontsize=10, color="#1e293b")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        insight_list([
            f"ROC-AUC validation sebesar {VALIDATION_ROC_AUC:.4f} menunjukkan kemampuan pemisahan kelas Open dan Closed yang sangat baik pada berbagai threshold.",
            "Threshold optimal diperoleh dari ROC Curve menggunakan Youden’s Index pada validation set.",
            "Pada hasil ini, threshold default 0.5 dan threshold optimal menghasilkan F1-score yang sama sehingga model terlihat cukup stabil terhadap perubahan threshold.",
            "Threshold optimal tetap dicatat sebagai acuan karena diperoleh melalui proses evaluasi, bukan hanya menggunakan nilai default.",
        ])
    else:
        missing("threshold_comparison.csv")

# =========================
# A/B TESTING & LATENCY
# =========================

elif menu == "A/B Testing & Latency":
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-title">A/B Testing & Latency</div>
        <div class="page-hero-sub">
            Perbandingan eksperimen offline dua strategi threshold dan pengujian kecepatan inference model untuk melihat potensi penggunaan sistem secara real-time.
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_header("01", "A/B Testing — Strategi Threshold")

    if ab_testing is not None:
        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.dataframe(ab_testing, use_container_width=True, hide_index=True)
            st.markdown("""
            <div class="bq-card" style="margin-top:14px;">
                <div class="bq-num">Strategi A</div>
                <div class="bq-text" style="font-size:0.82rem;">Threshold default <strong>0.5</strong></div>
            </div>
            <div class="bq-card">
                <div class="bq-num" style="color:#e67e22;">Strategi B</div>
                <div class="bq-text" style="font-size:0.82rem;">Threshold <strong>optimal</strong> dari validation set (Youden's Index)</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            metrics_cols = ["Accuracy", "Precision", "Recall", "F1-score"]
            fig, ax = plt.subplots(figsize=(9, 4.5))
            x = np.arange(len(metrics_cols))
            w = 0.35
            for i, (_, row) in enumerate(ab_testing.iterrows()):
                vals = [row[m] for m in metrics_cols]
                ax.bar(x + i * w, vals, w, label=row["Experiment"],
                       color=PALETTE[i], edgecolor="none", zorder=3)
            ax.set_xticks(x + w / 2)
            ax.set_xticklabels(metrics_cols)
            ax.set_ylim(0, 1.12)
            ax.yaxis.grid(True, zorder=0, alpha=0.5)
            ax.set_title("A/B Testing: Default vs Optimized Threshold")
            ax.set_ylabel("Score")
            ax.legend(framealpha=0.9)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        insight_list([
            "A/B Testing dilakukan secara offline menggunakan Python pada validation set untuk membandingkan dua strategi threshold.",
            "Pada hasil ini, threshold default 0.5 dan threshold optimal menghasilkan performa yang sama pada accuracy, precision, recall, dan F1-score.",
            "Hasil tersebut menunjukkan bahwa model cukup stabil terhadap perubahan threshold sehingga pergeseran threshold tidak mengubah hasil klasifikasi secara signifikan.",
        ])
    else:
        missing("ab_testing.csv")

    section_header("02", "Latency Test — Kecepatan Inferensi")

    if latency_result is not None:
        try:
            lat_val = latency_result[
                latency_result["Metric"] == "Average Latency per Image (s)"
            ]["Value"].values[0]
            fps_val = latency_result[
                latency_result["Metric"] == "Estimated FPS"
            ]["Value"].values[0]

            st.markdown(f"""
            <div class="metric-grid" style="grid-template-columns:1fr 1fr 1fr; margin-bottom:16px;">
                <div class="metric-card">
                    <div class="metric-label">Avg Latency / Gambar</div>
                    <div class="metric-value" style="font-size:1.35rem;">{lat_val:.6f}s</div>
                    <div class="metric-sub">per frame inference</div>
                </div>
                <div class="metric-card" style="border-top-color:#e67e22;">
                    <div class="metric-label">Estimasi FPS</div>
                    <div class="metric-value" style="font-size:1.35rem; color:#e67e22;">{fps_val:.2f}</div>
                    <div class="metric-sub">frame per detik</div>
                </div>
                <div class="metric-card" style="border-top-color:#16a34a;">
                    <div class="metric-label">Kecepatan Inference</div>
                    <div class="metric-value" style="font-size:1.35rem; color:#16a34a;">Sangat Cepat</div>
                    <div class="metric-sub">berpotensi real-time</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass

        st.dataframe(latency_result, use_container_width=True, hide_index=True)

        insight_list([
            "Latency diukur sebagai rata-rata waktu inference model untuk memproses satu gambar pada kondisi pengujian terisolasi.",
            "Rata-rata latency yang rendah menunjukkan bahwa model dapat melakukan prediksi dengan sangat cepat.",
            "Hasil FPS yang tinggi menunjukkan bahwa model berpotensi mendukung sistem real-time, tetapi performa aktual tetap perlu diuji kembali setelah terintegrasi dengan webcam, backend, dan frontend.",
        ])
    else:
        missing("latency_result.csv")

# =========================
# CONCLUSION
# =========================

elif menu == "Conclusion":
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-title">Conclusion</div>
        <div class="page-hero-sub">
            Ringkasan hasil analisis Data Science serta rekomendasi pengembangan sistem KEMUDI ke depan.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="success-box" style="font-size:0.95rem; padding:18px 22px;">
        <strong>Kesimpulan utama:</strong> Kondisi mata Open dan Closed dapat digunakan sebagai indikator awal untuk mendukung sistem deteksi kantuk pengemudi berbasis computer vision. Model yang dikembangkan menunjukkan performa yang baik pada dataset yang digunakan.
    </div>
    """, unsafe_allow_html=True)

    section_header("01", "Ringkasan Pipeline")

    summary_steps = [
        ("Gathering Data",      "Dataset gambar mata dari Kaggle dengan label Open dan Closed."),
        ("EDA",                 "Analisis distribusi kelas dan visualisasi sampel gambar."),
        ("Preprocessing",       "Resize, normalisasi, dan konversi grayscale untuk input model."),
        ("Feature Engineering", "Ekstraksi 6 fitur numerik sebagai analisis pendukung Data Science."),
        ("Model Evaluation",    "Evaluasi accuracy, precision, recall, F1-score, Evaluasi accuracy, precision, recall, F1-score, serta ROC-AUC pada validation set."),
        ("Threshold Analysis",  "Optimasi threshold menggunakan Youden's Index dari ROC Curve."),
        ("A/B Testing",         "Perbandingan offline strategi threshold default vs optimal."),
        ("Latency Test",        "Pengukuran rata-rata waktu inferensi model per frame gambar."),
    ]

    cols = st.columns(4)
    for i, (step, desc) in enumerate(summary_steps):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="bq-card" style="margin-bottom:12px; min-height:95px;">
                <div class="bq-num" style="color:#16a34a;">{step}</div>
                <div class="bq-text" style="font-size:0.8rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    section_header("02", "Keterbatasan")

    insight_list([
        "Hasil evaluasi belum sepenuhnya mewakili kondisi nyata karena pengujian dilakukan pada dataset statis.",
        "Variasi pencahayaan, sudut wajah, dan pengguna berkacamata belum tercakup secara menyeluruh dalam dataset.",
        "Latency terukur pada kondisi pengujian terisolasi dan belum memperhitungkan beban sistem pada aplikasi utama.",
    ])

    section_header("03", "Rekomendasi Pengembangan")

    recs = [
        "Uji model pada live webcam untuk melihat performa model dalam kondisi nyata dengan berbagai variasi pencahayaan.",
        "Gunakan beberapa frame berturut-turut sebelum menentukan status drowsy untuk mengurangi risiko false positive.",
        "Tambahkan dataset dengan variasi pengguna berkacamata, sudut wajah berbeda, dan kondisi low-light.",
        "Integrasikan output model dengan sistem alert audio atau visual pada aplikasi utama.",
        "Lakukan evaluasi lanjutan setelah model terhubung penuh dengan backend dan frontend pada aplikasi utama.",
    ]

    for text in recs:
        st.markdown(f"""
        <div class="rec-item">
            <span class="rec-bullet"></span>
            <span style="font-size:0.9rem; color:#1e293b; line-height:1.6;">{text}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top:24px;">
        Dashboard ini berfokus pada ringkasan hasil analisis Data Science. Pengembangan fitur live camera dan integrasi aplikasi utama dilanjutkan oleh learning path Full-Stack.
    </div>
    """, unsafe_allow_html=True)