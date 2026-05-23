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
        try:
            label_txt = f"{val:.4f}" if isinstance(val, float) and val < 10 else f"{int(val)}"
        except Exception:
            label_txt = str(val)
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

def get_table_value(df, key_col, key, value_col, default=None):
    try:
        return df.loc[df[key_col] == key, value_col].values[0]
    except Exception:
        return default

# =========================
# LOAD DATA
# =========================

class_counts        = load_csv("class_counts.csv")
split_distribution  = load_csv("split_distribution.csv")
eda_summary         = load_csv("eda_summary.csv")
preprocessing_sum   = load_csv("preprocessing_summary.csv")
data_dictionary     = load_csv("data_dictionary.csv")
model_metrics       = load_csv("model_metrics.csv")
confusion_summary   = load_csv("confusion_summary.csv")
threshold_decision  = load_csv("threshold_decision.csv")
ab_testing          = load_csv("ab_testing.csv")
latency_result      = load_csv("latency_result.csv")
latency_summary     = load_csv("latency_summary.csv")

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
    Kaggle · MRL Dataset<br>
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
            Proyek ini mengangkat tema Healthy Lives & Well-being dan dikembangkan untuk membantu
            mendeteksi indikasi awal kantuk pengemudi berbasis computer vision. Dashboard ini merangkum
            hasil kerja learning path Data Scientist, mulai dari pengolahan dataset mata Open dan Closed,
            feature engineering melalui preprocessing citra, EDA, evaluasi model, A/B testing threshold,
            hingga latency test.
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_header("01", "Pipeline Analisis")
    steps = ["Gathering Data", "Assessing", "Cleaning", "Feature Eng.",
             "Split Data", "EDA", "Model Eval.", "Threshold", "A/B Testing", "Latency"]
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
        ("BQ 03", "Apakah feature engineering melalui preprocessing citra mampu menghasilkan input gambar yang seragam dan siap digunakan untuk model klasifikasi?"),
        ("BQ 04", "Seberapa baik performa model dalam mengklasifikasikan kondisi mata Open dan Closed berdasarkan accuracy, precision, recall, F1-score, dan confusion matrix?"),
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
        Dashboard ini berfokus pada <strong>ringkasan hasil analisis dari learning path Data Scientist</strong>,
        seperti insight data, feature engineering, evaluasi model, A/B testing, latency, dan kesimpulan.
        Fitur live camera serta integrasi aplikasi utama dikembangkan secara terpisah oleh learning path Full-Stack Developer.
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
            Bagian ini merangkum kondisi dataset sebelum masuk ke model. Fokusnya adalah jumlah data,
            keseimbangan kelas, hasil pengecekan kualitas data, dan pembagian train, validation, serta test set.
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_header("01", "Ringkasan EDA")
    if eda_summary is not None:
        st.dataframe(eda_summary, use_container_width=True, hide_index=True)
        insight_list([
            "Total dataset terdiri dari 4.000 gambar dengan distribusi kelas yang seimbang antara Closed dan Open.",
            "Tidak ditemukan missing value, duplikasi image path, maupun corrupt image sehingga dataset layak digunakan pada tahap berikutnya.",
            "Karena dataset berbentuk gambar, pengecekan kualitas dilakukan melalui metadata gambar, duplikasi path, dan corrupt image.",
            "Hasil pengecekan menunjukkan tidak ada data bermasalah yang perlu dihapus pada tahap cleaning.",
            "Variasi ukuran gambar ditemukan pada dataset sehingga data perlu diseragamkan pada tahap Feature Engineering."
        ])
    else:
        missing("eda_summary.csv")

    section_header("02", "Distribusi Dataset Open vs Closed")
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
                        <div class="metric-label">Rasio Kelas</div>
                        <div class="metric-value" style="color:#e67e22;">{ratio:.2f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        with c2:
            plot_bar(class_counts, "Class", "Count",
                     "Distribusi Kelas Dataset", "Class", "Jumlah Gambar",
                     colors=["#2563eb", "#e67e22"])

        insight_list([
            "Dataset memiliki jumlah data yang seimbang, yaitu 2.000 gambar Closed dan 2.000 gambar Open.",
            "Distribusi yang seimbang membantu mengurangi risiko model terlalu condong pada salah satu kelas.",
            "Keseimbangan kelas ini menjadi dasar yang baik sebelum data digunakan untuk pengembangan dan evaluasi model.",
        ])
    else:
        missing("class_counts.csv")

    section_header("03", "Distribusi Train / Validation / Test Split")
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
            "Dataset dibagi dengan rasio 70% train, 15% validation, dan 15% test menggunakan stratified split.",
            "Setiap subset memiliki proporsi kelas Closed dan Open yang sama sehingga kedua kelas tetap terwakili pada semua tahap.",
            "Validation set digunakan untuk threshold analysis dan A/B testing, sedangkan test set digunakan untuk evaluasi akhir model.",
        ])
    else:
        missing("split_distribution.csv")

    section_header("04", "Data Dictionary")
    if data_dictionary is not None:
        st.markdown("""
        <div class="info-box">
            Tabel berikut menjelaskan kolom dan objek data yang digunakan dalam pipeline analisis,
            mulai dari metadata gambar sampai subset validation dan test yang digunakan untuk evaluasi.
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(data_dictionary, use_container_width=True, hide_index=True)
    else:
        missing("data_dictionary.csv")

# =========================
# FEATURE ENGINEERING
# =========================

elif menu == "Feature Engineering":
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-title">Feature Engineering</div>
        <div class="page-hero-sub">
            Pada proyek ini, feature engineering dilakukan melalui preprocessing citra. Gambar mentah diseragamkan
            melalui resize, konversi grayscale, normalisasi nilai piksel, dan penyesuaian dimensi agar sesuai dengan input model CNN.
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_header("01", "Ringkasan Feature Engineering")
    if preprocessing_sum is not None:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.dataframe(preprocessing_sum, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("""
            <div class="metric-grid" style="grid-template-columns:1fr 1fr;">
                <div class="metric-card">
                    <div class="metric-label">Ukuran Input</div>
                    <div class="metric-value" style="font-size:1.3rem;">96×96</div>
                    <div class="metric-sub">piksel · grayscale</div>
                </div>
                <div class="metric-card" style="border-top-color:#e67e22;">
                    <div class="metric-label">Channel</div>
                    <div class="metric-value" style="font-size:1.3rem; color:#e67e22;">1</div>
                    <div class="metric-sub">grayscale</div>
                </div>
                <div class="metric-card" style="border-top-color:#7c3aed;">
                    <div class="metric-label">Nilai Min Piksel</div>
                    <div class="metric-value" style="font-size:1.3rem; color:#7c3aed;">0.0</div>
                    <div class="metric-sub">setelah normalisasi</div>
                </div>
                <div class="metric-card" style="border-top-color:#16a34a;">
                    <div class="metric-label">Nilai Max Piksel</div>
                    <div class="metric-value" style="font-size:1.3rem; color:#16a34a;">1.0</div>
                    <div class="metric-sub">setelah normalisasi</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        insight_list([
            "Seluruh gambar di-resize menjadi 96×96 piksel agar dimensi input model seragam.",
            "Konversi ke grayscale atau 1 channel digunakan untuk menyederhanakan informasi warna tanpa menghilangkan pola utama pada area mata.",
            "Normalisasi nilai piksel ke rentang 0–1 membuat input lebih stabil dan sesuai dengan format yang dibutuhkan model CNN.",
        ])
    else:
        missing("preprocessing_summary.csv")

# =========================
# MODEL EVALUATION
# =========================

elif menu == "Model Evaluation":
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-title">Model Evaluation</div>
        <div class="page-hero-sub">
            Model dari learning path AI dievaluasi menggunakan test set. Evaluasi difokuskan pada accuracy,
            precision, recall, F1-score, dan confusion matrix untuk melihat kemampuan model membedakan kelas Open dan Closed.
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
            "Model mencapai accuracy, precision, recall, dan F1-score sebesar 1.0000 pada test set.",
            "Precision 1.0000 menunjukkan tidak ada data Open yang salah diprediksi sebagai Closed pada test set.",
            "Recall 1.0000 menunjukkan seluruh data Closed berhasil dikenali tanpa ada yang terlewat pada test set.",
            "Hasil ini tetap perlu diuji kembali pada kondisi nyata menggunakan input webcam dan variasi lingkungan yang lebih luas.",
        ])
    else:
        missing("model_metrics.csv")

    section_header("02", "Confusion Matrix")
    if confusion_summary is not None:
        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.dataframe(confusion_summary, use_container_width=True, hide_index=True)
        with c2:
            try:
                tn = int(get_table_value(confusion_summary, "Komponen", "True Negative", "Nilai", 0))
                fp = int(get_table_value(confusion_summary, "Komponen", "False Positive", "Nilai", 0))
                fn = int(get_table_value(confusion_summary, "Komponen", "False Negative", "Nilai", 0))
                tp = int(get_table_value(confusion_summary, "Komponen", "True Positive", "Nilai", 0))
                matrix = np.array([[tn, fp], [fn, tp]])
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.imshow(matrix, cmap="Blues")
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["Pred: Open", "Pred: Closed"])
                ax.set_yticklabels(["Actual: Open", "Actual: Closed"])
                ax.set_title("Confusion Matrix (Test Set)")
                labels = [["TN", "FP"], ["FN", "TP"]]
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, f"{labels[i][j]}\n{matrix[i, j]}",
                                ha="center", va="center", fontsize=13,
                                color="white" if matrix[i, j] > max(matrix.max(), 1) * 0.5 else "#1e293b",
                                fontweight="bold")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception:
                st.dataframe(confusion_summary, use_container_width=True, hide_index=True)

        insight_list([
            "Confusion matrix menunjukkan jumlah prediksi benar dan salah untuk masing-masing kelas pada test set.",
            "Pada hasil evaluasi ini, seluruh data Open dan Closed pada test set berhasil diklasifikasikan dengan benar.",
            "Hasil tersebut konsisten dengan nilai precision dan recall yang mencapai 1.0000.",
        ])
    else:
        missing("confusion_summary.csv")

    section_header("03", "Threshold Analysis")
    if threshold_decision is not None:
        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.dataframe(threshold_decision, use_container_width=True, hide_index=True)
        with c2:
            try:
                t_default = float(get_table_value(threshold_decision, "Komponen", "Threshold default", "Nilai", 0.5))
                t_analysis = float(get_table_value(threshold_decision, "Komponen", "Threshold hasil analysis", "Nilai", 0.0))
                t_final = float(get_table_value(threshold_decision, "Komponen", "Threshold final yang digunakan", "Nilai", 0.5))
                st.markdown(f"""
                <div class="metric-grid" style="grid-template-columns:1fr 1fr 1fr;">
                    <div class="metric-card">
                        <div class="metric-label">Threshold Default</div>
                        <div class="metric-value" style="font-size:1.5rem;">{t_default}</div>
                        <div class="metric-sub">baseline</div>
                    </div>
                    <div class="metric-card" style="border-top-color:#e67e22;">
                        <div class="metric-label">Threshold Hasil Analysis</div>
                        <div class="metric-value" style="font-size:1.5rem; color:#e67e22;">{t_analysis}</div>
                        <div class="metric-sub">dari validation set</div>
                    </div>
                    <div class="metric-card" style="border-top-color:#16a34a;">
                        <div class="metric-label">Threshold Final</div>
                        <div class="metric-value" style="font-size:1.5rem; color:#16a34a;">{t_final}</div>
                        <div class="metric-sub">yang digunakan</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                pass

        insight_list([
            "Threshold digunakan sebagai batas keputusan untuk mengubah probabilitas model menjadi label Open atau Closed.",
            "Threshold hasil analysis diperoleh dari validation set, sedangkan test set digunakan untuk evaluasi akhir.",
            "Threshold final yang digunakan adalah 0.5 karena performanya setara dengan threshold hasil analysis, lebih umum, dan mudah dijelaskan.",
        ])
    else:
        missing("threshold_decision.csv")

# =========================
# A/B TESTING & LATENCY
# =========================

elif menu == "A/B Testing & Latency":
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-title">A/B Testing & Latency</div>
        <div class="page-hero-sub">
            Bagian ini membandingkan dua strategi threshold dan mengukur kecepatan inference model
            untuk melihat potensinya dalam mendukung sistem deteksi kantuk secara real-time.
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_header("01", "A/B Testing — Strategi Threshold")
    if ab_testing is not None:
        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.dataframe(ab_testing, use_container_width=True, hide_index=True)
            try:
                t_b = ab_testing.loc[ab_testing["Experiment"].str.contains("Best|Optimized", case=False, na=False), "Threshold"].values[0]
                threshold_b_label = str(t_b)
            except Exception:
                threshold_b_label = "hasil analysis"
            st.markdown(f"""
            <div class="bq-card" style="margin-top:14px;">
                <div class="bq-num">Strategi A</div>
                <div class="bq-text" style="font-size:0.82rem;">Threshold default <strong>0.5</strong></div>
            </div>
            <div class="bq-card">
                <div class="bq-num" style="color:#e67e22;">Strategi B</div>
                <div class="bq-text" style="font-size:0.82rem;">Threshold <strong>{threshold_b_label}</strong> dari validation set</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            metrics_cols = [m for m in ["Accuracy", "Precision", "Recall", "F1-score"] if m in ab_testing.columns]
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
            ax.set_title("A/B Testing: Default vs Best Threshold")
            ax.set_ylabel("Score")
            ax.legend(framealpha=0.9)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        insight_list([
            "A/B testing dilakukan untuk membandingkan threshold default 0.5 dengan threshold hasil analysis pada validation set.",
            "Kedua strategi menghasilkan performa yang sama pada metrik accuracy, precision, recall, dan F1-score.",
            "Karena hasilnya setara, threshold 0.5 dipilih sebagai threshold final agar keputusan model lebih stabil dan mudah dijelaskan.",
        ])
    else:
        missing("ab_testing.csv")

    section_header("02", "Latency Test — Kecepatan Inferensi")
    if latency_result is not None:
        try:
            lat_val = float(get_table_value(latency_result, "Metric", "Average Latency per Image (s)", "Value", 0.0))
            fps_val = float(get_table_value(latency_result, "Metric", "Estimated FPS", "Value", 0.0))
            sample_val = int(get_table_value(latency_result, "Metric", "Sample Size", "Value", 0))
            st.markdown(f"""
            <div class="metric-grid" style="grid-template-columns:1fr 1fr 1fr 1fr; margin-bottom:16px;">
                <div class="metric-card">
                    <div class="metric-label">Sample Size</div>
                    <div class="metric-value" style="font-size:1.35rem;">{sample_val}</div>
                    <div class="metric-sub">gambar diuji</div>
                </div>
                <div class="metric-card" style="border-top-color:#7c3aed;">
                    <div class="metric-label">Avg Latency / Gambar</div>
                    <div class="metric-value" style="font-size:1.2rem; color:#7c3aed;">{lat_val:.6f}s</div>
                    <div class="metric-sub">per frame inference</div>
                </div>
                <div class="metric-card" style="border-top-color:#e67e22;">
                    <div class="metric-label">Estimasi FPS</div>
                    <div class="metric-value" style="font-size:1.35rem; color:#e67e22;">{fps_val:.2f}</div>
                    <div class="metric-sub">frame per detik</div>
                </div>
                <div class="metric-card" style="border-top-color:#16a34a;">
                    <div class="metric-label">Kecepatan Inference</div>
                    <div class="metric-value" style="font-size:1.1rem; color:#16a34a;">Sangat Cepat</div>
                    <div class="metric-sub">berpotensi real-time</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass

        st.dataframe(latency_result, use_container_width=True, hide_index=True)
        if latency_summary is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(latency_summary, use_container_width=True, hide_index=True)

        insight_list([
            "Latency diukur sebagai rata-rata waktu inference model untuk memproses satu gambar pada kondisi pengujian terisolasi menggunakan 100 sampel.",
            "Rata-rata latency yang rendah menunjukkan model dapat melakukan prediksi dengan sangat cepat pada kondisi pengujian terisolasi.",
            "Estimasi FPS yang tinggi menunjukkan model berpotensi mendukung sistem real-time, tetapi performa aktual tetap perlu diuji setelah terintegrasi dengan webcam, backend, dan frontend.",
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
        <strong>Kesimpulan utama:</strong> Kondisi mata Open dan Closed dapat digunakan sebagai indikator awal
        untuk mendukung sistem deteksi kantuk pengemudi berbasis computer vision. Model mencapai performa sangat baik
        pada test set dengan accuracy, precision, recall, dan F1-score sebesar 1.0000.
    </div>
    """, unsafe_allow_html=True)

    section_header("01", "Ringkasan Pipeline")
    summary_steps = [
        ("Gathering Data",      "Dataset 4.000 gambar mata dari Kaggle dengan kelas Open dan Closed."),
        ("EDA",                 "Analisis distribusi kelas; tidak ditemukan missing value, duplikasi, atau corrupt image."),
        ("Feature Engineering", "Resize 96×96, konversi grayscale, normalisasi piksel 0–1, dan input 96×96×1."),
        ("Model Evaluation",    "Accuracy, precision, recall, dan F1-score mencapai 1.0000 pada test set."),
        ("Threshold Analysis",  "Threshold hasil analysis dibandingkan dengan threshold default; threshold final tetap 0.5."),
        ("A/B Testing",         "Threshold default 0.5 dan threshold hasil analysis menghasilkan performa yang setara."),
        ("Latency Test",        "Latency rendah dengan estimasi FPS tinggi pada kondisi pengujian terisolasi."),
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
        "Uji model langsung pada live webcam untuk melihat performanya pada kondisi nyata.",
        "Tambahkan variasi data seperti pengguna berkacamata, sudut wajah berbeda, dan kondisi low-light.",
        "Gunakan beberapa frame berturut-turut sebelum menetapkan status drowsy agar sistem tidak mudah salah membaca kedipan sebagai kantuk."
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
        Dashboard ini berfokus pada ringkasan hasil analisis dari learning path Data Scientist.
        Pengembangan fitur live camera dan integrasi aplikasi utama dilanjutkan oleh learning path Full-Stack Developer.
    </div>
    """, unsafe_allow_html=True)
