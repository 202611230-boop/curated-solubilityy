import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ── 페이지 설정 ──────────────────────────────────────────
st.set_page_config(
    page_title="화합물 수용성 등급 예측",
    page_icon="🧪",
    layout="wide"
)

# ── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 800;
        color: #1a3c5e; text-align: center; padding: 1rem 0 0.3rem;
    }
    .sub-title {
        font-size: 1rem; color: #555; text-align: center; margin-bottom: 1.5rem;
    }
    .grade-box {
        border-radius: 12px; padding: 1.2rem 1.5rem;
        font-size: 1.4rem; font-weight: 700;
        text-align: center; margin: 0.5rem 0;
    }
    .info-card {
        background: #f0f6ff; border-left: 4px solid #2874c5;
        border-radius: 8px; padding: 0.8rem 1rem; margin: 0.4rem 0;
        font-size: 0.93rem;
    }
    .metric-card {
        background: #ffffff; border: 1px solid #e0e8f0;
        border-radius: 10px; padding: 0.9rem 1rem; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── 등급 색상 및 설명 ────────────────────────────────────
GRADE_INFO = {
    'G1': {'color': '#1a7f37', 'bg': '#d4f0de', 'label': 'G1 — 매우 높은 수용성', 'desc': 'LogS ≥ 0'},
    'G2': {'color': '#2d6a9f', 'bg': '#d0e8f8', 'label': 'G2 — 높은 수용성',     'desc': '-1 ≤ LogS < 0'},
    'G3': {'color': '#b08000', 'bg': '#fef3c7', 'label': 'G3 — 보통 수용성',     'desc': '-2 ≤ LogS < -1'},
    'G4': {'color': '#c05000', 'bg': '#fde8d0', 'label': 'G4 — 낮은 수용성',     'desc': '-3 ≤ LogS < -2'},
    'G5': {'color': '#b91c1c', 'bg': '#fde0e0', 'label': 'G5 — 매우 낮은 수용성','desc': 'LogS < -3'},
}

# ── 모델 학습 (캐싱) ─────────────────────────────────────
@st.cache_resource
def train_model():
    import kagglehub, os
    path = kagglehub.dataset_download("sorkun/aqsoldb-a-curated-aqueous-solubility-dataset")
    csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
    df_raw = pd.read_csv(os.path.join(path, csv_file))

    # 컬럼명 한글로 맞추기 (노트북과 동일)
    col_map = {
        df_raw.columns[0]: '화합물명',
        df_raw.columns[1]: '분자량',
        df_raw.columns[2]: '수소결합수용체수',
        df_raw.columns[3]: '수소결합공여체수',
        df_raw.columns[4]: '회전가능결합수',
        df_raw.columns[5]: '지질친화성',
        df_raw.columns[6]: '극성표면적',
        df_raw.columns[7]: '표준편차',
        df_raw.columns[8]: '용해도등급',
    }
    df_raw.rename(columns=col_map, inplace=True)

    feature_cols = ['분자량','수소결합공여체수','수소결합수용체수','극성표면적','회전가능결합수','지질친화성']
    target_col   = '용해도등급'

    df = df_raw[feature_cols + [target_col]].dropna().copy()
    df = df[(df['분자량'] > 10) & (df['분자량'] < 1500)]
    df = df[(df['지질친화성'] > -10) & (df['지질친화성'] < 15)]
    df = df[(df['극성표면적'] >= 0) & (df['극성표면적'] < 500)]
    df = df[(df['수소결합공여체수'] >= 0) & (df['수소결합공여체수'] < 20)]
    df = df[(df['수소결합수용체수'] >= 0) & (df['수소결합수용체수'] < 30)]

    # 파생 변수
    df['고지용성']   = (df['지질친화성'] >= 3).astype(int)
    df['수소결합총합'] = df['수소결합공여체수'] + df['수소결합수용체수']
    df['고분자량']   = (df['분자량'] >= 500).astype(int)

    feature_cols_new = feature_cols + ['고지용성','수소결합총합','고분자량']

    X = df[feature_cols_new]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, scaler, feature_cols_new, df

# ── 앱 시작 ──────────────────────────────────────────────
st.markdown('<div class="main-title">🧪 화합물 수용성 등급 예측</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AQSolDB 기반 · 분자 특성으로 수용성 등급(G1~G5)을 예측합니다</div>', unsafe_allow_html=True)

with st.spinner("데이터 로드 및 모델 학습 중..."):
    model, scaler, feature_cols_new, df = train_model()

# ── 탭 구성 ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔬 등급 예측", "📊 데이터 탐색", "📈 모델 성능"])

# ════════════════════════════════════════════════════════
# TAB 1: 예측
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("분자 특성 입력")
    st.caption("슬라이더로 값을 조정하면 실시간으로 등급이 예측됩니다.")

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        mw   = st.slider("⚖️ 분자량 (MW)",         10.0, 1500.0, 250.0, 1.0)
        logp = st.slider("💧 지질친화성 (LogP)",    -10.0, 15.0,   1.5,  0.01)
        tpsa = st.slider("🔵 극성표면적 (TPSA)",     0.0,  500.0,  60.0, 0.5)

    with col_r:
        hd  = st.slider("🔗 수소결합 공여체 수",     0, 20, 1)
        ha  = st.slider("🔗 수소결합 수용체 수",     0, 30, 3)
        nrb = st.slider("🔄 회전가능 결합 수",       0, 50, 3)

    # 파생 변수 계산
    high_logp   = int(logp >= 3)
    hbond_total = hd + ha
    high_mw     = int(mw >= 500)

    input_df = pd.DataFrame(
        [[mw, hd, ha, tpsa, nrb, logp, high_logp, hbond_total, high_mw]],
        columns=feature_cols_new
    )

    pred_class = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]
    grade      = GRADE_INFO[pred_class]

    st.markdown("---")
    st.subheader("예측 결과")

    res_col, prob_col = st.columns([1, 1], gap="large")

    with res_col:
        st.markdown(f"""
        <div class="grade-box" style="background:{grade['bg']};color:{grade['color']};">
            {grade['label']}<br>
            <span style="font-size:0.85rem;font-weight:400;">({grade['desc']})</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-card">
            <b>수용성 등급 기준</b><br>
            G1: 매우 높음 (LogS ≥ 0)<br>
            G2: 높음 (-1 ~ 0)<br>
            G3: 보통 (-2 ~ -1)<br>
            G4: 낮음 (-3 ~ -2)<br>
            G5: 매우 낮음 (LogS &lt; -3)
        </div>
        """, unsafe_allow_html=True)

    with prob_col:
        fig, ax = plt.subplots(figsize=(5, 3))
        classes = model.classes_
        colors  = [GRADE_INFO[c]['color'] for c in classes]
        bars = ax.barh(classes, pred_proba, color=colors)
        for bar, prob in zip(bars, pred_proba):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{prob*100:.1f}%', va='center', fontsize=10)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel('Probability')
        ax.set_title('Grade Probability')
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════
# TAB 2: 데이터 탐색
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("데이터 분포 탐색")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**등급별 데이터 개수**")
        counts = df['용해도등급'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors_bar = [GRADE_INFO[g]['color'] for g in counts.index]
        ax.bar(counts.index, counts.values, color=colors_bar)
        for i, v in enumerate(counts.values):
            ax.text(i, v + 30, str(v), ha='center', fontsize=9)
        ax.set_xlabel('Solubility Grade')
        ax.set_ylabel('Count')
        ax.set_title('Data Count by Grade')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown("**등급별 지질친화성(LogP) 분포**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for g in ['G1','G2','G3','G4','G5']:
            subset = df[df['용해도등급'] == g]['지질친화성']
            ax.boxplot(subset, positions=[list(GRADE_INFO.keys()).index(g)],
                       patch_artist=True,
                       boxprops=dict(facecolor=GRADE_INFO[g]['bg'], color=GRADE_INFO[g]['color']))
        ax.set_xticks(range(5))
        ax.set_xticklabels(['G1','G2','G3','G4','G5'])
        ax.set_xlabel('Solubility Grade')
        ax.set_ylabel('LogP')
        ax.set_title('LogP Distribution by Grade')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("**데이터 기술 통계**")
    st.dataframe(df.describe().round(2), use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 3: 모델 성능
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("모델 정확도 비교")

    acc_data = {
        '모델':   ['로지스틱 회귀', '랜덤포레스트', 'KNN', 'SVM'],
        '정확도': [0.7768,         0.7433,         0.7615, 0.7768]
    }
    acc_df = pd.DataFrame(acc_data)

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_colors = ['#4e79a7','#f28e2b','#e15759','#76b7b2']
    bars = ax.bar(acc_df['모델'], acc_df['정확도'], color=bar_colors, width=0.5)
    for bar, acc in zip(bars, acc_df['정확도']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc*100:.2f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("특성 중요도 (랜덤포레스트)")
    feat_names_display = ['분자량','수소결합공여체수','수소결합수용체수','극성표면적','회전가능결합수','지질친화성','고지용성','수소결합총합','고분자량']
    importances = model.feature_importances_
    imp_df = pd.DataFrame({'특성': feat_names_display, '중요도': importances}).sort_values('중요도', ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp_df['특성'], imp_df['중요도'], color='#2874c5')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance (Random Forest)')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    <div class="info-card">
    <b>모델 성능 요약</b><br>
    • 로지스틱 회귀 & SVM: 77.68% — 공동 1위<br>
    • KNN: 76.15% — 이웃 기반 분류<br>
    • 랜덤포레스트: 74.33% — 특성 중요도 제공<br>
    • 핵심 특성: <b>지질친화성(LogP)</b>가 수용성 예측에 가장 큰 영향
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("데이터 출처: AQSolDB (Kaggle) · 11230 최준우")
