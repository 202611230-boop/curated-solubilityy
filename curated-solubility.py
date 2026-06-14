import pandas as pd
import streamlit as st
import joblib
import os

# ─────────────────────────────────────────────
# 1. 페이지 설정 (반드시 최상단에 위치해야 함)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="신약 후보 물질 수용성 예측기",
    page_icon="🧪",
    layout="centered"
)

# ─────────────────────────────────────────────
# 2. 모델 로드 함수
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    """저장된 모델, 스케일러, 특성 이름을 불러옵니다."""
    model_path  = "rf_model.pkl"
    scaler_path = "scaler.pkl"

    missing = [p for p in [model_path, scaler_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {', '.join(missing)}")

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 학습 시 사용했던 특성 이름 (순서 엄수)
    feature_cols = [
        "MolecularWeight",
        "NumHDonors",
        "NumHAcceptors",
        "TPSA",
        "NumRotatableBonds",
        "LogP",
        "high_logp",
        "hbond_total",
        "high_mw",
    ]
    return model, scaler, feature_cols


# ─────────────────────────────────────────────
# 3. 모델 로드 실행
# ─────────────────────────────────────────────
try:
    rf_model, scaler, feature_cols = load_models()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    load_error   = str(e)

# ─────────────────────────────────────────────
# 4. UI — 제목 및 설명
# ─────────────────────────────────────────────
st.title("🧪 신약 후보 물질 수용성 등급 예측기")
st.markdown(
    """
    분자의 구조적 정보와 물성 데이터를 입력하면,  
    **Random Forest** 모델이 수용성 등급을 예측합니다.
    """
)
st.divider()

if not model_loaded:
    st.error(
        f"🚨 모델 파일 로드 실패\n\n"
        f"{load_error}\n\n"
        f"`rf_model.pkl` 과 `scaler.pkl` 파일이 "
        f"`app.py` 와 **같은 폴더**에 있는지 확인해 주세요."
    )
    st.stop()

# ─────────────────────────────────────────────
# 5. 입력 폼
# ─────────────────────────────────────────────
st.subheader("📝 화합물 정보 입력")

col1, col2 = st.columns(2)

with col1:
    mw  = st.number_input("분자량 (MolecularWeight)",              min_value=0.0, value=180.16, step=0.01,  format="%.2f")
    hd  = st.number_input("수소결합 공여체 수 (NumHDonors)",        min_value=0,   value=1,      step=1)
    ha  = st.number_input("수소결합 수용체 수 (NumHAcceptors)",     min_value=0,   value=4,      step=1)

with col2:
    logp = st.number_input("분배계수 (LogP)",                       value=1.19,   step=0.01,   format="%.2f")
    tpsa = st.number_input("위상 극성 표면적 (TPSA)",               min_value=0.0, value=63.6,  step=0.1,   format="%.1f")
    nrb  = st.number_input("회전 가능한 결합 수 (NumRotatableBonds)", min_value=0,  value=3,     step=1)

# ─────────────────────────────────────────────
# 6. 파생 변수 자동 계산 (사이드바에 표시)
# ─────────────────────────────────────────────
high_logp    = int(logp >= 3)
hbond_total  = int(hd + ha)
high_mw      = int(mw >= 500)

with st.sidebar:
    st.header("🔧 자동 계산된 파생 변수")
    st.metric("high_logp  (LogP ≥ 3)",   high_logp)
    st.metric("hbond_total (HD + HA)",   hbond_total)
    st.metric("high_mw  (MW ≥ 500)",     high_mw)

# ─────────────────────────────────────────────
# 7. 등급 설명 딕셔너리
# ─────────────────────────────────────────────
class_desc = {
    "G1": "매우 높은 수용성 (LogS ≥ 0)",
    "G2": "높은 수용성    (LogS −1 ~ 0)",
    "G3": "보통 수용성    (LogS −2 ~ −1)",
    "G4": "낮은 수용성    (LogS −3 ~ −2)",
    "G5": "매우 낮은 수용성 (LogS < −3)",
}

# 등급별 색상 (Streamlit color hex)
grade_color = {
    "G1": "🟢", "G2": "🟩",
    "G3": "🟡", "G4": "🟠", "G5": "🔴",
}

st.divider()

# ─────────────────────────────────────────────
# 8. 예측 버튼
# ─────────────────────────────────────────────
if st.button("🔮 수용성 등급 예측하기", type="primary", use_container_width=True):

    # 입력 데이터프레임 구성
    input_df = pd.DataFrame(
        [[mw, hd, ha, tpsa, nrb, logp, high_logp, hbond_total, high_mw]],
        columns=feature_cols,
    )

    try:
        # 스케일러 적용
        input_scaled = scaler.transform(input_df)

        # 예측
        prediction   = rf_model.predict(input_scaled)[0]
        probabilities = None

        # 확률 값이 있으면 가져오기
        if hasattr(rf_model, "predict_proba"):
            proba        = rf_model.predict_proba(input_scaled)[0]
            classes      = rf_model.classes_
            probabilities = dict(zip(classes, proba))

        # 결과 출력
        icon = grade_color.get(str(prediction), "🔵")
        st.success(f"### {icon} 예측 등급: **{prediction}**")

        desc = class_desc.get(str(prediction))
        if desc:
            st.info(f"💡 {desc}")
        else:
            st.warning("정의되지 않은 등급이 반환되었습니다.")

        # 확률 바 차트
        if probabilities:
            st.subheader("📊 등급별 예측 확률")
            prob_df = (
                pd.DataFrame.from_dict(
                    probabilities, orient="index", columns=["확률"]
                )
                .sort_index()
            )
            st.bar_chart(prob_df)

        # 입력값 요약 테이블
        with st.expander("📋 입력 데이터 확인"):
            st.dataframe(input_df.T.rename(columns={0: "입력값"}), use_container_width=True)

    except Exception as e:
        st.error(f"❌ 예측 중 오류가 발생했습니다: {e}")
