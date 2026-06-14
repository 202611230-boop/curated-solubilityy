import streamlit as st
import pandas as pd
import numpy as np
import joblib  # 모델 파일을 불러오기 위해 필요합니다.

st.set_page_config(page_title="화합물 수용성 예측기", page_icon="🧪", layout="centered")
st.title("🧪 화합물 수용성 등급 예측기")
st.markdown("화합물의 물리화학적 특성을 입력하여 수용성 등급(LogS)을 예측합니다.")
st.divider()

# -----------------------------------------------------------------------------
# [중요] 모델 및 스케일러 파일 로드 구문
# -----------------------------------------------------------------------------
# 본인이 저장한 모델 및 스케일러 파일명으로 변경하세요. (예: 'my_rf_model.pkl')
@st.cache_resource  # 모델을 매번 새로 로드하지 않고 캐싱하여 속도를 높입니다.
def load_models():
    try:
        scaler = joblib.load('scaler.pkl')      # 스케일러 파일명 확인!
        rf_model = joblib.load('rf_model.pkl')  # 모델 파일명 확인!
        return scaler, rf_model
    except Exception as e:
        # 파일이 없을 경우를 대비한 예외 처리
        return None, None

scaler, rf_model = load_models()

# -----------------------------------------------------------------------------
# 화합물 정보 입력 화면 구성
# -----------------------------------------------------------------------------
st.subheader("📝 화합물 정보 입력")

# 예시 데이터 채우기 콜백 함수
def load_aspirin_data():
    st.session_state['mw'] = 180.16
    st.session_state['hd'] = 1
    st.session_state['ha'] = 4
    st.session_state['tpsa'] = 63.6
    st.session_state['nrb'] = 3
    st.session_state['logp'] = 1.19

st.button("💡 예시 데이터 채우기 (아스피린)", on_click=load_aspirin_data)

col1, col2 = st.columns(2)

with col1:
    mw = st.number_input("분자량 (Molecular Weight)", min_value=0.0, max_value=2000.0, step=0.01, key='mw', value=st.session_state.get('mw', 180.16))
    hd = st.number_input("수소결합 공여체 수 (NumHDonors)", min_value=0, max_value=50, step=1, key='hd', value=st.session_state.get('hd', 1))
    ha = st.number_input("수소결합 수용체 수 (NumHAcceptors)", min_value=0, max_value=50, step=1, key='ha', value=st.session_state.get('ha', 4))

with col2:
    tpsa = st.number_input("위상 극성 표면적 (TPSA)", min_value=0.0, max_value=1000.0, step=0.1, key='tpsa', value=st.session_state.get('tpsa', 63.6))
    nrb = st.number_input("회전 가능한 결합 수 (NumRotatableBonds)", min_value=0, max_value=50, step=1, key='nrb', value=st.session_state.get('nrb', 3))
    logp = st.number_input("지질친화성 (LogP)", min_value=-10.0, max_value=15.0, step=0.01, key='logp', value=st.session_state.get('logp', 1.19))

# 파생 변수 계산
feature_cols_new = ['분자량', '수소결합공여체수', '수소결합수용체수', '극성표면적', '회전가능결합수', '지질친화성', '고지용성', '수소결합총합', '고분자량']
high_logp = int(logp >= 3)
hbond_total = hd + ha
high_mw = int(mw >= 500)

input_data = pd.DataFrame(
    [[mw, hd, ha, tpsa, nrb, logp, high_logp, hbond_total, high_mw]],
    columns=feature_cols_new
)

st.divider()

# -----------------------------------------------------------------------------
# 🚨 핵심: "수용성 등급 예측하기" 버튼 클릭 로직 수정
# -----------------------------------------------------------------------------
if st.button("🔮 수용성 등급 예측하기", type="primary"):
    
    # 등급 설명 딕셔너리 정의
    class_desc = {
        'G1': '매우 높은 수용성 (LogS 0 이상)',
        'G2': '높은 수용성 (LogS -1 ~ 0)',
        'G3': '보통 수용성 (LogS -2 ~ -1)',
        'G4': '낮은
