import streamlit as st
import pandas as pd
import numpy as np

# 임시 방편: 기존 코드의 scaler와 model이 로드되어 있다고 가정합니다.
# 실제 실행 시에는 이 부분에 학습된 모델 파일(.pkl 등)을 로드하는 코드가 들어가야 합니다.
# 예: 
# import joblib
# scaler = joblib.load('scaler.pkl')
# rf_model = joblib.load('rf_model.pkl')

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 제목
# -----------------------------------------------------------------------------
st.set_page_config(page_title="화합물 수용성 예측기", page_icon="🧪", layout="centered")
st.title("🧪 화합물 수용성 등급 예측기")
st.markdown("화합물의 물리화학적 특성을 입력하여 수용성 등급(LogS)을 예측합니다.")
st.divider()

# -----------------------------------------------------------------------------
# 2. 사이드바 또는 메인 화면에 입력 폼 구성
# -----------------------------------------------------------------------------
st.subheader("📝 화합물 정보 입력")

# 예시 데이터 (아스피린) 버튼 제공 (사용자 편의용)
if st.button("💡 예시 데이터 채우기 (아스피린)"):
    st.session_state.mw = 180.16
    st.session_state.hd = 1
    st.session_state.ha = 4
    st.session_state.tpsa = 63.6
    st.session_state.nrb = 3
    st.session_state.logp = 1.19

# 두 열로 나누어 입력창 배치
col1, col2 = st.columns(2)

with col1:
    mw = st.number_input("분자량 (Molecular Weight)", min_value=0.0, max_value=2000.0, value=st.session_state.get('mw', 180.16), step=0.01, key='input_mw')
    hd = st.number_input("수소결합 공여체 수 (NumHDonors)", min_value=0, max_value=50, value=st.session_state.get('hd', 1), step=1, key='input_hd')
    ha = st.number_input("수소결합 수용체 수 (NumHAcceptors)", min_value=0, max_value=50, value=st.session_state.get('ha', 4), step=1, key='input_ha')

with col2:
    tpsa = st.number_input("위상 극성 표면적 (TPSA)", min_value=0.0, max_value=1000.0, value=st.session_state.get('tpsa', 63.6), step=0.1, key='input_tpsa')
    nrb = st.number_input("회전 가능한 결합 수 (NumRotatableBonds)", min_value=0, max_value=50, value=st.session_state.get('nrb', 3), step=1, key='input_nrb')
    logp = st.number_input("지질친화성 (LogP)", min_value=-10.0, max_value=15.0, value=st.session_state.get('logp', 1.19), step=0.01, key='input_logp')

# -----------------------------------------------------------------------------
# 3. 파생 변수 계산 및 DataFrame 생성
# -----------------------------------------------------------------------------
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
# 4. 예측 실행 및 결과 출력
# -----------------------------------------------------------------------------
if st.button("🔮 수용성 등급 예측하기", type="primary"):
    
    # 예외 처리: 실제 모델과 스케일러가 로드되어 있는지 확인
    if 'rf_model' in globals() and 'scaler' in globals():
        
        # 스케일링 및 예측
        input_scaled = scaler.transform(input_data)
        predicted_class = rf_model.predict(input_scaled)[0]
        predicted_proba = rf_model.predict_proba(input_scaled)[0]
        
        # 등급 설명 딕셔너리
        class_desc = {
            'G1': '매우 높은 수용성 (LogS 0 이상)',
            'G2': '높은 수용성 (LogS -1 ~ 0)',
            'G3': '보통 수용성 (LogS -2 ~ -1)',
            'G4': '낮은 수용성 (LogS -3 ~ -2)',
            'G5': '매우 낮은 수용성 (LogS -3 미만)'
        }
        
        st.subheader("📊 예측 결과")
        
        # 결과 메트
