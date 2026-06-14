import streamlit as st
import pandas as pd
import numpy as np

# 1. 페이지 기본 설정
st.set_page_config(page_title="화합물 수용성 예측기", page_icon="🧪", layout="centered")
st.title("🧪 화합물 수용성 등급 예측기 (시뮬레이션)")
st.markdown("화합물의 물리화학적 특성을 입력하여 수용성 등급(LogS)을 예측합니다. (AI 가상 예측 엔진 작동 중)")
st.divider()

# 2. 예시 데이터 세션 상태 초기화
if 'mw_input' not in st.session_state: st.session_state['mw_input'] = 180.16
if 'hd_input' not in st.session_state: st.session_state['hd_input'] = 1
if 'ha_input' not in st.session_state: st.session_state['ha_input'] = 4
if 'tpsa_input' not in st.session_state: st.session_state['tpsa_input'] = 63.6
if 'nrb_input' not in st.session_state: st.session_state['nrb_input'] = 3
if 'logp_input' not in st.session_state: st.session_state['logp_input'] = 1.19

def fill_aspirin():
    st.session_state['mw_input'] = 180.16
    st.session_state['hd_input'] = 1
    st.session_state['ha_input'] = 4
    st.session_state['tpsa_input'] = 63.6
    st.session_state['nrb_input'] = 3
    st.session_state['logp_input'] = 1.19

st.button("💡 예시 데이터 채우기 (아스피린)", on_click=fill_aspirin)

# -----------------------------------------------------------------------------
# 3. 입력 폼 (st.form으로 묶어 데이터 흐름 보장)
# -----------------------------------------------------------------------------
with st.form(key='solubility_form'):
    st.subheader("📝 화합물 정보 입력")
    
    col1, col2 = st.columns(2)
    with col1:
        mw = st.number_input("분자량 (Molecular Weight)", min_value=0.0, max_value=2000.0, step=0.01, key='mw_input')
        hd = st.number_input("수소결합 공여체 수 (NumHDonors)", min_value=0, max_value=50, step=1, key='hd_input')
        ha = st.number_input("수소결합 수용체 수 (NumHAcceptors)", min_value=0, max_value=50, step=1, key='ha_input')
    with col2:
        tpsa = st.number_input("위상 극성 표면적 (TPSA)", min_value=0.0, max_value=1000.0, step=0.1, key='tpsa_input')
        nrb = st.number_input("회전 가능한 결합 수 (NumRotatableBonds)", min_value=0, max_value=50, step=1, key='nrb_input')
        logp = st.number_input("지질친화성 (LogP)", min_value=-10.0, max_value=15.0, step=0.01, key='logp_input')

    submit_button = st.form_submit_button(label="🔮 수용성 등급 예측하기", type="primary")

# -----------------------------------------------------------------------------
# 4. 버튼 클릭 후 가상 예측 실행 로직 (야메 엔진)
# -----------------------------------------------------------------------------
if submit_button:
    
    # 🧪 화학적 규칙 기반 수용성 점수 계산 (가상 알고리즘)
    # 수용성에 가장 큰 영향을 주는 것은 지질친화성(LogP)과 분자량(MW), 그리고 친수성기(TPSA, 수소결합)입니다.
    
    # 기본 점수 50점에서 시작 (0 ~ 100점 스케일)
    score = 50.0
    
    # 1. LogP가 높을수록 기름과 친하므로 수용성은 떨어짐 (패널티)
    score -= logp * 15.0
    
    # 2. 분자량이 너무 크면 수용성이 떨어짐
    if mw > 500:
        score -= (mw - 500) * 0.05
    elif mw < 200:
        score += (200 - mw) * 0.05
        
    # 3. 극성 표면적(TPSA)과 수소결합 수가 많을수록 물에 잘 녹음 (보너스)
    score += (tpsa * 0.1)
    score += (hd + ha) * 2.0
    
    # 점수를 기반으로 등급 판정 (G1: 매우 좋음 ~ G5: 매우 나쁨)
    if score >= 70:
        predicted_class = "G1"
        proba_base = [65.0, 25.0, 7.0, 2.0, 1.0]
    elif score >= 45:
        predicted_class = "G2"
        proba_base = [20.0, 55.0, 18.0, 5.0, 2.0]
    elif score >= 20:
        predicted_class = "G3"
        proba_base = [5.0, 20.0, 60.0, 10.0, 5.0]
    elif score >= -10:
        predicted_class = "G4"
        proba_base = [2.0, 5.0, 18.0, 60.0, 15.0]
    else:
        predicted_class = "G5"
        proba_base = [1.0, 2.0, 5.0, 22.0, 70.0]

    # 등급 설명 매핑
    class_desc = {
        'G1': '매우 높은 수용성 (LogS 0 이상)', 
        'G2': '높은 수용성 (LogS -1 ~ 0)',
        'G3': '보통 수용성 (LogS -2 ~ -1)', 
        'G4': '낮은 수용성 (LogS -3 ~ -2)',
        'G5': '매우 낮은 수용성 (LogS -3 미만)'
    }

    # 결과 화면 출력
    st.success("✅ 가상 AI 엔진 예측 완료!")
    st.subheader("📊 예측 결과")
    
    c1, c2 = st.columns(2)
    c1.metric(label="예측된 수용성 등급", value=predicted_class)
    c2.info(f"**등급 설명:** \n{class_desc.get(predicted_class, '알 수 없음')}")
    
    # 예측 확률 데이터프레임 생성 및 시각화
    prob_df = pd.DataFrame({
        '등급': ['G1', 'G2', 'G3', 'G4', 'G5'], 
        '확률 (%)': proba_base
    })
    st.bar_chart(prob_df.set_index('등급'))
