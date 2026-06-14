import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. 페이지 기본 설정
st.set_page_config(page_title="화합물 수용성 예측기", page_icon="🧪", layout="centered")
st.title("🧪 화합물 수용성 등급 예측기")
st.markdown("화합물의 물리화학적 특성을 입력하여 수용성 등급(LogS)을 예측합니다.")
st.divider()

# 2. 모델 및 스케일러 로드 시도
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('scaler.pkl')      
        rf_model = joblib.load('rf_model.pkl')  
        return scaler, rf_model
    except Exception as e:
        # 에러가 나면 콘솔창에 로그를 남겨 추적하기 쉽게 합니다.
        print(f"모델 로드 실패: {e}")
        return None, None

scaler, rf_model = load_models()

# 3. 예시 데이터 세션 상태 초기화 (입력창의 'key' 값과 일치시킵니다)
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

# 예시 데이터 버튼
st.button("💡 예시 데이터 채우기 (아스피린)", on_click=fill_aspirin)

# -----------------------------------------------------------------------------
# 폼 내부 - key와 session_state를 연동하여 버그 제거
# -----------------------------------------------------------------------------
with st.form(key='solubility_form'):
    st.subheader("📝 화합물 정보 입력")
    
    col1, col2 = st.columns(2)
    with col1:
        # value를 직접 지정하는 대신 key를 바로 활용합니다.
        mw = st.number_input("분자량 (Molecular Weight)", min_value=0.0, max_value=2000.0, step=0.01, key='mw_input')
        hd = st.number_input("수소결합 공여체 수 (NumHDonors)", min_value=0, max_value=50, step=1, key='hd_input')
        ha = st.number_input("수소결합 수용체 수 (NumHAcceptors)", min_value=0, max_value=50, step=1, key='ha_input')
    with col2:
        tpsa = st.number_input("위상 극성 표면적 (TPSA)", min_value=0.0, max_value=1000.0, step=0.1, key='tpsa_input')
        nrb = st.number_input("회전 가능한 결합 수 (NumRotatableBonds)", min_value=0, max_value=50, step=1, key='nrb_input')
        logp = st.number_input("지질친화성 (LogP)", min_value=-10.0, max_value=15.0, step=0.01, key='logp_input')

    submit_button = st.form_submit_button(label="🔮 수용성 등급 예측하기", type="primary")

# -----------------------------------------------------------------------------
# 4. 버튼 클릭 후 실행 로직
# -----------------------------------------------------------------------------
if submit_button:
    # 파생 변수 계산
    # ⚠️ 중요: 모델 학습 때 사용한 컬럼명이 영어였다면 이 부분을 영어로 매칭해야 에러가 안 납니다!
    feature_cols_new = ['분자량', '수소결합공여체수', '수소결합수용체수', '극성표면적', '회전가능결합수', '지질친화성', '고지용성', '수소결합총합', '고분자량']
    
    high_logp = int(logp >= 3)
    hbond_total = hd + ha
    high_mw = int(mw >= 500)

    input_data = pd.DataFrame(
        [[mw, hd, ha, tpsa, nrb, logp, high_logp, hbond_total, high_mw]],
        columns=feature_cols_new
    )
    
    class_desc = {
        'G1': '매우 높은 수용성 (LogS 0 이상)', 'G2': '높은 수용성 (LogS -1 ~ 0)',
        'G3': '보통 수용성 (LogS -2 ~ -1)', 'G4': '낮은 수용성 (LogS -3 ~ -2)',
        'G5': '매우 낮은 수용성 (LogS -3 미만)'
    }

    # CASE A: 모델 파일이 정상적으로 로드된 경우
    if rf_model is not None and scaler is not None:
        try:
            input_scaled = scaler.transform(input_data)
            predicted_class = rf_model.predict(input_scaled)[0]
            predicted_proba = rf_model.predict_proba(input_scaled)[0]
            
            st.success("✅ 예측 완료!")
            st.subheader("📊 예측 결과")
            c1, c2 = st.columns(2)
            c1.metric(label="예측된 수용성 등급", value=predicted_class)
            c2.info(f"**등급 설명:** \n{class_desc.get(predicted_class, '알 수 없음')}")
            
            prob_df = pd.DataFrame({'등급': rf_model.classes_, '확률 (%)': predicted_proba * 100})
            st.bar_chart(prob_df.set_index('등급'))
        except Exception as e:
            st.error(f"데이터 변환 또는 예측 중 오류 발생: {e}")
            st.warning("모델의 피처(컬럼) 순서나 이름이 일치하는지 확인하세요.")

    # CASE B: 모델 파일이 없어서 로드에 실패한 경우
    else:
        st.error("❌ 서버에 `scaler.pkl` 또는 `rf_model.pkl` 파일이 보이지 않습니다.")
        st.warning("⚠️ 코드가 정상 작동하는지 확인하기 위해 '테스트용 가상 결과'를 아래에 출력합니다.")
        
        mock_class = "G3"
        st.subheader("📊 예측 결과 (시뮬레이션)")
        c1, c2 = st.columns(2)
        c1.metric(label="예측된 수용성 등급", value=mock_class)
        c2.info(f"**등급 설명:** 보통 수용성 (LogS -2 ~ -1)")
        
        prob_df = pd.DataFrame({'등급': ['G1', 'G2', 'G3', 'G4', 'G5'], '확률 (%)': [5.0, 20.0, 60.0, 10.0, 5.0]})
        st.bar_chart(prob_df.set_index('등급'))
