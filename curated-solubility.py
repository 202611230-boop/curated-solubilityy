import pandas as pd
import streamlit as st
import joblib  # 모델과 스케일러를 불러오기 위한 패키지

# 1. 저장된 모델, 스케일러, 특성 이름 불러오기
# @st.cache_resource를 사용하면 앱이 실행될 때 모델을 한 번만 로드하여 속도가 빨라집니다.
@st.cache_resource
def load_models():
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # 학습에 사용했던 정확한 특성 이름 리스트 (순서 주의)
    feature_cols = [
        'MolecularWeight', 'NumHDonors', 'NumHAcceptors', 
        'TPSA', 'NumRotatableBonds', 'LogP', 
        'high_logp', 'hbond_total', 'high_mw'
    ]
    return model, scaler, feature_cols

try:
    rf_model, scaler, feature_cols_new = load_models()
except FileNotFoundError:
    st.error("🚨 'rf_model.pkl' 또는 'scaler.pkl' 파일을 찾을 수 없습니다. 모델 파일이 스크립트와 같은 경로에 있는지 확인해주세요.")
    st.stop()

# 2. 앱 제목 및 설명
st.title("🧪 신약 후보 물질 수용성 등급 예측기")
st.markdown("""
분자의 구조적 정보와 물성 데이터를 입력하면, **joblib**으로 불러온 Random Forest 모델을 통해 수용성 등급을 예측합니다.
""")

st.divider()

# 3. 화합물 정보 입력 폼
st.subheader("📝 화합물 정보 입력")

col1, col2 = st.columns(2)

with col1:
    mw = st.number_input("분자량 (MolecularWeight)", min_value=0.0, value=180.16, step=0.01)
    hd = st.number_input("수소결합 공여체 수 (NumHDonors)", min_value=0, value=1, step=1)
    ha = st.number_input("수소결합 수용체 수 (NumHAcceptors)", min_value=0, value=4, step=1)

with col2:
    logp = st.number_input("분배계수 (LogP)", value=1.19, step=0.01)
    tpsa = st.number_input("위상 극성 표면적 (TPSA)", min_value=0.0, value=63.6, step=0.1)
    nrb = st.number_input("회전 가능한 결합 수 (NumRotatableBonds)", min_value=0, value=3, step=1)

# 4. 파생 변수 자동 계산
high_logp = int(logp >= 3)
hbond_total = hd + ha
high_mw = int(mw >= 500)

# 5. 등급 설명 딕셔너리
class_desc = {
    'G1': '매우 높은 수용성 (LogS 0 이상)',
    'G2': '높은 수용성 (LogS -1 ~ 0)',
    'G3': '보통 수용성 (LogS -2 ~ -1)',
    'G4': '낮은 수용성 (LogS -3 ~ -2)',
    'G5': '매우 낮은 수용성 (LogS -3 미만)'
}

st.divider()

# 6. 예측 버튼 및 결과 출력
if st.button("🔮 수용성 등급 예측하기", type="primary"):
    
    # [수정 포인트]: 변수명을 feature_cols_new로 변경하고 괄호를 정확히 닫아줌
    input_data = pd.DataFrame(
        [[mw, hd, ha, tpsa, nrb, logp, high_logp, hbond_total, high_mw]],
        columns=feature_cols_new
    )
    
    # 7. 데이터 전처리 및 예측 진행 (추가된 기능)
    try:
        # 학습할 때 사용한 스케일러로 데이터 변환
        input_scaled = scaler.transform(input_data)
        
        # 모델 예측 값 추출 (예: 'G1', 'G2' 등)
        prediction = rf_model.predict(input_scaled)[0]
        
        # 결과 화면에 출력
        st.success(f"### 🎉 예측 결과: **{prediction}**")
        if prediction in class_desc:
            st.info(f"💡 **등급 설명:** {class_desc[prediction]}")
        else:
            st.warning("정의되지 않은 등급 결과가 반환되었습니다.")
            
    except Exception as e:
        st.error(f"예측 과정 중 오류가 발생했습니다: {e}")
