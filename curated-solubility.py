import streamlit as st
import pandas as pd
import joblib
import os

# 1. 웹 페이지 기본 레이아웃 설정
st.set_page_config(
    page_title="화합물 수용성 용해도 등급 예측",
    page_icon="🧪",
    layout="centered"
)

# 2. 타이틀 및 설명
st.title("🧪 머신러닝 기반 화합물 용해도 등급 예측 시스템")
st.markdown("""
이 애플리케이션은 유기 화합물의 물리화학적 특성을 입력받아 **수용성 용해도 등급(Group)**을 실시간으로 예측합니다.
왼쪽 사이드바 또는 아래 입력창에 예측하고 싶은 화합물의 특성 값을 입력해 보세요.
""")

# 3. 모델 파일 로드 함수 (캐싱 처리하여 속도 향상)
@st.cache_resource
def load_model():
    model_path = "curated-solubility-dataset.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"⚠️ 모델 파일('{model_path}')을 찾을 수 없습니다. 경로를 확인해 주세요.")
        return None

rf_model = load_model()

# 4. 사용자 입력 받기 (메인 화면 또는 사이드바)
st.subheader("📋 화합물 특성 정보 입력")

col1, col2 = st.columns(2)

with col1:
    mol_wt = st.number_input("🔹 분자량 (Molecular Weight)", min_value=0.0, max_value=6000.0, value=266.6, step=0.1, help="화합물의 분자량입니다.")
    h_donors = st.number_input("🔹 수소결합 공여체 수 (Num H Donors)", min_value=0, max_value=30, value=1, step=1)
    h_acceptors = st.number_input("🔹 수소결합 수용체 수 (Num H Acceptors)", min_value=0, max_value=90, value=3, step=1)

with col2:
    tpsa = st.number_input("🔹 극성 표면적 (TPSA)", min_value=0.0, max_value=1300.0, value=62.4, step=0.1, help="Topological Polar Surface Area")
    rot_bonds = st.number_input("🔹 회전 가능 결합 수 (Rotatable Bonds)", min_value=0, max_value=150, value=4, step=1)
    log_p = st.number_input("🔹 지질친화성 (MolLogP)", min_value=-45.0, max_value=70.0, value=1.9, step=0.1, help="Octanol-water partition coefficient")

# 5. 예측 실행 버튼
if st.button("🚀 용해도 등급 예측하기", use_container_width=True):
    if rf_model is not None:
        # 노트북의 feature_names와 완벽히 일치하는 DataFrame 생성 (Warning 방지)
        input_data = pd.DataFrame([{
            '분자량': mol_wt,
            '수소결합공여체수': float(h_donors),
            '수소결합수용체수': float(h_acceptors),
            '극성표면적': tpsa,
            '회전가능결합수': float(rot_bonds),
            '지질친화성': log_p
        }])
        
        # 모델 예측
        prediction = rf_model.predict(input_data)[0]
        
        # 확률 계산 (가능한 경우)
        try:
            pred_proba = rf_model.predict_proba(input_data)
            max_proba = max(pred_proba[0]) * 100
        except:
            max_proba = None

        # 결과 출력 레이아웃
        st.success("### 🎉 예측 결과 완료!")
        
        # 큰 글씨로 등급 강조
        st.metric(label="예측된 용해도 등급", value=f"{prediction}")
        
        if max_proba is not None:
            st.info(f"💡 모델의 예측 신뢰도(확률): **{max_proba:.2f}%**")
            
        # 데이터 시각적 확인
        st.markdown("**입력된 데이터 확인:**")
        st.dataframe(input_data)
        
    else:
        st.warning("모델 파일 로드에 실패하여 예측을 진행할 수 없습니다.")
