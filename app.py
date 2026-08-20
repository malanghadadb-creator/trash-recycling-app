import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(
    page_title="진주시 AI 분리배출 가이드",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ AI 스마트 분리배출 가이드")
st.subheader("진주시 재활용 배출 규정에 맞춘 쓰레기 분류 서비스")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r", encoding="utf-8").readlines()
    return model, class_names

try:
    model, class_names = load_model()
except Exception as e:
    st.error("모델 파일을 불러오지 못했습니다. keras_model.h5와 labels.txt 파일이 저장소에 올바르게 존재하는지 확인해주세요.")
    st.stop()

# 진주시 재활용 배출 가이드 정보
RECYCLING_GUIDE = {
    "플라스틱": {
        "icon": "🥤",
        "method": "내용물을 비우고 물로 헹군 후, 라벨(비닐)을 제거하여 플라스틱 수거함에 배출합니다.",
        "tip": "음료수 병의 비닐 라벨은 반드시 따로 떼어 비닐류로 분리 배출하세요."
    },
    "페트병": {
        "icon": "🍾",
        "method": "투명 페트병은 내용물을 비우고 라벨을 제거한 뒤 찌그러뜨려서 뚜껑을 닫아 전용 수거함에 배출합니다.",
        "tip": "진주시 규정에 따라 투명 페트병은 일반 플라스틱과 별도로 분리배출해야 합니다."
    },
    "종이": {
        "icon": "📦",
        "method": "테이프나 운송장을 제거하고 펼쳐서 상자에 담아 배출합니다.",
        "tip": "음식물이 묻은 종이 상자나 영수증, 비닐 코팅지는 재활용이 불가하므로 종량제 봉투에 버리세요."
    },
    "캔": {
        "icon": "🥫",
        "method": "내용물을 비우고 헹군 뒤 가능한 한 찌그러뜨려서 캔류 수거함에 배출합니다.",
        "tip": "부탄가스나 부탄캔은 구멍을 뚫어 가스를 완전히 뺀 후 배출하세요."
    },
    "유리": {
        "icon": "🍾",
        "method": "내용물을 비우고 헹군 후 유리병 수거함에 배출합니다.",
        "tip": "깨진 유리는 재활용이 되지 않으므로 신문지에 싸서 종량제 봉투나 특수 쓰레기 봉투에 배출하세요."
    },
    "비닐": {
        "icon": "🛍️",
        "method": "이물질을 깨끗이 씻어 말린 후 비닐류 전용 수거함에 배출합니다.",
        "tip": "음식물 등 이물질이 닦이지 않는 비닐은 종량제 봉투에 버려주세요."
    },
    "일반쓰레기": {
        "icon": "🗑️",
        "method": "재활용이 불가능한 쓰레기는 진주시 지정 종량제 봉투에 담아 배출합니다.",
        "tip": "배출 시간을 준수하여 지정된 장소에 놓아주세요."
    }
}

tab1, tab2 = st.tabs(["📷 사진 촬영 및 업로드", "ℹ️ 배출 가이드"])

with tab1:
    img_file_buffer = st.camera_input("분리배출할 쓰레기를 촬영해주세요")
    if img_file_buffer is None:
        img_file_buffer = st.file_uploader("또는 이미지 파일 업로드", type=["jpg", "jpeg", "png"])

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer).convert("RGB")
        st.image(image, caption="입력된 이미지", use_column_width=True)

        # 이미지 전처리
        size = (224, 224)
        image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image_resized)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # 예측
        with st.spinner("AI가 분석 중입니다..."):
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index].strip()
            confidence_score = float(prediction[0][index])

        # 클래스명 정제 (숫자 태그 제거)
        clean_class = class_name.split(' ', 1)[-1] if ' ' in class_name else class_name

        st.success(f"**분석 결과:** {clean_class} (확신도: {confidence_score*100:.1f}%)")

        # 결과 가이드 출력
        matched_key = None
        for key in RECYCLING_GUIDE.keys():
            if key in clean_class:
                matched_key = key
                break

        if matched_key:
            info = RECYCLING_GUIDE[matched_key]
            st.info(f"{info['icon']} **{matched_key} 분리배출 방법**\n\n{info['method']}")
            st.warning(f"💡 **주의사항:** {info['tip']}")
        else:
            st.info("🗑️ **배출 가이드:** 내용물을 깨끗이 비우고 이물질 제거 후 재활용품 또는 종량제 봉투에 배출하세요.")

with tab2:
    st.header("진주시 재활용 품목별 배출 가이드")
    for category, details in RECYCLING_GUIDE.items():
        with st.expander(f"{details['icon']} {category}"):
            st.write(f"**배출 방법:** {details['method']}")
            st.caption(f"💡 {details['tip']}")
