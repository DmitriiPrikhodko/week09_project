import streamlit as st

# --------------------------
# Настройки страницы
# --------------------------
# st.set_page_config(page_title="09-04/05 Детекция и сегментация", layout="wide")

# --------------------------
# Sidebar с навигацией
# --------------------------
# Ссылки на страницы (имена файлов в папке pages)
st.sidebar.markdown("### Перейти к странице:")
st.sidebar.page_link("main.py", label="🏠 Главная")
st.sidebar.page_link("pages/Faces.py", label="👤 Детекция лиц")
st.sidebar.page_link("pages/Tumors.py", label="🧠 Детекция опухолей мозга")
st.sidebar.page_link("pages/Forest.py", label="🌲 Сегментация лесов")
st.set_page_config(
    page_title="09-04/05 Детекция и сегментация",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏠",
)

# --------------------------
# Основной контент главной страницы
# --------------------------
st.title("09-04/05 Детекция и сегментация изображений")

st.markdown(
    """
    <div style="font-size:18px; line-height:1.6;">
    В проекте были реализованы следующие задачи:

    - **Детекция лиц** с помощью YOLO с последующей маскировкой детектированной области  
    - **Детекция опухолей мозга** по фотографии с помощью YOLOv11  
    - **Семантическая сегментация аэрокосмических снимков** с помощью модели U-Net
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

st.subheader("Состав команды")
st.markdown(
    """
    <div style="font-size:20px; line-height:1.6;">

    - Дарья Спренгель  
    - Марат Алекберов  
    - Дмитрий Приходько
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Распределение обязанностей")
st.markdown(
    """
    <div style="font-size:20px; line-height:1.6;">
    Каждый участник проекта обучал одну модель и создавал соответствующую страницу для Streamlit приложения:
    
    - Дарья — детекция лиц (YOLO)  
    - Марат — детекция опухолей мозга (YOLOv11)  
    - Дмитрий — сегментация аэрокосмических снимков (U-Net)
    </div>
    """,
    unsafe_allow_html=True,
)

# st.image(
#     "resources/images/project_overview.jpg",
#     caption="Обзор проекта",
#     use_container_width=True,
# )
