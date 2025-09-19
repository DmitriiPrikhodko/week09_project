# --------------------------
# 1. Настройки и импорты
# --------------------------
import os
from main import *
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import torch
from PIL import Image
import requests
from io import BytesIO
import time
from torchvision import transforms as T
from resources.model_unet.model_unet import UNet
import matplotlib.pyplot as plt
import io

DEVICE = torch.device("cpu")
current_dir = os.path.dirname(__file__)
project_root = os.path.join(current_dir, "..")

st.set_page_config(page_title="Images AI processing", layout="wide")
st.sidebar.markdown("### Перейти к странице:")
st.sidebar.page_link("main.py", label="🏠 Главная")
st.sidebar.page_link("pages/Faces.py", label="👤 Детекция лиц")
st.sidebar.page_link("pages/Tumors.py", label="🧠 Детекция опухолей мозга")
st.sidebar.page_link("pages/Forest.py", label="🌲 Сегментация лесов")
st.title("Распознавание лесов на спутниковых фотографиях")

# --------------------------
# 2. Загрузка модели
# --------------------------
WEIGHTS_DIR = os.path.join(project_root, "resources/model_unet")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "unet_weights_2.pth")


@st.cache_resource
def load_model():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if not os.path.exists(WEIGHTS_PATH):
        public_url = "https://disk.yandex.ru/d/2dijVvxAxH8WNg"
        r = requests.get(
            "https://cloud-api.yandex.net/v1/disk/public/resources/download",
            params={"public_key": public_url},
        )
        r.raise_for_status()
        download_url = r.json()["href"]
        with requests.get(download_url, stream=True) as response:
            response.raise_for_status()
            with open(WEIGHTS_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    model = UNet(n_class=2)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    return model


model = load_model()
model.to(DEVICE)


# --------------------------
# 3. Функция предсказания
# --------------------------
def get_prediction(
    model, img, device="cpu", img_size=(256, 256), overlay_color=(0, 0, 255), alpha=0.4
):
    model.eval()
    with torch.no_grad():
        orig_size = img.size
        transform = T.Compose([T.Resize(img_size), T.ToTensor()])
        img_tensor = transform(img).unsqueeze(0).to(device)

        output = model(img_tensor)
        if output.shape[1] == 1:
            mask = torch.sigmoid(output)
            mask = (mask > 0.5).float()
        else:
            mask = torch.argmax(output, dim=1, keepdim=True).float()
        mask = mask.squeeze(0).squeeze(0).cpu().numpy()

        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(orig_size)
        mask_np = np.array(mask_img) / 255.0

        overlay = np.zeros((orig_size[1], orig_size[0], 3), dtype=np.uint8)
        overlay[..., 0] = overlay_color[0]
        overlay[..., 1] = overlay_color[1]
        overlay[..., 2] = overlay_color[2]

        img_np = np.array(img).astype(np.uint8)
        combined_np = (
            img_np * (1 - alpha * mask_np[..., None])
            + overlay * (alpha * mask_np[..., None])
        ).astype(np.uint8)
        combined_img = Image.fromarray(combined_np)

    return mask_img, combined_img


# --------------------------
# 4. Загрузка изображений от пользователя
# --------------------------
st.subheader("Загрузите изображения или вставьте ссылки")
col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader(
        "📁 Загрузить изображения",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
with col2:
    urls_input = st.text_area(
        "🌐 Вставьте ссылки на изображения (каждую с новой строки)"
    )

images = []
if uploaded_files:
    for f in uploaded_files:
        try:
            img = Image.open(f).convert("RGB")
            images.append(("Файл: " + f.name, img))
        except Exception as e:
            st.warning(f"Не удалось открыть файл {f.name}: {e}")
if urls_input:
    urls = [u.strip() for u in urls_input.split("\n") if u.strip()]
    for idx, url in enumerate(urls):
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            images.append((f"Ссылка {idx+1}", img))
        except Exception as e:
            st.warning(f"Не удалось загрузить изображение по ссылке {url}: {e}")

# --------------------------
# 5. Предсказание и вывод с кнопками скачивания
# --------------------------
if images:
    st.subheader("Результаты сегментации")
    cols_per_row = 4
    for i in range(0, len(images), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (name, image) in enumerate(images[i : i + cols_per_row]):
            with cols[j]:
                start_time = time.time()
                mask_img, combined_img = get_prediction(model, image)
                st.image(
                    [image, combined_img],
                    width=200,
                    caption=[name, "Сегментированное фото"],
                )
                end_time = time.time()
                st.info(f"⏱ {end_time-start_time:.3f} сек")

                # Кнопки скачивания
                buf_mask = io.BytesIO()
                mask_img.save(buf_mask, format="PNG")
                buf_mask.seek(0)
                st.download_button(
                    f"📥 Скачать маску {name}",
                    buf_mask,
                    file_name=f"mask_{name}.png",
                    mime="image/png",
                )

                buf_combined = io.BytesIO()
                combined_img.save(buf_combined, format="PNG")
                buf_combined.seek(0)
                st.download_button(
                    f"📥 Скачать сегментированное фото {name}",
                    buf_combined,
                    file_name=f"segmented_{name}.png",
                    mime="image/png",
                )

# --------------------------
# 6. Описание процесса обучения
# --------------------------
st.markdown("## 🧠 Процесс обучения модели")
st.markdown(
    """
    <div style="font-size:20px; line-height:1.6;">
    Для сегментации изображений использовалась модель 
    <a href="https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3/" target="_blank">Unet</a> 
    с 2 классами (Лес / Остальное)<br><br>

    Для обучения использовался 
    <a href="https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation" target="_blank">датасет</a>
    с аэрокосмическими снимками<br><br>

    - Датасет состоял из <strong>5108</strong> снимков и соответствующих масок<br>
    - Разделение: <strong>80%</strong> на обучение, <strong>20%</strong> на валидацию<br>
    - Обучение: <strong>10 эпох</strong> (~25 минут)<br>
    - Проверка на <strong>20 эпохах</strong> не дала значимого улучшения метрик после 10-й<br><br>

    Ниже приведены финальные метрики и графики их эволюции во время обучения:
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# 7. Чтение CSV и графики Plotly
# --------------------------
METRICS_PATH = os.path.join(project_root, "resources/model_unet/metrics")
metrics_files = {
    "Loss": (
        os.path.join(METRICS_PATH, "train_loss.csv"),
        os.path.join(METRICS_PATH, "valid_loss.csv"),
    ),
    "Accuracy": (
        os.path.join(METRICS_PATH, "train_accuracy.csv"),
        os.path.join(METRICS_PATH, "valid_accuracy.csv"),
    ),
    "Dice": (
        os.path.join(METRICS_PATH, "train_dice.csv"),
        os.path.join(METRICS_PATH, "valid_dice.csv"),
    ),
    "IoU": (
        os.path.join(METRICS_PATH, "train_iou.csv"),
        os.path.join(METRICS_PATH, "valid_iou.csv"),
    ),
}

# Таблица итоговых значений
summary_data = []
for metric_name, (train_file, val_file) in metrics_files.items():
    try:
        df_train = pd.read_csv(train_file)
        df_val = pd.read_csv(val_file)
        last_train = round(df_train["value"].iloc[-1], 3)
        last_val = round(df_val["value"].iloc[-1], 3)
        summary_data.append(
            {"Метрика": metric_name, "Train": last_train, "Valid": last_val}
        )
    except FileNotFoundError:
        summary_data.append({"Метрика": metric_name, "Train": "-", "Valid": "-"})

summary_df = pd.DataFrame(summary_data)


def highlight_columns(x):
    df = x.copy()
    df["Train"] = "background-color: #d0f0c0"
    df["Valid"] = "background-color: #add8e6"
    df["Метрика"] = ""
    return df


styled_df = summary_df.style.apply(highlight_columns, axis=None).format(
    "{:.3f}", subset=["Train", "Valid"]
)
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📊 Итоговые значения метрик (последняя эпоха)")
    st.dataframe(styled_df, use_container_width=True)

# Графики Plotly
col1, col2 = st.columns(2)
for idx, (metric_name, (train_file, val_file)) in enumerate(metrics_files.items()):
    try:
        df_train = pd.read_csv(train_file)
        df_val = pd.read_csv(val_file)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_train["step"],
                y=df_train["value"],
                mode="lines+markers",
                name="train",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_val["step"], y=df_val["value"], mode="lines+markers", name="valid"
            )
        )
        fig.update_layout(
            title=f"{metric_name} по эпохам",
            xaxis_title="Эпоха",
            yaxis_title=metric_name,
            template="plotly_white",
            title_x=0.5,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(title=""),
        )
        if idx % 2 == 0:
            col1.plotly_chart(fig, use_container_width=True)
        else:
            col2.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.warning(
            f"⚠️ Файл {train_file} или {val_file} не найден. Пропускаем график {metric_name}."
        )
