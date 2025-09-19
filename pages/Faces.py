import pandas as pd
import numpy as np
import streamlit as st

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

import os
import requests
from PIL import Image, ImageDraw, ImageFilter
from io import BytesIO

from ultralytics import YOLO
#import cv2


# Название
st.title("Детекция лиц на фотографиях")

st.sidebar.markdown("### Перейти к странице:")
st.sidebar.page_link("main.py", label="🏠 Главная")
st.sidebar.page_link("pages/Faces.py", label="👤 Детекция лиц")
st.sidebar.page_link("pages/Tumors.py", label="🧠 Детекция опухолей мозга")
st.sidebar.page_link("pages/Forest.py", label="🌲 Сегментация лесов")

# 0. Загрузка предобученной модели
model_path = 'face_detection_model/best.pt'
if os.path.exists(model_path):
    try:
        model = YOLO(model_path)
        model_loaded = True
    except Exception as e:
        st.error(f'Ошибка загрузки модели: {e}')
        model_loaded = False
else:
    st.error(f'Файл модели не найден: {model_path}')
    st.info("Пожалуйста, убедитесь что файл best.pt расположен в папке с проектом")
    model_loaded = False

# 1. Загрузка файлов
col1, col2 = st.columns(2)
uploaded_files = []

with col1:
    uploaded_files_local = st.file_uploader('Загрузите изображения', type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    if uploaded_files_local:
        uploaded_files.extend(uploaded_files_local)
with col2:
    url_input = st.text_input("Или введите URL изображения")
    if url_input:
        try:
            response = requests.get(url_input)
            img = Image.open(BytesIO(response.content))
            uploaded_files.append(img)
        except:
            st.error('Ошибка загрузки по URL')

# 3. Запуск детекции
DEVICE = 'cpu'
# Функция размытия детектированных областей
def blur_detections(img, results): # type: ignore
    draw = ImageDraw.Draw(img)
    # Обрабатываем только первый элемент списка results (т.к. у вас по одному изображению)
    detections = results[0].boxes.xyxy.cpu().numpy()  # Получить numpy массив координат боксов
    for box in detections:
        x1, y1, x2, y2 = map(int, box)
        region = img.crop((x1, y1, x2, y2))
        blurred = region.filter(ImageFilter.GaussianBlur(15))
        img.paste(blurred, (x1, y1))
    return img


import io
from streamlit.runtime.uploaded_file_manager import UploadedFile  # импорт типа файлов

if model_loaded and uploaded_files:
    if st.button('Запустить детекцию', type='primary'):
        st.header('Результаты детекции:')
        for i, img_file in enumerate(uploaded_files):
            model.to(DEVICE)
            
            try:
                # Проверяем, файл ли это из загрузчика или уже PIL.Image
                if isinstance(img_file, UploadedFile):
                    # Если UploadedFile, читаем байты
                    file_bytes = img_file.read()
                    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
                elif isinstance(img_file, Image.Image):
                    img = img_file.convert('RGB')
                else:
                    st.error(f"Неподдерживаемый тип файла: {type(img_file)}")
                    continue
            except Exception as e:
                st.error(f"Ошибка открытия файла: {e}")
                continue

            results = model(img)
            img_blurred = blur_detections(img.copy(), results)
            st.image(img_blurred, caption=f"Результат изображения {i+1}")

# 2. Информация о модели
st.header('Информация о модели')
model_metrics = pd.read_csv('face_detection_model/results.csv')
last_epoch = model_metrics.iloc[-1]

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.metric('Число эпох', '5')
    st.metric('Объем train выборки', '13386 фото')
    st.metric('Объем valid выборки', '3347 фото')

with info_col2:
    st.metric('mAP50', '0.885')
    st.metric('Precision', '0.883')
    st.metric('Recall', '0.809')

fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=['Precision', 'Recall', 'mAP50', 'mAP50-95'],
    y=[last_epoch['metrics/precision(B)'], 
    last_epoch['metrics/recall(B)'], 
    last_epoch['metrics/mAP50(B)'], 
    last_epoch['metrics/mAP50-95(B)']],
    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
))

fig1.update_layout(
    title='Основные метрики',
    yaxis=dict(range=[0, 1]),
    showlegend=False
)
st.plotly_chart(fig1)

# 3. Запуск детекции
DEVICE = 'cpu'
# Функция размытия детектированных областей
def blur_detections(img, results):
    draw = ImageDraw.Draw(img)
    # Обрабатываем только первый элемент списка results (т.к. у вас по одному изображению)
    detections = results[0].boxes.xyxy.cpu().numpy()  # Получить numpy массив координат боксов
    for box in detections:
        x1, y1, x2, y2 = map(int, box)
        region = img.crop((x1, y1, x2, y2))
        blurred = region.filter(ImageFilter.GaussianBlur(15))
        img.paste(blurred, (x1, y1))
    return img


import io
from streamlit.runtime.uploaded_file_manager import UploadedFile  # импорт типа файлов

if model_loaded and uploaded_files:
    if st.button('Запустить детекцию', type='primary'):
        st.header('Результаты детекции:')
        for i, img_file in enumerate(uploaded_files):
            model.to(DEVICE)
            
            try:
                # Проверяем, файл ли это из загрузчика или уже PIL.Image
                if isinstance(img_file, UploadedFile):
                    # Если UploadedFile, читаем байты
                    file_bytes = img_file.read()
                    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
                elif isinstance(img_file, Image.Image):
                    img = img_file.convert('RGB')
                else:
                    st.error(f"Неподдерживаемый тип файла: {type(img_file)}")
                    continue
            except Exception as e:
                st.error(f"Ошибка открытия файла: {e}")
                continue

            results = model(img)
            img_blurred = blur_detections(img.copy(), results)
            st.image(img_blurred, caption=f"Результат изображения {i+1}")