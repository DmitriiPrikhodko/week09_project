# app.py — Streamlit UI для YOLOv12 (инференс + валидация)
# — качает веса/датасет по ссылкам (в т.ч. Яндекс.Диск) с кэшем,
# — интерактивные графики (PR и P/R/F1/mAP50 vs conf) с hover.

import os, glob, hashlib, tarfile, zipfile, requests, yaml, shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import torch

# -------------------- Настройки страницы --------------------
st.set_page_config(page_title="YOLOv12 Brain Tumor", layout="wide")

# ===================== A. ССЫЛКИ (ЗАПОЛНИ СВОИ) =====================
WEIGHTS_SRC = (
    "https://disk.yandex.ru/d/EwMSpkabdWAnMQ"  # URL на best.pt (публичная ссылка)
)
DATASET_SRC = "https://disk.yandex.ru/d/c1yAhhMy9sEwuA"  # URL на merged_yolo.zip (архив с images/labels/data.yaml)

# ===================== B. Хелперы: секреты / Я.Диск / загрузка / датасет =====================

YA_PUBLIC_API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"


def get_secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


@st.cache_resource(show_spinner=False)
def yadisk_direct_href(public_url: str, path: Optional[str] = None) -> str:
    params = {"public_key": public_url}
    if path:
        params["path"] = path
    headers = {}
    token = get_secret("YADISK_TOKEN")
    if token:
        headers["Authorization"] = f"OAuth {token}"
    r = requests.get(YA_PUBLIC_API, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    href = r.json().get("href")
    if not href:
        raise RuntimeError(f"No download href for: {public_url}")
    return href


@st.cache_resource(show_spinner=False)
def download_to_cache(
    url_or_path: str,
    cache_subdir: str,
    suggested_name: Optional[str] = None,
    force: bool = False,
) -> Path:
    p = Path(url_or_path)
    if p.exists():
        return p.resolve()

    cache_dir = Path.home() / ".cache" / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = suggested_name or url_or_path.split("?")[0].split("/")[-1] or "file.bin"
    h = hashlib.md5(url_or_path.encode()).hexdigest()[:10]
    out = cache_dir / f"{h}_{base}"

    if force and out.exists():
        try:
            out.unlink()
        except Exception:
            pass

    if out.exists():
        return out

    url = (
        yadisk_direct_href(url_or_path) if "disk.yandex" in url_or_path else url_or_path
    )
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
    return out


@st.cache_resource(show_spinner=True)
def fetch_dataset(url_or_dir: str, force: bool = False) -> Path:
    p = Path(url_or_dir)
    if p.exists() and p.is_dir():
        return p.resolve()

    # YAML по ссылке — возьмём его директорию
    if isinstance(url_or_dir, str) and url_or_dir.lower().endswith((".yaml", ".yml")):
        local_yaml = download_to_cache(
            url_or_dir, "yolo_dataset", "data.yaml", force=force
        )
        return local_yaml.parent.resolve()

    # Архив → скачать и распаковать
    arc = (
        p
        if p.exists() and p.is_file()
        else download_to_cache(url_or_dir, "yolo_dataset", force=force)
    )
    dst = arc.with_suffix("").parent / (arc.stem + "_extracted")
    if force and dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    if not dst.exists():
        dst.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(arc):
            import zipfile as _zip

            with _zip.ZipFile(arc) as z:
                z.extractall(dst)
        elif tarfile.is_tarfile(arc):
            with tarfile.open(arc, "r:*") as t:
                t.extractall(dst)
        else:
            raise ValueError(f"Unknown archive format: {arc}")
    return dst.resolve()


def find_and_fix_data_yaml(root: Path) -> Path:
    if (root / "data.yaml").exists():
        dy = root / "data.yaml"
    else:
        cands = list(root.rglob("data.yaml"))
        if not cands:
            raise FileNotFoundError(f"data.yaml not found under: {root}")
        dy = sorted(cands, key=lambda p: len(p.parts))[0]

    cfg = yaml.safe_load(dy.read_text())
    cfg["path"] = str(dy.parent.resolve())

    droot = dy.parent
    val_dir = droot / "images" / "val"
    test_dir = droot / "images" / "test"
    if not val_dir.exists() or not any(val_dir.glob("*")):
        if test_dir.exists() and any(test_dir.glob("*")):
            cfg["val"] = "images/test"

    dy.write_text(yaml.dump(cfg, sort_keys=False), encoding="utf-8")
    return dy


# ===================== C. Утилиты для модели/инференса =====================


@st.cache_resource(show_spinner=False)
def load_model(weights_path: Union[str, Path]) -> YOLO:
    return YOLO(str(weights_path))


def postfilter_per_class(result, per_class_thr: Dict[int, float], default_thr: float):
    b = result.boxes
    if b is None or getattr(b, "shape", [0])[0] == 0:
        return result
    keep = []
    for i in range(b.shape[0]):
        cls_i = int(b.cls[i].item())
        conf_i = float(b.conf[i].item())
        if conf_i >= per_class_thr.get(cls_i, default_thr):
            keep.append(i)
    result.boxes = b[keep] if keep else b[:0]
    return result


# ===================== D. Мини-интерактив: PR и P/R/F1/mAP50 vs conf =====================


def _quick_conf_sweep(
    model, data_yaml, device, split="val", imgsz=640, iou=0.70, steps=11
):
    """Быстрый свип по conf (11 точек): возвращает DataFrame с P/R/F1/mAP50."""
    ths = np.linspace(0.05, 0.95, steps)
    rows = []
    for cf in ths:
        r = model.val(
            data=str(data_yaml),
            split=split,
            device=device,
            imgsz=imgsz,
            iou=iou,
            conf=float(cf),
            plots=False,
            verbose=False,
        )
        P = r.results_dict.get("metrics/precision(B)")
        R = r.results_dict.get("metrics/recall(B)")
        m50 = r.results_dict.get("metrics/mAP50(B)")
        F1 = (2 * P * R) / (P + R + 1e-9)
        rows.append(
            {"conf": float(cf), "precision": P, "recall": R, "f1": F1, "mAP50": m50}
        )
    return pd.DataFrame(rows)


def show_interactive_curves(df, split="val"):
    """Две интерактивные фигуры: PR и метрики vs conf (hover-подсказки)."""
    # PR (Precision vs Recall)
    d = df.dropna(subset=["precision", "recall"]).sort_values("recall")
    fig_pr = px.line(
        d, x="recall", y="precision", markers=True, title=f"PR curve ({split})"
    )
    fig_pr.update_layout(hovermode="x unified", height=420, template="plotly_white")
    fig_pr.update_xaxes(title="Recall", range=[0, 1])
    fig_pr.update_yaxes(title="Precision", range=[0, 1])
    st.plotly_chart(fig_pr, use_container_width=True)

    # P/R/F1/mAP50 vs conf
    fig_m = px.line(
        df,
        x="conf",
        y=["precision", "recall", "f1", "mAP50"],
        markers=True,
        title=f"Metrics vs confidence ({split})",
    )
    fig_m.update_layout(hovermode="x unified", height=420, template="plotly_white")
    fig_m.update_xaxes(title="Confidence threshold", range=[0, 1])
    fig_m.update_yaxes(title="Score", range=[0, 1])
    st.plotly_chart(fig_m, use_container_width=True)


# ===================== E. Sidebar (контролы + override ссылок) =====================
try:
    st.sidebar.markdown("### Перейти к странице:")
    st.sidebar.page_link("main.py", label="🏠 Главная")
    st.sidebar.page_link("pages/Faces.py", label="👤 Детекция лиц")
    st.sidebar.page_link("pages/Tumors.py", label="🧠 Детекция опухолей мозга")
    st.sidebar.page_link("pages/Forest.py", label="🌲 Сегментация лесов")
except Exception:
    st.sidebar.markdown("[🏠 Вернуться на главную](./)")


st.sidebar.title("Settings")

device_opt = st.sidebar.selectbox("Device", ["auto", "cuda:0", "cpu"], index=0)
infer_conf = st.sidebar.slider("Default conf", 0.0, 1.0, 0.20, 0.01)
infer_iou = st.sidebar.slider("NMS IoU", 0.1, 0.95, 0.65, 0.01)
max_det = st.sidebar.number_input("max_det", min_value=1, max_value=2000, value=300)

st.sidebar.markdown("---")
st.sidebar.subheader("Sources override (optional)")
weights_override = st.sidebar.text_input(
    "Weights URL or local path (override)", value=""
)
dataset_override = st.sidebar.text_input(
    "Dataset ZIP/dir URL or local path (override)", value=""
)
force_redownload = st.sidebar.checkbox("Re-download (ignore cache)", value=False)

# ===================== F. Скачиваем веса и датасет =====================

weights_src = weights_override.strip() or WEIGHTS_SRC
dataset_src = dataset_override.strip() or DATASET_SRC

if not weights_src or not dataset_src:
    st.error(
        "Заполни WEIGHTS_SRC и DATASET_SRC в начале файла или укажи ссылки в сайдбаре."
    )
    st.stop()

try:
    weights_local = download_to_cache(
        weights_src, "yolo_weights", suggested_name="weights.pt", force=force_redownload
    )
except Exception as e:
    st.error(f"Не удалось скачать веса: {e}")
    st.stop()

try:
    ds_root = fetch_dataset(dataset_src, force=force_redownload)
    data_yaml_path = find_and_fix_data_yaml(ds_root)
except Exception as e:
    st.error(f"Не удалось подготовить датасет: {e}")
    st.stop()

# ===================== G. Грузим модель =====================

device = (
    0
    if (device_opt == "auto" and torch.cuda.is_available())
    else (device_opt if device_opt != "auto" else "cpu")
)

if not Path(weights_local).exists():
    st.error(f"Weights not found: {weights_local}")
    st.stop()

model = load_model(weights_local)
name_map: Dict[int, str] = model.model.names
per_class = {
    cls_id: st.sidebar.slider(
        f"conf for '{cls_name}'",
        0.0,
        1.0,
        0.55 if cls_name == "positive" else infer_conf,
        0.01,
    )
    for cls_id, cls_name in name_map.items()
}
st.sidebar.info(f"Loaded model with {len(name_map)} classes: {list(name_map.values())}")

# ===================== H. Вкладки: Inference / Validate =====================

tab_inf, tab_val = st.tabs(["🔎 Inference", "📊 Validate (Interactive)"])

# ---------- Inference ----------
with tab_inf:
    st.header("Inference")
    st.write("Загрузи изображения/папку; пороги меняются на лету (слайдеры слева).")

    up = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )
    folder_hint = st.text_input(
        "...или путь к папке с изображениями (опционально)", value=""
    )
    run_btn = st.button("Run inference")
    if run_btn:
        sources: List[str] = []
        if up:
            tmpdir = Path("./_uploads")
            tmpdir.mkdir(exist_ok=True, parents=True)
            for f in up:
                p = tmpdir / f.name
                p.write_bytes(f.getvalue())
                sources.append(str(p))
        if folder_hint and Path(folder_hint).exists():
            exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
            sources += [
                str(p) for p in Path(folder_hint).glob("*") if p.suffix.lower() in exts
            ]

        if not sources:
            st.warning("Нет входных изображений.")
        else:
            base_conf = min([infer_conf, *per_class.values()])
            raw_results = model.predict(
                source=sources,
                device=device,
                imgsz=640,
                conf=base_conf,
                iou=infer_iou,
                max_det=int(max_det),
                save=False,
                verbose=False,
            )
            cols = st.columns(2)
            for i, r in enumerate(raw_results):
                r = postfilter_per_class(
                    r, per_class_thr=per_class, default_thr=infer_conf
                )
                plotted = r.plot()  # BGR ndarray
                pil_img = (
                    Image.fromarray(plotted[:, :, ::-1])
                    if plotted.ndim == 3
                    else Image.fromarray(plotted)
                )
                with cols[i % 2]:
                    st.image(
                        pil_img, caption=Path(r.path).name, use_container_width=True
                    )
                    if r.boxes is not None and getattr(r.boxes, "shape", [0])[0] > 0:
                        rows = []
                        for j in range(r.boxes.shape[0]):
                            c = int(r.boxes.cls[j].item())
                            confj = float(r.boxes.conf[j].item())
                            x1, y1, x2, y2 = r.boxes.xyxy[j].tolist()
                            rows.append(
                                [
                                    name_map[c],
                                    f"{confj:.2f}",
                                    int(x1),
                                    int(y1),
                                    int(x2),
                                    int(y2),
                                ]
                            )
                        st.dataframe(
                            rows,
                            hide_index=True,
                            column_config={
                                0: "class",
                                1: "conf",
                                2: "x1",
                                3: "y1",
                                4: "x2",
                                5: "y2",
                            },
                        )

# ---------- Validate (Interactive) ----------
with tab_val:
    st.header("Validate (interactive): hoverable charts")
    st.write(f"Using data.yaml: `{data_yaml_path}`")
    split = st.selectbox("Split", ["val", "test", "train"], index=0)

    run_val = st.button("Run validation & charts")
    if run_val:
        with st.spinner("Running validation..."):
            res = model.val(
                data=str(data_yaml_path),
                split=split,
                device=device,
                imgsz=640,
                conf=0.001,
                iou=0.70,
                plots=False,
                verbose=False,
                save_hybrid=False,
            )
        st.success("Validation done.")
        st.write(
            {
                "precision": round(
                    res.results_dict.get("metrics/precision(B)", float("nan")), 4
                ),
                "recall": round(
                    res.results_dict.get("metrics/recall(B)", float("nan")), 4
                ),
                "mAP50": round(
                    res.results_dict.get("metrics/mAP50(B)", float("nan")), 4
                ),
                "mAP50-95": round(
                    res.results_dict.get("metrics/mAP50-95(B)", float("nan")), 4
                ),
            }
        )
        st.divider()

        # Интерактивные графики (11 точек по conf)
        with st.spinner("Building interactive charts..."):
            df_metrics = _quick_conf_sweep(
                model,
                data_yaml_path,
                device,
                split=split,
                imgsz=640,
                iou=0.70,
                steps=11,
            )
        show_interactive_curves(df_metrics, split=split)
