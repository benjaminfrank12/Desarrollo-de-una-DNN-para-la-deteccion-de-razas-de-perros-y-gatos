import streamlit as st
import streamlit.components.v1 as com
import numpy as np
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from PIL import Image
import gdown
import os
import asyncio
import tempfile



# Function to create a card
def create_card(title, image_url):
    card_html = f"""
    <div class="card">
        <img class="card-image" src="{image_url}" alt="{title}">
        <div class="card-title">{title}</div>
    </div>
    """
    return card_html

# Function to download the model from Google Drive
def download_model_from_gdrive(gdrive_url, output_path):
    gdown.download(gdrive_url, output_path, quiet=False, fuzzy=True)

# Cache the model loading
@st.cache_resource
def load_model():
    model_path = 'yolov9c-seg-10epochs.pt'  # Especifica la ruta a tu archivo de modelo
    if not os.path.exists(model_path):
        st.error(f"El archivo de modelo no se encuentra en la ruta especificada: {model_path}")
        return None
    model = YOLO(model_path)
    return model

model = load_model()

classes = [
    'cat persa','dog labrador','cat siames','dog chihuahua','dog husky'
]

detected_classes = set()

def get_class_html(cls, detected_classes):
    detected_style = 'background-color:#FF4B4B;padding:4px 4px;border-radius:5px;margin:2px; display:inline-block; color:white;'
    default_style = 'padding:4px 4px;border-radius:5px;margin:2px; display:inline-block; background-color:white; color:black;'
    style = detected_style if cls in detected_classes else default_style
    return f'<span style="{style}">{cls}</span>'

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.confidence = 0.25

    def set_params(self, model, confidence):
        self.model = model
        self.confidence = confidence

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.model:
            results = self.model(img_rgb, conf=self.confidence)
            if results:
                annotated_frame = results[0].plot()
                return av.VideoFrame.from_ndarray(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR), format="bgr24")
        return av.VideoFrame.from_ndarray(img, format="bgr24")

async def process_image(image, model, confidence):
    img = Image.open(image)
    results = await asyncio.to_thread(model, img, conf=confidence)
    return results

def main():

    st.title("Detección de razas de perros y gatos")
    html_classesp = [get_class_html(cls, detected_classes) for cls in classes]
    st.markdown(
        f"<div style='padding:4px; border: 2px solid #FF4B4B; border-radius: 10px;'><h4 style='color:#FF4B4B;text-align:center;'>5 Clases</h4><p style='color:white;text-align:center;'>{' '.join(html_classesp)}</p></div><br>",
        unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        # Crear botones
        use_camera_button = st.button("Usar cámara")
    with col2:
        upload_image_button = st.button("Subir imagen")
    with col3:
        upload_video_button = st.button("Subir video")

    # CSS para cambiar el color de fondo de la página y estilizar los botones
    st.markdown("""
        <style>
        body {
            background-color: #f0f0f0;
            margin-top: 0;
            padding-top: 0;
        }
        .stButton > button {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 10px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            height: 300px; /* Ajusta la altura del botón para acomodar la imagen y el texto */
            width: 100%; /* Asegura que el botón ocupe todo el ancho de la columna */
        }
        .stButton > button:hover {
            opacity: 0.8;
        }
        .stButton > button img {
            margin-right: 10px;
        }
        .stButton > button:nth-of-type(1) {
            background-color: lightcoral;
        }
        .stButton > button:nth-of-type(2) {
            background-color: lightblue;
        }
        .stButton > button:nth-of-type(3) {
            background-color: lightgreen;
        }
        </style>
        <script>
        const elements = window.parent.document.querySelectorAll('.stButton button');
        if (elements.length >= 3) {
            elements[0].innerHTML = '<div class="card"><img class="card-image" src="https://st2.depositphotos.com/1915171/5331/v/450/depositphotos_53312473-stock-illustration-webcam-sign-icon-web-video.jpg" width="200.54px" height="235px" alt="camera"/><div class="card-title">Usar cámara</div></div>';
            elements[1].innerHTML = '<div class="card"><img class="card-image" src="https://i.pinimg.com/736x/e1/91/5c/e1915cea845d5e31e1ec113a34b45fd8.jpg" width="200.54px" height="235px" alt="camera"/><div class="card-title">Subir imagen</div></div>';
            elements[2].innerHTML = '<div class="card"><img class="card-image" src="https://static.vecteezy.com/system/resources/previews/005/919/290/original/video-play-film-player-movie-solid-icon-illustration-logo-template-suitable-for-many-purposes-free-vector.jpg" width="200.54px" height="235px" alt="camera"/><div class="card-title">Subir video</div></div>';
        }
        </script>
        """, unsafe_allow_html=True)

    # Acciones cuando se presionan los botones
    if use_camera_button:
        st.header("Utiliza tu cámara")
        if model:
            confidence_slider = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
            start_detection = st.checkbox("Iniciar detección de objetos")
            video_transformer = VideoTransformer()
            if start_detection:
                video_transformer.set_params(model, confidence_slider)
            webrtc_streamer(key="example", video_transformer_factory=lambda: video_transformer,rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


    if upload_image_button:
        st.header("Subir imagen")
        confidence_slider = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
        html_classes = [get_class_html(cls, detected_classes) for cls in classes]
        text_placeholder = st.empty()
        text_placeholder.markdown(
            f"<div style='padding:4px; border: 2px solid #FF4B4B; border-radius: 10px;'><h4 style='color:#FF4B4B;text-align:center;'>5 Clases</h4><p style='color:white;text-align:center;'>{' '.join(html_classes)}</p></div>",
            unsafe_allow_html=True)
        change_text = st.checkbox("Objetos Detectados")
        image = st.file_uploader('Sube imagen', type=['png', 'jpg', 'jpeg', 'gif'])

        if image:
            col1, col2, col3 = st.columns([1, 1, 1])
            col1.image(image, caption='Imagen original', use_column_width=True)  # Ajusta el uso del ancho de la columna
            if model:
                with col2:
                    with st.spinner('Procesando imagen...'):
                        results = asyncio.run(process_image(image, model, confidence_slider))
                        if results:
                            annotated_frame = results[0].plot()
                            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                            col2.image(annotated_frame, caption='Imagen anotada',
                                       use_column_width=True)  # Ajusta el uso del ancho de la columna
                            for result in results[0].boxes:
                                idx = int(result.cls.cpu().numpy()[0])
                                confidence = result.conf.cpu().numpy()[0]
                                detected_class = classes[idx]
                                detected_classes.add(detected_class)
                                col3.markdown(
                                    f"<div style='background-color:#f0f0f0;padding:5px;border-radius:5px;margin:5px 0; color:black;'><b>Clase:</b> <span style='color:black'>{detected_class}</span><br><b>Confianza:</b> {confidence:.2f}<br></div>",
                                    unsafe_allow_html=True)
                        else:
                            col3.write("No se detectaron objetos.")
            else:
                st.error("Model is not loaded. Please check the logs for errors.")
        if change_text:
            html_classes = [get_class_html(cls, detected_classes) for cls in classes]
            text_placeholder.markdown(
                f"<div style='padding:4px; border: 2px solid #FF4B4B; border-radius: 10px;'><h4 style='color:#FF4B4B;text-align:center;'>5 Clases</h4><p style='color:white;text-align:center;'>{' '.join(html_classes)}</p></div>",
                unsafe_allow_html=True)


    if upload_video_button:
        st.header("Subir video")
        # Aquí puedes añadir el código para subir un video


if __name__ == "__main__":
    main()