import streamlit as st
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

# Custom CSS for card styling
st.markdown("""
    <style>
    .title-op{
        padding:4px; 
        border: 2px solid #FF4B4B; 
        border-radius: 10px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        alingn-items: center;
    }

    .container{
        display: flex:
        flex-direction: row;
        justify-content: space-between;
    }

    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        text-align: center;
        width: 90%;
        max-width: 300px;  /* Limit width for smaller screens */
        margin: 10px auto;  /* Center cards on the screen */
    }

    .card-title {
        font-size: 1.2em;  /* Slightly smaller font size */
        margin-bottom: 10px;
        color: black;
    }
    .card-image {
        width: 100%;  /* Use full width of the card */
        height: auto;  /* Adjust height to maintain aspect ratio */
        object-fit: cover;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    h4{
        color:#FF4B4B;
        text-align:center;
    }

    .button {
            display: inline-block;
            border-radius: 12px;
            background-color: #f0f0f0;
            border: none;
            color: black;
            text-align: center;
            padding: 15px;
            width: 300px;
            transition: all 0.3s;
            cursor: pointer;
            margin: 20px;
        }
        .button:hover {
            background-color: #7133FF;
        }
        .button img {
            vertical-align: middle;
            margin-right: 10px;
        }
    </style>
""", unsafe_allow_html=True)


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
    'cat persa', 'dog labrador', 'cat siames', 'dog chihuahua', 'dog husky'
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


def create_card(title, image_url):
    card_html = f"""
        <div class="card">
        <img class="card-image" src="{image_url}" alt="{title}">
        <div class="card-title">{title}</div>
        </div>
    """
    return card_html


def main():
    st.markdown("<h1 style='text-align: center;'>Detección de razas de perros y gatos</h1>", unsafe_allow_html=True)
    img = Image.open(r"C:\Users\VECTOR\Documents\OD_APP_WEB\OD_APP_WEB\img1.png")
    st.image(img)

    # Añadir el título de la tabla
    html_classesp = [get_class_html(cls, detected_classes) for cls in classes]
    st.markdown("<div class='title-op'><h4>Selecciona una opción</h4></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Crear botones con imágenes
    with col1:
        use_camera_button = st.markdown("""
               <a href="#" class="button">
                   <img class='card-image' src="https://st2.depositphotos.com/1915171/5331/v/450/depositphotos_53312473-stock-illustration-webcam-sign-icon-web-video.jpg" alt="Usar cámara">
                   <div class='card-title'>Usar camara</div>
               </a>
           """, unsafe_allow_html=True)

    with col2:
        upload_image_button = st.markdown("""
               <a href="#" class="button">
                   <img class='card-image' src="https://i.pinimg.com/736x/e1/91/5c/e1915cea845d5e31e1ec113a34b45fd8.jpg" alt="Subir imagen">
                   <div class='card-title'>Subir imagen</div>
               </a>
           """, unsafe_allow_html=True)

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
                col1.image(image, caption='Imagen original',
                           use_column_width=True)  # Ajusta el uso del ancho de la columna
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


if __name__ == "__main__":
    main()