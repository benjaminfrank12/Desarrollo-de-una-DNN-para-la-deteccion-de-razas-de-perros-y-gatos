import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from streamlit_option_menu import option_menu
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
        margin: 10px;
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
            
        }
        .button:hover {
            background-color: #7133FF;
        }
        .button img {
            vertical-align: middle;
            margin-right: 10px;
        }
        .stButton > button{
            border: none;
            color: black;
            background-color: #f0f0f0;
            text-align: center;
            padding: 15px;
            width: 300px;
            transition: all 0.3s;
            cursor: pointer;
            border: 2px solid #FF4B4B; 
        }
        
        .stButton > button:hover{
            color: white;
            background-color: #7133FF;
            border: 2px solid #f0f0f0; 
        }
        
    </style>
""", unsafe_allow_html=True)

def download_model_from_gdrive(gdrive_url, output_path):
    gdown.download(gdrive_url, output_path, quiet=False, fuzzy=True)


# Cache the model loading
@st.cache_resource
def load_model():
    model_path = 'yolov9c-seg-50epochs.pt'  # Especifica la ruta a tu archivo de modelo
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
    selected = option_menu(
        menu_title=None,
        options = ["Inicio", "Subir imagen", "Usar camara"],
        icons = ['house-door-fill', 'file-earmark-image-fill', "webcam-fill"],
        default_index=0,
        orientation="horizontal",
    )


    # Añadir el título de la tabla
    html_classesp = [get_class_html(cls, detected_classes) for cls in classes]
    #st.markdown("<div class='title-op'><h4>Selecciona una opción</h4></div>", unsafe_allow_html=True)
    if selected == "Inicio":
        inicio()

    if selected == "Subir imagen":
        subir_imagen()

    if selected == "Usar camara":
        usar_camara()


def inicio():
    html_classes = [get_class_html(cls, detected_classes) for cls in classes]
    text_placeholder = st.empty()
    text_placeholder.markdown(
        f"<div style='padding:4px; border: 2px solid #FF4B4B; border-radius: 10px;'><h4 style='color:#FF4B4B;text-align:center;'>5 Clases</h4><p style='color:white;text-align:center;'>{' '.join(html_classes)}</p></div>",
        unsafe_allow_html=True)
    img = Image.open(r"img.png")
    st.image(img)

def subir_imagen():
    st.header("Subir imagen")
    confidence_slider = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
    html_classes = [get_class_html(cls, detected_classes) for cls in classes]
    text_placeholder = st.empty()
    text_placeholder.markdown(
        f"<div style='padding:4px; border: 2px solid #FF4B4B; border-radius: 10px;'><h4 style='color:#FF4B4B;text-align:center;'>5 Clases</h4><p style='color:white;text-align:center;'>{' '.join(html_classes)}</p></div>",
        unsafe_allow_html=True)
    change_text = st.checkbox("Objetos Detectados",value=True)
    image = st.file_uploader('Sube imagen', type=['png', 'jpg', 'jpeg', 'gif'])
    #confidence_slider = st.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
    if image:
        #col1.image(image, caption='Imagen original', use_column_width=True)
        # Procesar la imagen y realizar anotaciones
        if model:
            with st.spinner('Procesando imagen...'):
                col1, col2 = st.columns([1, 1])
                results = asyncio.run(process_image(image, model, confidence_slider))

                if results:
                    annotated_frame = results[0].plot()
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

                    # Mostrar la imagen anotada en la misma columna 1
                    col1.image(annotated_frame, caption='Imagen anotada', use_column_width=True)

                    for result in results[0].boxes:
                        idx = int(result.cls.cpu().numpy()[0])
                        confidence = result.conf.cpu().numpy()[0]
                        detected_class = classes[idx]
                        detected_classes.add(detected_class)

                        col2.markdown(
                            f"<div style='background-color:#f0f0f0;padding:5px;border-radius:5px;margin:5px 0; color:black;'><b>Clase:</b> <span style='color:black'>{detected_class}</span><br><b>Confianza:</b> {confidence:.2f}<br></div>",
                            unsafe_allow_html=True)
                else:
                    col2.write("No se detectaron objetos.")
        else:
            st.error("Model is not loaded. Please check the logs for errors.")
    if change_text:
        html_classes = [get_class_html(cls, detected_classes) for cls in classes]
        text_placeholder.markdown(
            f"<div style='padding:4px; border: 2px solid #FF4B4B; border-radius: 10px;'><h4 style='color:#FF4B4B;text-align:center;'>5 Clases</h4><p style='color:white;text-align:center;'>{' '.join(html_classes)}</p></div>",
            unsafe_allow_html=True)

def usar_camara():
    st.header("Utiliza tu cámara")
    if model:
        confidence_slider = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
        start_detection = st.checkbox("Iniciar detección de objetos")
        video_transformer = VideoTransformer()
        if start_detection:
            video_transformer.set_params(model, confidence_slider)
        webrtc_streamer(key="example", video_transformer_factory=lambda: video_transformer,
                        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if __name__ == "__main__":
    main()