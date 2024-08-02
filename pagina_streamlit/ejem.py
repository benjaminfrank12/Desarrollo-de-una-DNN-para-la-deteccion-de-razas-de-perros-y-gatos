import streamlit as st
from streamlit.components.v1 import html
import base64
import io
from PIL import Image
import asyncio
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, ClientSettings

st.title("Acceso a la Cámara en Streamlit")

html_code = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Camera Access</title>
</head>
<body>
    <h1>Acceso a la Cámara</h1>
    <video id="video" width="320" height="240" autoplay></video>
    <br>
    <button id="start">Encender Cámara</button>
    <button id="stop">Apagar Cámara</button>
    <button id="switch">Cambiar Cámara</button>
    <canvas id="canvas" width="320" height="240" style="display:none;"></canvas>
    <script>
        let video = document.getElementById('video');
        let canvas = document.getElementById('canvas');
        let context = canvas.getContext('2d');
        let stream = null;
        let currentFacingMode = "user";

        document.getElementById('start').addEventListener('click', async () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: currentFacingMode }
            });
            video.srcObject = stream;
            video.play();
        });

        document.getElementById('stop').addEventListener('click', () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
                video.pause();
                video.srcObject = null;
            }
        });

        document.getElementById('switch').addEventListener('click', async () => {
            if (currentFacingMode === "user") {
                currentFacingMode = { exact: "environment" };
            } else {
                currentFacingMode = "user";
            }
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: currentFacingMode }
            });
            video.srcObject = stream;
            video.play();
        });
    </script>
</body>
</html>
'''
html(html_code, height=500)

