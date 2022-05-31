from flask import Flask, flash, request, redirect, url_for, render_template
import torch
import numpy as np
from PIL import Image
from fen_model import FENModel
from chess_utils import detect_chessboard, get_FEN
import base64
from io import BytesIO
    

app = Flask(__name__)
app.secret_key = 'the secretest key'
app.config['UPLOAD_EXTENSIONS'] = ['jpg', 'png', 'jpeg']

detector_model_path = r'models_weights\chessboard_detector_weights.pt'
detector = torch.hub.load('ultralytics/yolov5', 'custom', path = detector_model_path)

fen_model_path = r'models_weights\fen_model_weights'
FENModel = FENModel(fen_model_path)

img_template = bytes("data:image/png;base64,", encoding='utf-8')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    uploaded_file = request.files['file']
    
    if uploaded_file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    right_type = '.' in uploaded_file.filename and uploaded_file.filename.split('.')[-1].lower() in app.config['UPLOAD_EXTENSIONS']
    if not right_type:
        flash('Wrong type. Allowed image types are -> png, jpg, jpeg')
        return redirect(request.url)

    src_str = base64.b64encode(uploaded_file.stream.read())
    src_base64 = img_template + src_str

    boards_b64 = detect_chessboard(detector, src_str)
    FENS = get_FEN(FENModel, boards_b64)

    result = [((img_template + boards).decode('utf-8'), fen) for boards, fen in zip(boards_b64, FENS)]

    return render_template('index.html', src_base64=src_base64.decode('utf-8'), result=result)

@app.route('/example', methods=['POST','GET'])
def example():
    img = Image.open('example.png')
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    src_str = base64.b64encode(byte_data)
    src_base64 = img_template + src_str

    boards_b64 = detect_chessboard(detector, src_str)
    FENS = get_FEN(FENModel, boards_b64)

    result = [((img_template + boards).decode('utf-8'), fen) for boards, fen in zip(boards_b64, FENS)]

    return render_template('index.html', src_base64=src_base64.decode('utf-8'), result=result)

    