from flask import Flask, flash, request, redirect, url_for, render_template
import torch
import numpy as np
from PIL import Image, ImageOps
from fen_model import FENModel
from chess_utils import numeric_to_FEN, slice_board
from typing import List
import requests
import base64
from io import BytesIO

class ChessFENConvertor():
    def __init__(self, fen_model_path:str, detector_model_path:str):
        self.FENModel = FENModel(fen_model_path)
        self.detector = torch.hub.load('ultralytics/yolov5', 'custom', path = detector_model_path)
        self.grayscaler = ImageOps.grayscale
    
    def detect_chessboard(self, image_base64:bytes):
        byte_data = base64.b64decode(image_base64)
        image_data = BytesIO(byte_data)
        img = Image.open(image_data)

        found_boards = self.detector(img)
        found_boards = found_boards.xyxy[0].numpy()
        boards_b64 = []
        for board in found_boards:
            board = board[:4]
            board = img.crop(board)

            output_buffer = BytesIO()
            board.save(output_buffer, format='PNG')
            byte_data = output_buffer.getvalue()
            boards_b64.append(base64.b64encode(byte_data))
        return boards_b64


    def get_FEN(self, boards:List[bytes]):
        FENS = []
        for board in boards:
            byte_data = base64.b64decode(board)
            image_data = BytesIO(byte_data)
            img = Image.open(image_data)
            img = img.resize((400,400))

            bw_board = self.grayscaler(img)
            bw_board = np.array(bw_board)
            sliced_board = slice_board(bw_board)
            sliced_board = torch.FloatTensor(sliced_board)

            ans = self.FENModel(sliced_board).argmax(1)
            if ans.is_cuda:
                ans = ans.cpu()
            ans = ans.numpy().reshape(ans.shape[0]//64,64)
            FEN = numeric_to_FEN(ans, self.FENModel.figures_dict)

            FENS.append(FEN)
        
        return FENS
    

app = Flask(__name__)
app.secret_key = 'the secretest key'
app.config['UPLOAD_EXTENSIONS'] = ['jpg', 'png', 'jpeg']

FENconvertor = ChessFENConvertor(r'models_weights\fen_model_weights', r'models_weights\chessboard_detector_weights.pt')
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

    boards_b64 = FENconvertor.detect_chessboard(src_str)
    FENS = FENconvertor.get_FEN(boards_b64)

    result = [((img_template + boards).decode('utf-8'), fen) for boards, fen in zip(boards_b64, FENS)]

    return render_template('index.html', src_base64=src_base64.decode('utf-8'), result=result)
