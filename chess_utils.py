import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from PIL import Image,ImageOps
import base64
from io import BytesIO

def prepare_dataset(dataset_type):
    assert dataset_type in ('train','test')
    all_files = Path(dataset_type).glob('*.jpeg')
    
    return [x.name for x in all_files]

def get_batches(dataset:List[str], batch_size:int):
    N = len(dataset)
    batches = np.ceil(N/batch_size)
    result = [0]*batches
    
    for batch in range(batches):
        result[batch] = dataset[batch*batch_size:batch*batch_size+batch_size]
    
    return result

def prepare_batch(batch:List[str], encoder_dict: Dict[str,int], dataset_type:str='train'):
    assert dataset_type in ('train','test')
    
    N = len(batch)
    X = np.zeros((N*64, 2500))
    y = np.zeros(N*64)
    for ind, file in enumerate(batch):
        fen = file.split('.')[0]
        num_fen = FEN_to_numeric(fen, encoder_dict)
        y[ind*64:ind*64+64] = num_fen
        
        board_pic = Image.open(dataset_type+'/'+file)
        bw_board = ImageOps.grayscale(board_pic)#.resize((200,200))
        sliced_board = slice_board(np.array(bw_board))
        X[ind*64:ind*64+64] = sliced_board
        
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    
    return X, y

def slice_board(board:np.array) -> np.array:
    #split array of board to ranks (lines)
    res = np.array(np.split(board, 8, axis=0))
    #concatenate all ranks into one big line
    res = np.concatenate(res, axis=1)
    #now split that line into cells. Using flatten as we will use pixels as features
    res = [x.flatten() for x in np.split(res, 64, axis = 1)]
    
    return np.array(res)

def FEN_to_numeric(fen: str, encoder_dict: Dict[str,int]) -> np.array:
    """Transform FEN notation of board to a vector with 64 values. 
       It'll be used as label in training process.
       fen(str): a FEN string, which showning figures positions on board
       encoder_dict(Dict[str]): a dictionary, which is used to transform a piece character to a number
       
       Example:
       fen = '1B1B1K2-3p1N2-6k1-R7-5P2-4q3-7R-1B6'
       encoder_dict = {'p': 1, 'b': 2, 'n': 3,
            'r': 4, 'q': 5, 'k': 6, 'P': 7, 'B': 8,
            'N': 9, 'R': 10, 'Q': 11, 'K': 12}
       FEN_to_numeric(fen, encoder_dict)
       
       >array([ 0.,  8.,  0.,  8.,  0., 12.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
        9.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  6.,  0., 10.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  7.,  0.,
        0.,  0.,  0.,  0.,  0.,  5.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0., 10.,  0.,  8.,  0.,  0.,  0.,  0.,  0.,  0.])
       """
    ranks = fen.split('-')
    board = np.zeros((8,8))
    for ind, rank in enumerate(ranks):
        board[ind] = encode_FEN(rank,encoder_dict)
    
    board = board.flatten()
    
    return board
        
def encode_FEN(rank: str, encoder_dict:Dict[str,int]) -> np.array:
    encoded_fen = np.zeros(8)
    i = 0
    for c in rank:
        if c.isnumeric():
            i += int(c)
        elif c.isalpha():
            encoded_fen[i] = encoder_dict[c]
            i += 1
    
    return encoded_fen

def numeric_to_FEN(board: np.array, encoder_dict:Dict[int, str]) -> str:
    """Transform a vector interpretation of board to a FEN. 
   It'll be used to transform a prediction of a model.
   board(np.array): a numpy array with shape (64,)
   encoder_dict(Dict[str]): a dictionary, which was used to transform a piece character to a number
   Will be used to create a decoder_dict

   Example:
   board = np.array([ 0.,  2.,  0.,  2.,  0.,  2.,  0.,  0.,  0.,  0.,  0.,  4.,  0.,
        0.,  0.,  0.,  0.,  4., 12.,  0.,  0.,  0.,  0.,  2., 10.,  0.,
        0.,  0.,  0.,  0.,  0.,  0., 10.,  0.,  0., 10.,  0.,  6.,  0.,
        0.,  0.,  0.,  8.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  7.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  4.,  0.,  0.,  0.,  0.,  0.])
   decoder_dict = {1: 'p', 2: 'b',  3: 'n',  4: 'r',
         5: 'q', 6: 'k', 7: 'P', 8: 'B', 9: 'N',
         10: 'R', 11: 'Q', 12: 'K'}
   
   numeric_to_FEN(board, decoder_dict)

   >'1b1b1b2-3r4-1rK4b-R7-R2R1k2-2Bp4-2P5-2r5'
   """
    board = board.reshape((8,8))
    decoder_dict = {v:k for k,v in encoder_dict.items()}
    fen = ''
    for rank_numeric in board:
        fen += decode_FEN(rank_numeric, decoder_dict)
        fen += '-'
    #we don't need the last '-'
    return fen[:-1]
        
def decode_FEN(rank_numeric: np.array, decoder_dict: Dict[int, str]) -> str:
    result = ''
    empty = 0
    for n in rank_numeric:
        if n == 0:
            empty += 1
        else:
            if empty > 0:
                result += str(empty)
                empty = 0
            result += decoder_dict[n]
    if empty > 0:
        result += str(empty)
    
    return result

def detect_chessboard(detector, image_base64:bytes) -> List[bytes]:
        byte_data = base64.b64decode(image_base64)
        image_data = BytesIO(byte_data)
        img = Image.open(image_data)

        found_boards = detector(img)
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

def get_FEN(model:torch.nn.Module, boards:List[bytes]):
        FENS = []
        for board in boards:
            byte_data = base64.b64decode(board)
            image_data = BytesIO(byte_data)
            img = Image.open(image_data)
            img = img.resize((400,400))

            bw_board = ImageOps.grayscale(img)
            bw_board = np.array(bw_board)
            sliced_board = slice_board(bw_board)
            sliced_board = torch.FloatTensor(sliced_board)

            ans = model(sliced_board).argmax(1)
            if ans.is_cuda:
                ans = ans.cpu()
            ans = ans.numpy().reshape(ans.shape[0]//64,64)
            FEN = numeric_to_FEN(ans, model.figures_dict)

            FENS.append(FEN)
        
        return FENS