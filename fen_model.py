import torch
from torch import nn
import numpy as np
from typing import List,Tuple
import logging
from chess_utils import prepare_dataset, prepare_batch, get_batches,numeric_to_FEN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TO DO:
#add training code

class FENModel(torch.nn.Module):
    def __init__(self, load_path:str='', input_shape:int=2500, hidden_shapes:List[int]=[1250, 625, 250, 50]):
        super().__init__()
        self.output_shape = 13
        self.model = self._get_model(input_shape, hidden_shapes)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()
        self.trained = False
        self.figures_dict = {'p': 1,'b': 2,'n': 3,'r': 4,       #lowercase - white figures, uppercase - black
                                'q': 5,'k': 6,'P': 7,'B': 8,    #p - pawn, b - bishop, n - knight, r - rook, q - queen, k - king
                                'N': 9,'R': 10,'Q': 11,'K': 12}

        logging.info('Trying to load model parameters...')
        if load_path != '':
            try:
                load_params = torch.load(load_path, map_location=device)
                self.model.load_state_dict(load_params['model_state_dict'])
                self.optimizer.load_state_dict(load_params['optimizer_state_dict'])
                self.trained = True
                logging.info('Parameters are loaded successfully')
            except:
                logging.error("Couldn't load parameters. Check if the path is given in correct form.")
        
    def _get_model(self, input_shape, hidden_shapes):
        if hidden_shapes:
            model = [nn.Linear(input_shape, hidden_shapes[0]), nn.ReLU()]
            for i,v in enumerate(hidden_shapes[1:],1):
                model.append(nn.Linear(hidden_shapes[i-1], v))
                model.append(nn.ReLU())
            model += [nn.Linear(hidden_shapes[-1], self.output_shape)]
        else:
            model = [np.Linear(input_shape, self.output_shape)]
        
        return nn.Sequential(*model).to(device)
    
    def forward(self, batch:torch.Tensor):
        logits = self.model(batch)
        return nn.functional.log_softmax(logits) 
    
    def train(self, val_split:int=70000, epochs:int=10) -> None:
        train = prepare_dataset('train')
        train, val = train[:val_split], train[val_split:]
        train_batches = get_batches(train, 10)
        val_batches = get_batches(val, 10)

        EPOCHS=epochs
        for epoch in range(EPOCHS):
            for i, batch in enumerate(train_batches):
                X, y = prepare_batch(batch, encoder_dict = self.figures_dict)
                X = X.to(device)
                y = y.to(device)
                                
                preds = self.model(X)
                batch_loss = self.criterion(preds, y)
                if i%50 == 0:
                    print('', end='\r')
                    print(f'Epoch:{epoch+1}; Batch {i} of {len(train_batches)}; Loss={batch_loss}',end='\x1b[1K\r')
                    print('', end='\r')
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
        print()
        print('Done!')
    
    def save(self, path:str) -> None:
        torch.save({'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()}, path)
    
    def load(self, path:str) -> None:
        params = torch.load(path, map_location=device)
        self.model.load_state_dict(params['model_state_dict'])
        self.optimizer.load_state_dict(params['optimizer_state_dict'])

    def check_accuracy(self, dataset, dataset_type:str = 'train', got_batches:bool=False, batch_size:int=10) -> Tuple[float]:
        if not got_batches:
            FENs = [x.split('.')[0] for x in dataset]
            dataset = get_batches(dataset, batch_size)
        else:
            FENs = [x.split('.')[0] for y in dataset for x in y]
            
        y_a = []
        y_p = []
        for i,batch in enumerate(dataset):
            print('Batch #', i+1, 'of', len(dataset), end='\r')
            X, y = prepare_batch(batch, self.figures_dict, dataset_type)
            X = X.to(device)
            y_a += y.numpy().reshape(y.shape[0]//64,64).tolist()
            y_pred = self.model(X).argmax(1)
            y_pred = y_pred.cpu().numpy().reshape(y_pred.shape[0]//64,64).tolist() if y_pred.is_cuda else y_pred.numpy().reshape(y_pred.shape[0]//64,64).tolist()
            y_p += y_pred
        
        y_a = np.array(y_a)
        y_p = np.array(y_p)
        TP = (y_a==y_p).sum()
            
        pred_FENs = [numeric_to_FEN(x, self.figures_dict) for x in y_p]
        assert len(FENs) == len(pred_FENs)
        correct_fens = 0
        for i in range(len(FENs)):
            if FENs[i] == pred_FENs[i]:
                correct_fens += 1
        
        return TP/(y_p.shape[0]*y_p.shape[1]), correct_fens/len(FENs)