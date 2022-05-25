import torch
from torch import nn
import numpy as np
from typing import List
import logging

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
    
