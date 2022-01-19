import os
import time
import torch
import torch.nn as nn
import pdb


class Checkpoint(object):
    def __init__(
            self,
            model = None,                   
            optimizer = None,                     
            epoch = None,     
            config=None,                    
    ):

        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.exp_day = config.train.exp_day
        self.architecture = config.model.architecture
        self.config = config
        self.SAVE_PATH = 'outputs/model_pt/'+str(self.architecture)+'/'+str(self.exp_day)
        self.LOAD_PATH = 'outputs/model_pt/'+str(self.architecture)+'/'+str(self.exp_day)
        self.TRAINER_STATE_NAME = 'trainer_states.pt'
        self.LAST_MODEL = 'checkpoint_last.pth'
        
    def save(self):

        trainer_states = {
            'optimizer': self.optimizer,
            'epoch': self.epoch
        }
        
        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)
        
        torch.save(self.model, os.path.join(self.SAVE_PATH, 'checkpoint_epoch_'+str(self.epoch)+'.pt'))
        torch.save(trainer_states, os.path.join(self.SAVE_PATH, self.TRAINER_STATE_NAME))
        torch.save(self.model, os.path.join(self.SAVE_PATH, self.LAST_MODEL))

    def load(self, path, n=None):
        
        pt_path = f'checkpoint_epoch_{n}.pt' if n else self.LAST_MODEL

        print('load checkpoints\n%s\n%s'
                    % (os.path.join(path, self.TRAINER_STATE_NAME),
                       os.path.join(path, pt_path)))

        resume_checkpoint = torch.load(os.path.join(path, self.TRAINER_STATE_NAME))
        model = torch.load(os.path.join(path, pt_path))

        epoch = n if n else resume_checkpoint['epoch']

        return Checkpoint(
            model=model, 
            optimizer=resume_checkpoint['optimizer'],
            epoch=epoch,
            config = self.config,
        )

    def get_latest_checkpoint(self):
        """
        returns the path to the last saved checkpoint's subdirectory.
        Precondition: at least one checkpoint has been made (i.e., latest checkpoint subdirectory exists).
        """

        return self.LOAD_PATH