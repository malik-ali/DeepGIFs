import torch

import json
import os
import time
import shutil

class ModelTrainer:
    CHECKPOINTS_DIR = 'checkpoints'
    STATS_DIR = 'stats'

    def __init__(self, model, optimizer, train_fn, save_root, tags=[], experiment_num=None, save_models=False):
        self.model = model
        self.optimizer = optimizer
        self.train_fn = train_fn
        self.save_models = save_models
        self.tags = tags

        # make project folder
        if not os.path.exists(save_root):
            os.makedirs(save_root)
            
        # make experiment folder
        if experiment_num != None:
            self.save_dir = os.path.join(save_root, f'experiment_{experiment_num}')

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        else:
            self.save_dir = self.get_experiment_dir(save_root)

        # make stats and checkpoints for experiments
        if not os.path.exists(os.path.join(self.save_dir, self.STATS_DIR)):
            os.makedirs(os.path.join(self.save_dir, self.STATS_DIR))

        if not os.path.exists(os.path.join(self.save_dir, self.CHECKPOINTS_DIR)):
            os.makedirs(os.path.join(self.save_dir, self.CHECKPOINTS_DIR))

    def get_experiment_dir(self, save_root):
        experiment_dirs = [dirent for dirent in os.listdir(save_root) if 'experiment' in dirent]
        experiment_nums = [int(dirent.split('_')[-1]) for dirent in experiment_dirs]
        experiment_highest = max(experiment_nums) if experiment_nums else -1
        
        experiment_dir = os.path.join(save_root, f'experiment_{experiment_highest + 1}')
        os.makedirs(experiment_dir)
        return experiment_dir
        
    def update_optimizer(self, new_optim):
        self.optimizer = new_optim

    def train(self, 
              loader, 
              epochs=1,
              print_every=10,
              print_return_every=None,
              save_every=100, 
              stats_every=10,
              verbose=True): 

        """
            - print_every: If verbose, how often to print stats
            - stats_every: Frequency of measuring statistics
            - save_every: Frequency of saving model and stats to disk
        """
        self.losses = []
        loss = None

        for e in range(epochs):
            for t, x in enumerate(loader):
                loss = self.train_fn(self.model, self.optimizer, x) 
                
                if verbose and print_return_every and t % print_return_every == 0:
                    print(f'Epoch [{e}] ({t}/{len(loader)}), loss = {loss:.4f}', end="\r")
                
                if verbose and t % print_every == 0:
                    print(f'Epoch [{e}] ({t}/{len(loader)}), loss = {loss:.4f}')

                if t % stats_every == 0:
                    self.losses.append(loss)

                if t % save_every == 0:
                    self.save_data()
            if verbose:
                print (f'Epoch [{e}] done                                 ')


    def get_savepaths(self, filename):
        model_savepath = os.path.join(self.save_dir, self.CHECKPOINTS_DIR, filename)
        stats_savepath = os.path.join(self.save_dir, self.STATS_DIR, f'{filename}.json')
        return model_savepath, stats_savepath

    def save_data(self, force_save=False):
        filename = str(time.time())

        model_savepath, stats_savepath = self.get_savepaths(filename)

        model_state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        stats = {
            'optimizer_params': str(self.optimizer.state_dict()['param_groups']),
            'losses': self.losses,
            'model_filename': filename,
            'tags' : self.tags
        }
       
        if self.save_models or force_save:
            torch.save(model_state, model_savepath)

        json.dump(stats, open(stats_savepath, 'w'))

        self.reset_stats()

    @staticmethod
    def load_model(checkpoint_path, model, optimizer=None): # optimizer
        last_checkpoint = sorted(os.listdir(checkpoint_path))[-1]
        last_checkpoint = os.path.join(checkpoint_path, last_checkpoint)
        state = torch.load(last_checkpoint)
        model.load_state_dict(state['state_dict'])
        if optimizer:
            optimizer.load_state_dict(state['optimizer'])

#         self.reset_stats()

    def reset_stats(self):
        self.losses = []
