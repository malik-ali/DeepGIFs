import torch

import json
import os
import time
import shutil

class ModelTrainer:
    CHECKPOINTS_DIR = 'checkpoints'
    STATS_DIR = 'stats'

    def __init__(self, gen_model, disc_model, gen_optimizer, disc_optimizer, train_fn, save_root, tags=[], experiment_num=None, save_models=False):
        self.gen_model = gen_model
        self.disc_model = disc_model
        
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        
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
              verbose=True,
              time_every=10): 

        """
            - print_every: If verbose, how often to print stats
            - stats_every: Frequency of measuring statistics
            - save_every: Frequency of saving model and stats to disk
        """
        self.losses = []
        gen_loss = disc_loss = None
        
        
        current_batches = 0
        remaining_batches = epochs * len(loader)
        total_time = 0

        for e in range(epochs):
            for t, x in enumerate(loader):
                batch_start_time = time.time()
                
                gen_loss, disc_loss, gen_loss_str, disc_loss_str = self.train_fn(self.gen_model, self.disc_model, self.gen_optimizer, self.disc_optimizer, x) 
                
                
                if verbose and t % print_every == 0:
                    print(f'Epoch [{e}] ({t}/{len(loader)}), \t gen_loss  = {gen_loss:.4f} \t({gen_loss_str}), \n\t\t\t ' 
                      +  f'disc_loss = {disc_loss:.4f} \t({disc_loss_str})\n')

                if t % stats_every == 0:
                    self.losses.append({
                        'gen_loss': gen_loss.item(),
                        'disc_loss': disc_loss.item(),
                        'gen_loss_str': gen_loss_str,
                        'disc_loss_str': disc_loss_str
                    })

                if t % save_every == 0:
                    self.save_data()
                    
                if verbose and t % time_every == 0 and current_batches:
                    average_batch_time = total_time / current_batches
                    seconds = int(average_batch_time * remaining_batches)
                    minutes = seconds // 60
                    seconds = seconds % 60
                    print (f'Time remaining: {minutes}m {seconds}s     ', end='\r')
                    
                batch_end_time = time.time()
                total_time += batch_end_time - batch_start_time
                current_batches += 1
                remaining_batches -= 1
                    
            if verbose:
                print (f'Epoch [{e}] done                                 ')
            self.save_data(force_save=True) # TODO: remove


    def get_savepaths(self, filename):
        model_savepath = os.path.join(self.save_dir, self.CHECKPOINTS_DIR, filename)
        stats_savepath = os.path.join(self.save_dir, self.STATS_DIR, f'{filename}.json')
        return model_savepath, stats_savepath

    def save_data(self, force_save=False):
        filename = str(time.time())

        model_savepath, stats_savepath = self.get_savepaths(filename)

        model_state = {
            'gen_state_dict': self.gen_model.state_dict(),
            'disc_state_dict': self.disc_model.state_dict(),
            
            'gen_optimizer': self.gen_optimizer.state_dict(),
            'disc_optimizer': self.disc_optimizer.state_dict()
        }
        
        stats = {
            'gen_optimizer_params': str(self.gen_optimizer.state_dict()['param_groups']),
            'disc_optimizer_params': str(self.disc_optimizer.state_dict()['param_groups']),            
            'losses': self.losses,
            'model_filename': filename,
            'tags' : self.tags
        }
       
        if self.save_models or force_save:
            torch.save(model_state, model_savepath)

        json.dump(stats, open(stats_savepath, 'w'))

        self.reset_stats()

    @staticmethod
    def load_model(checkpoint_path, gen_model, disc_model, gen_optimizer=None, disc_optimizer=None): # optimizer
        last_checkpoint = sorted(os.listdir(checkpoint_path))[-1]
        last_checkpoint = os.path.join(checkpoint_path, last_checkpoint)
        state = torch.load(last_checkpoint)
        
        gen_model.load_state_dict(state['gen_state_dict'])
        disc_model.load_state_dict(state['disc_state_dict'])        
        
        if gen_optimizer:
            gen_optimizer.load_state_dict(state['gen_optimizer'])

        if disc_optimizer:
            disc_optimizer.load_state_dict(state['disc_optimizer'])
            
#         self.reset_stats()

    def reset_stats(self):
        self.losses = []
