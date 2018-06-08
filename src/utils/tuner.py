import torch
import os
import numpy as np

TUNE_EPOCHS_COARSE = 2
TUNE_EPOCHS_FINE = 5
TUNE_N_FINE = 4
TUNE_N_COARSE = 4
EXPERIMENTS_ROOT = os.path.join(os.getcwd(), 'src', 'experiments')

class ModelTuner:
    def __init__(self, experiment_name, model_class, initializer_class, device, model_args=[], model_kwargs={}):
        self.tune_coarse_stats_dir = os.path.join(EXPERIMENTS_ROOT, experiment_name, 'tune-coarse')
        self.tune_fine_stats_dir = os.path.join(EXPERIMENTS_ROOT, experiment_name, 'tune-fine')
        self.device = device
        self.model_class = model_class
        self.initializer_class = initializer_class
        self.model_args = model_args
        self.model_kwargs = model_kwargs
    
    def fine_tune(self, lr_base, latent_sz_base, n=TUNE_N_FINE, epochs=TUNE_EPOCHS_FINE):
        best_lr = lr_base
        best_latent_sz = latent_sz_base
        best_loss = float('inf')

        tune_offsets_lr = [lr_base / denom for denom in [2, 2, 4, 8, 16, 32]]
        tune_offsets_latent_sz = [lr_base / denom for denom in [2, 4, 10, 20, 30]]

        for lr_offset, latent_sz_offset in zip(tune_offsets_lr, tune_offsets_latent_sz):
            for i in range(n):
                lr = max(best_lr + (np.random.rand() * 2 - 1) * lr_offset, 1e-10)
                latent_sz = max(best_latent_sz + int((np.random.rand() * 2 - 1) * latent_sz_offset), 20)

                print (f'  [FINE] [lr = {lr}, latent_sz = {latent_sz}] ', end="")

                net = self.model_class(latent_size=latent_sz, 
                                       device=self.device)

                Initializer.initialize(model=net,
                                       initialization=init.xavier_uniform_,
                                       gain=init.calculate_gain('relu'))

                optimizer = optim.Adam(net.parameters(), 
                                       lr=lr)

                trainer = ModelTrainer(net, 
                                       optimizer, 
                                       train_fn, 
                                       self.tune_coarse_stats_dir,
                                       tags=['vae-tune', 'fine'],
                                       save_models=False)

                trainer.train(img_train_loader, 
                              verbose=False,
                              epochs=epochs,
                              save_every=50)

                loss = validation_loss(net, img_val_loader)

                print (f'Validation loss: {loss}')

                if loss < best_loss:
                    print (f'  [FINE]     [new best] Old best ({best_loss}) > new best ({loss})')
                    best_lr = lr
                    best_latent_sz = latent_sz
                    best_loss = loss
                    gif_test_autoenc(net, gif_val_loader)

        print (f'  [FINE] Done.')
        print ()
        print (f'  [FINE] Best learning rate = {all_best_lr}')
        print (f'  [FINE] Best latent size = {all_best_latent_sz}')
        return best_loss, best_lr, best_latent_sz
    
    def coarse_tune(self, llr_lower, llr_upper, llatent_sz_lower, llatent_sz_upper, n=TUNE_N_COARSE, epochs=TUNE_EPOCHS_COARSE):
        best_lr = None
        best_latent_sz = None
        best_loss = float('inf')

        for i in range(n):
            lr = 10 ** -np.random.uniform(llr_lower, llr_upper)
            latent_sz = 10 * np.random.randint(llatent_sz_lower, llatent_sz_upper)

            print (f'[COARSE] [lr = {lr}, latent_sz = {latent_sz}] ', end="")

            net = self.model_class(latent_size=latent_sz, 
                                   device=self.device)

            Initializer.initialize(model=net,
                                   initialization=init.xavier_uniform_,
                                   gain=init.calculate_gain('relu'))

            optimizer = optim.Adam(net.parameters(), 
                                   lr=lr)

            trainer = ModelTrainer(net, 
                                   optimizer, 
                                   train_fn, 
                                   self.tune_fine_stats_dir,
                                   tags=['vae-tune', 'coarse'],
                                   save_models=False)

            trainer.train(img_train_loader, 
                          verbose=False,
                          epochs=epochs,
                          save_every=50)

            loss = validation_loss(net, img_val_loader)

            print (f'Validation loss: {loss}')

            if loss < best_loss:
                print (f'[COARSE]     [NEW BEST] Old best ({best_loss}) > new best ({loss})')
                best_lr = lr
                best_latent_sz = latent_sz
                best_loss = loss
                gif_test_autoenc(net, gif_val_loader)

        print (f'[COARSE] Done.')
        print ()
        print (f'[COARSE] Best learning rate = {best_lr}')
        print (f'[COARSE] Best latent size = {best_latent_sz}')
        return best_loss, best_lr, best_latent_sz