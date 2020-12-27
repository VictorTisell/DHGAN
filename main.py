from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from src.gtgan import GTGAN
import sys
import os
import QuantLib as ql
import datetime
import pandas as pd
import argparse
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
def main(args, option_settings):
    Model = GTGAN(args.symbols, vars(args), option_data ,option_settings)
    model_names = ['Generator', 'Supervisor', 'Recovery', 'Regression']
    os.chdir(sys.path[0] + '/' + sys.path[1])
    if args.train:
        Model.train(model_names, save = args.save, gbm = args.gbm)
    if args.simulate:
        input_ = input('number of trajectories to simulate: ')
        nb_sims = int(input_)
        W = Model.BrownianMotion(nb_sims, 31, 10)
        [gen_p, sup_p, rec_p, reg_p] = Model.Restore(model_names, gbm = args.gbm, P = True)
        [gen_q, sup_q, rec_q, reg_q] = Model.Restore(model_names, gbm = args.gbm, P = False)
        X_hat_p, eta_p = Model.SimulateHedge_Prices(W, gen_p, sup_p, rec_p, reg_p, gbm = args.gbm)
        X_hat_q, eta_q = Model.SimulateHedge_Prices(W, gen_q, sup_q, rec_q, reg_q, gbm = args.gbm)
        if args.plot_trajectories:
            for i in range(nb_sims):
                plt.plot(X_hat_p[i, :, 0])
            plt.show()
            plt.close()
            for i in range(nb_sims):
                plt.plot(X_hat_q[i, :, 0])
            plt.show()
            plt.close()
    return X_hat_p, eta_p, X_hat_q, eta_q


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--symbols',
        help = '(Yahoo Finance) symbols',
        default = ['^GSPC', '^VIX', '^NDX', '^RUT', '^DJI',
                    '^FVX','^TNX', '^TYX', 'EURUSD=X','JPY=X'],
        type = list)
    parser.add_argument(
        '--seq_len',
        help = 'maximum sequence length',
        default = 31,
        type = int)
    parser.add_argument(
        '--module',
        help = 'module for GTGAN algorithm',
        choices = ['gru', 'lstm'],
        default = 'lstm',
        type = str)
    parser.add_argument(
        '--hidden_dim',
        help = 'number of neurons for GAN (optimize)',
        default = 24,
        type = int)
    parser.add_argument(
        '--latent_dim',
        help = 'dimensions of latent state representation (optimize)',
        default = 2,
        type = int)
    parser.add_argument(
        '--num_layers',
        help = 'number of hidden layers for GAN (optimize)',
        default = 3,
        type = int)
    parser.add_argument(
        '--iterations',
        help = 'number of iterations (optimize)',
        default = 50000,
        type  = int)
    parser.add_argument(
        '--learning_rate',
        help = 'learning rate',
        default = 0.001,
        type  = float)
    parser.add_argument(
        '--regression_learning_rate',
        help = 'regression learning rate',
        default = 0.0001,
        type  = float)
    parser.add_argument(
        '--regression_hidden_dim',
        help = 'hidden dimension for regression',
        default = 70,
        type  = int)
    parser.add_argument(
        '--batch_size',
        help = 'batch size (optimize)',
        default = 100,
        type = int)
    parser.add_argument(
        '--regression_features',
        help = 'number of features for regression algorithm',
        default = 1,
        type = int),
    parser.add_argument(
        '--gbm',
        help = 'use geometric brownian motion',
        default = False,
        type = bool)
    parser.add_argument(
        '--train',
        help = 'train model',
        default = False,
        type = bool)
    parser.add_argument(
        '--save',
        help = 'save model',
        default = False,
        type = bool)
    parser.add_argument(
        '--simulate',
        help = 'simulate data',
        default = False,
        type = bool)
    parser.add_argument(
        '--plot_trajectories',
        help = 'trajectory plot',
        default = False,
        type = bool)
    ### Add arguments for metrics (distributional)
    datafolder_path = 'src/datafolder'
    filename = 'quotedata_SPX.csv'
    filepath = datafolder_path + '/' + filename
    option_data = pd.read_csv(filepath, sep = ';')
    calendar = ql.UnitedStates()
    Ks = [3000, 3100, 3200, 3300, 3400]
    ttms = [10, 21, 31]
    option_settings = {'start': datetime.date(2020,9,11),
                'day_count': ql.Actual365Fixed(),
                'calendar': ql.UnitedStates(),
                'strikes': Ks,
                'maturities':ttms}
    args = parser.parse_args()
    X_hat_p, eta_p, X_hat_q, eta_q = main(args, option_settings)
