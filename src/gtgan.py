import tensorflow as tf
import sys
import os
sys.path.insert(1, 'src')
from utils import Loader, Option_data, Losses
import pandas as pd
import numpy as np
import datetime
import QuantLib as ql
import matplotlib.pyplot as plt
from tensorflow.keras.layers import TimeDistributed
from tensorflow.python.client import device_lib
import scipy.stats as sci
import time
class GTGAN(Loader, Option_data, Losses):
    def __init__(self, symbol_list, settings, option_data, option_settings):
        Loader.__init__(self, symbol_list = symbol_list)
        Option_data.__init__(self, option_data ,option_settings)
        self.settings = settings
        self.input_data = np.asarray(self.data)
        self.module = self.settings['module']
        self.hidden_dim = self.settings['hidden_dim']
        self.latent_dim = self.settings['latent_dim']
        self.num_layers = self.settings['num_layers']
        self.iterations = self.settings['iterations']
        self.batch_size = self.settings['batch_size']
        self.learning_rate = self.settings['learning_rate']
        self.regression_features = self.settings['regression_features']
        self.regression_hidden_dim = self.settings['regression_hidden_dim']
        self.regression_learning_rate = self.settings['regression_learning_rate']
        self.no, self.seq_len, self.dim = self.input_data.shape
        self.W_dim  = self.dim
        ## simulated data (for illustration purposes)
        self.gbmdta = self.GeometricBrownian(self.no, self.seq_len, self.W_dim)
        self.norm_GBM_data, self.gbm_min_val, self.gbm_max_val  = self.MinMaxScaler(self.gbmdta)

        self.Embedder = self.EmbedderNet()
        self.Embedder.summary()
        self.Embedder_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.Recovery = self.RecoveryNet()
        self.Recovery.summary()
        self.Recovery_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.Supervisor = self.SupervisorNet()
        self.Supervisor.summary()
        self.Supervisor_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.Generator = self.GeneratorNet()
        self.Generator.summary()
        self.Generator_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.Discriminator = self.DiscriminatorNet()
        self.Discriminator.summary()
        self.Discriminator_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.Regression = self.RegressionNet()
        self.Regression_optimizer = tf.keras.optimizers.Adam(self.regression_learning_rate)
        self.Regression.summary()

    def GeometricBrownian(self, nb_sims, timesteps, W_dim, S0 = 3200, sigma = 0.2, mu = 0.1, dt = 1/365):
        S = np.zeros(shape = (nb_sims, timesteps, W_dim))
        S[:, 0, :] = S0
        for t in range(1, timesteps):
            W = np.random.randn(nb_sims, W_dim)* np.sqrt(dt)
            S[:,t, : ] = S[:, t-1, :]* np.exp((mu - 0.5*sigma**2)*dt + sigma*W)
        return S
    def BrownianMotion(self,nb_sims, timesteps, W_dim, dt = 1/365):
        W = np.zeros(shape = (nb_sims, timesteps, W_dim))
        for t in range(1, timesteps):
            W[:,t,:] = W[:,t-1,:] + np.random.randn(nb_sims, W_dim)* np.sqrt(dt)
        return W
    def BlackScholes(self, K, T, S0 = 3200 ,r = 0.01, sigma = 0.2):
        d1 = (np.log(S0/K) + (r + 0.5 * sigma **2) * T) /(sigma * np.sqrt(T))
        d2 = (np.log(S0/K) + (r - 0.5 * sigma **2) * T) /(sigma * np.sqrt(T))
        return S0 * sci.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T)* sci.norm.cdf(d2, 0.0, 1.0)
    #@tf.function
    def Renormalize_tf(self, X, gbm = False):
        if gbm:
            max_val = tf.cast(self.gbm_max_val, tf.float32)
            min_val = tf.cast(self.gbm_min_val, tf.float32)
            X = X * max_val
            X = X + min_val
        else:
            max_val = tf.cast(self.max_val, tf.float32)
            min_val = tf.cast(self.min_val, tf.float32)
            X = X * max_val
            X = X + min_val
            X = tf.math.exp(X)
            X = X * tf.convert_to_tensor(np.asarray(self.starting_values), dtype = tf.float32)
        return X
    def EmbedderNet(self):
        model = tf.keras.Sequential(name = 'EmbedderNet')
        if self.module == 'lstm':
            model.add(tf.keras.layers.LSTM(self.dim, batch_input_shape = (self.batch_size, self.seq_len, self.dim), activation = 'tanh',return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.LSTM(self.hidden_dim, activation = 'tanh', return_sequences = True))
        elif self.module == 'gru':
            model.add(tf.keras.layers.GRU(self.dim, batch_input_shape = (self.batch_size, self.seq_len, self.dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.GRU(self.hidden_dim, activation = 'tanh', return_sequences = True))
        else:
            raise NotImplementedError('Please Choose gru or lstm as module')
        model.add(TimeDistributed(tf.keras.layers.Dense(self.latent_dim, activation = 'sigmoid')))
        return model
    def RecoveryNet(self):
        model = tf.keras.Sequential(name = 'RecoveryNet')
        if self.module == 'lstm':
            model.add(tf.keras.layers.LSTM(self.hidden_dim, batch_input_shape = (self.batch_size, self.seq_len, self.latent_dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.LSTM(self.hidden_dim, activation = 'tanh', return_sequences = True))
        elif self.module == 'gru':
            model.add(tf.keras.layers.GRU(self.hidden_dim, batch_input_shape = (self.batch_size, self.seq_len, self.latent_dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.GRU(self.hidden_dim, activation = 'tanh', return_sequences = True))
        model.add(TimeDistributed(tf.keras.layers.Dense(self.dim, activation = 'sigmoid')))
        return model
    def SupervisorNet(self):
        model = tf.keras.Sequential(name = 'SupervisorNet')
        if self.module == 'lstm':
            model.add(tf.keras.layers.LSTM(self.hidden_dim, batch_input_shape = (self.batch_size, self.seq_len, self.latent_dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.LSTM(self.hidden_dim, activation = 'tanh', return_sequences = True))
        elif self.module == 'gru':
            model.add(tf.keras.layers.GRU(self.hidden_dim, batch_input_shape = (self.batch_size, self.seq_len, self.latent_dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.GRU(self.hidden_dim, activation = 'tanh', return_sequences = True))
        model.add(TimeDistributed(tf.keras.layers.Dense(self.latent_dim, activation = 'sigmoid')))
        return model
    def GeneratorNet(self, time = True):
        model = tf.keras.Sequential(name = 'GeneratorNet')
        if self.module == 'lstm':
            model.add(tf.keras.layers.LSTM(self.hidden_dim, batch_input_shape = (self.batch_size, self.seq_len, self.dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.LSTM(self.hidden_dim, activation = 'tanh', return_sequences = True))
        elif self.module == 'gru':
            model.add(tf.keras.layers.GRU(self.hidden_dim, batch_input_shape = (self.batch_size, self.seq_len, self.dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.GRU(self.hidden_dim, activation = 'tanh', return_sequences = True))
        if time:
            model.add(TimeDistributed(tf.keras.layers.Dense(self.latent_dim, activation = 'sigmoid')))
        else:
            model.add(TimeDistributed(tf.keras.layers.Dense(self.dim, activation = 'sigmoid')))
        return model
    def DiscriminatorNet(self, time = True):
        model = tf.keras.Sequential(name = 'DiscriminatorNet')
        if self.module == 'lstm':
            if time:
                model.add(tf.keras.layers.LSTM(self.hidden_dim, batch_input_shape = (self.batch_size, self.seq_len, self.latent_dim), activation = 'tanh', return_sequences = True))
            else:
                model.add(tf.keras.layers.LSTM(self.hidden_dim, batch_input_shape = (self.batch_size, self.seq_len, self.dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.LSTM(self.hidden_dim, activation = 'tanh', return_sequences = True))
        elif self.module == 'gru':
            if time:
                model.add(tf.keras.layers.GRU(self.hidden_dim, batch_input_shape = (self.batch_size, self.seq_len, self.latent_dim), activation = 'tanh', return_sequences = True))
            else:
                model.add(tf.keras.layers.GRU(self.hidden_dim, batch_input_shape = (self.batch_size, self.seq_len, self.dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.GRU(self.hidden_dim, activation = 'tanh', return_sequences = True))
        if time:
            model.add(TimeDistributed(tf.keras.layers.Dense(self.latent_dim, activation = 'linear')))
        else:
            model.add(TimeDistributed(tf.keras.layers.Dense(self.dim, activation = 'linear')))
        # model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.Dense(1, activation = 'linear'))
        return model
    def RegressionNet(self):
        inp = tf.keras.layers.Input(batch_shape = (self.batch_size, self.seq_len, self.dim))
        if self.module == 'lstm':
            X = tf.keras.layers.LSTM(self.regression_hidden_dim, activation = 'tanh', return_sequences = False)(inp)
            X = tf.keras.layers.BatchNormalization()(X)
            T1 = tf.keras.layers.RepeatVector(self.maturities[0]-1)(X)
            T2 = tf.keras.layers.RepeatVector(self.maturities[1]-1)(X)
            T3 = tf.keras.layers.RepeatVector(self.maturities[2]-1)(X)
            T1 = tf.keras.layers.LSTM(self.regression_hidden_dim, activation = 'tanh', return_sequences = True)(T1)
            T2 = tf.keras.layers.LSTM(self.regression_hidden_dim, activation = 'tanh', return_sequences = True)(T2)
            T3 = tf.keras.layers.LSTM(self.regression_hidden_dim, activation = 'tanh', return_sequences = True)(T3)
        elif self.module == 'gru':
            X = tf.keras.layers.GRU(self.regression_hidden_dim, activation = 'tanh', return_sequences = False)(inp)
            X = tf.keras.layers.BatchNormalization()(X)
            T1 = tf.keras.layers.RepeatVector(self.maturities[0]-1)(X)
            T2 = tf.keras.layers.RepeatVector(self.maturities[1]-1)(X)
            T3 = tf.keras.layers.RepeatVector(self.maturities[2]-1)(X)
            T1 = tf.keras.layers.GRU(self.regression_hidden_dim, activation = 'tanh', return_sequences = True)(T1)
            T2 = tf.keras.layers.GRU(self.regression_hidden_dim, activation = 'tanh', return_sequences = True)(T2)
            T3 = tf.keras.layers.GRU(self.regression_hidden_dim, activation = 'tanh', return_sequences = True)(T3)
        T1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.regression_features * len(self.strikes), activation = 'linear'))(T1)
        T2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.regression_features * len(self.strikes), activation = 'linear'))(T2)
        T3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.regression_features * len(self.strikes), activation = 'linear'))(T3)
        model = tf.keras.Model(inputs = inp, outputs = [T1, T2, T3], name = 'RegressionNet')
        return model
    #@tf.function
    def RegressionLoss(self, H, X, gbm = False):
        L1, L2, L3 = [], [], []
        P01, P02, P03 = [], [], []
        H1 = H[0]
        H2 = H[1]
        H3 = H[2]
        strikes = self.price_array[:,0]
        HlistT1 = tf.split(H[0], len(self.strikes), axis = 2)
        HlistT2 = tf.split(H[1], len(self.strikes), axis = 2)
        HlistT3 = tf.split(H[2], len(self.strikes), axis = 2)
        HKT1 = zip(HlistT1, strikes)
        HKT2 = zip(HlistT2, strikes)
        HKT3 = zip(HlistT3, strikes)
        X1 = X[:,:self.maturities[0], :self.regression_features]
        X2 = X[:,:self.maturities[1], :self.regression_features]
        X3 = X[:,:self.maturities[2], :self.regression_features]
        dX1 = X1[:, 1:, :] - X1[:, :-1, :]
        dX2 = X2[:, 1:, :] - X2[:, :-1, :]
        dX3 = X3[:, 1:, :] - X3[:, :-1, :]
        for (H1, K1),(H2, K2),(H3, K3) in zip(HKT1, HKT2, HKT3):
            Yhat1 = tf.reduce_sum(tf.reduce_sum(tf.multiply(H1, dX1), axis = 1), axis = 1)
            # Yhat1 = tf.reduce_sum(tf.multiply(H1, X1[:, :-1, :]), axis = 1)
            p01 = H1[:, 0, :] * X1[:, 0, :]
            P01.append(tf.reduce_mean(p01))
            Y1 = tf.math.maximum(X1[:,-1:, 0] - K1, 0)
            Y1 = tf.squeeze(Y1)
            loss1 =tf.reduce_mean((Y1-Yhat1))
            L1.append(loss1)


            Yhat2 = tf.reduce_sum(tf.reduce_sum(tf.multiply(H2, dX2), axis = 1), axis = 1)
            # Yhat2 = tf.reduce_sum(tf.multiply(H2, X2[:, :-1, :]), axis = 1)
            p02 = H2[:, 0, :] * X2[:, 0, :]
            P02.append(tf.reduce_mean(p02))
            Y2 = tf.math.maximum(X2[:,-1:, 0] - K2, 0)
            Y2 = tf.squeeze(Y1)
            loss2 = tf.reduce_mean((Y2-Yhat2))
            L2.append(loss2)

            # Yhat3 = tf.reduce_sum(tf.multiply(H3, X3[:, :-1, :]), axis = 1)
            Yhat3 = tf.reduce_sum(tf.reduce_sum(tf.multiply(H3, dX3), axis = 1), axis = 1)
            p03 = H2[:, 0, :] * X2[:, 0, :]
            P03.append(tf.reduce_mean(p03))
            Y3 = tf.math.maximum(X3[:,-1:, 0] - K3, 0)
            Y3 = tf.squeeze(Y3)
            loss3 = tf.reduce_mean(Y3-Yhat3)
            L3.append(loss3)
        P01 = tf.convert_to_tensor(P01, dtype = tf.float32)
        P02 = tf.convert_to_tensor(P02, dtype = tf.float32)
        P03 = tf.convert_to_tensor(P03, dtype = tf.float32)
        L1 = tf.convert_to_tensor(L1, dtype = tf.float32)
        L2 = tf.convert_to_tensor(L2, dtype = tf.float32)
        L3 = tf.convert_to_tensor(L3, dtype = tf.float32)
        if gbm:
            K = self.price_array[:, 0]
            T = self.price_array[:, 1] / 365
            market_prices = tf.cast(self.BlackScholes(K, T), tf.float32)
        else:
            market_prices = tf.cast(self.price_array[:, 2], dtype = tf.float32)
        losses = tf.concat([L1, L2, L3], axis = 0)
        replication_loss = tf.reduce_sum(tf.square(losses))
        replication_prices = tf.concat([P01, P02, P03], axis = 0)
        price_error = tf.keras.losses.mse(replication_prices, market_prices)
        return replication_loss, replication_prices, price_error, market_prices
    @tf.function
    def Embedder_train_step(self, X):
        with tf.GradientTape() as tape:
            H = self.Embedder(X, training = True)
            X_tilde = self.Recovery(H, training = True)
            loss = self.EmbedderNetLosst0(X, X_tilde)
        grads = tape.gradient(loss, self.Embedder.trainable_variables + self.Recovery.trainable_variables)
        self.Embedder_optimizer.apply_gradients(zip(grads, self.Embedder.trainable_variables + self.Recovery.trainable_variables))
        return loss
    @tf.function
    def Supervised_Generator_train_step(self, X):
        with tf.GradientTape() as tape:
            H = self.Embedder(X)
            H_hat_supervise = self.Supervisor(H)
            loss = self.GeneratorNet_SupervisedLoss(H, H_hat_supervise)
        grads = tape.gradient(loss, self.Supervisor.trainable_variables)
        self.Supervisor_optimizer.apply_gradients(zip(grads, self.Supervisor.trainable_variables))
        # alternatively add generator vars
        return loss
    @tf.function
    def Generator_train_step(self, X, W):
        with tf.GradientTape(persistent = True) as tape:
            H = self.Embedder(X)
            E_hat = self.Generator(W, training = True)
            H_hat = self.Supervisor(E_hat, training = True)
            H_hat_supervise = self.Supervisor(H, training = True)
            X_hat = self.Recovery(H_hat, training = True)
            y_fake = self.Discriminator(H_hat, training = True)
            y_fake_e = self.Discriminator(E_hat, training = True)
            G_loss_U, _, G_loss_S, _, _, G_loss_V,G_loss = self.GeneratorNetLoss(y_fake, y_fake_e, H, H_hat_supervise, X_hat, X)
            X_tilde = self.Recovery(H, training = True)
            embedder_loss = self.EmbedderNetLosst0(X, X_tilde)
        generator_vars = self.Generator.trainable_variables + self.Supervisor.trainable_variables
        embedder_vars = self.Embedder.trainable_variables + self.Recovery.trainable_variables
        generator_grads = tape.gradient(G_loss, generator_vars)
        embedder_grads = tape.gradient(embedder_loss, embedder_vars)
        return G_loss_U, G_loss_S, G_loss_V, G_loss, embedder_loss
    @tf.function
    def Discriminator_train_step(self, X, W):
        with tf.GradientTape() as tape:
            H = self.Embedder(X, training = True)
            E_hat = self.Generator(W, training = True)
            H_hat = self.Supervisor(E_hat, training = True)
            y_fake = self.Discriminator(H_hat, training = True)
            y_real = self.Discriminator(H, training = True)
            y_fake_e = self.Discriminator(E_hat, training = True)
            _,_,_, discriminator_loss = self.DiscriminatorNetLoss(y_real, y_fake, y_fake_e)
        discriminator_grads = tape.gradient(discriminator_loss, self.Discriminator.trainable_variables)
        self.Discriminator_optimizer.apply_gradients(zip(discriminator_grads, self.Discriminator.trainable_variables))
        return discriminator_loss
    @tf.function
    def Regression_train_step(self, W, gbm = False):
        with tf.GradientTape() as tape:
            E_hat = self.Generator(W, training = True)
            H_hat = self.Supervisor(E_hat, training = True)
            X_hat = self.Recovery(H_hat, training = True)
            X = self.Renormalize_tf(X_hat, gbm)
            eta = self.Regression(X, training = True)
            replication_loss, replication_prices, price_error, market_prices = self.RegressionLoss(eta, X, gbm = gbm)
        regression_grads = tape.gradient(replication_loss, self.Regression.trainable_variables)
        self.Regression_optimizer.apply_gradients(zip(regression_grads, self.Regression.trainable_variables))
        return replication_loss, replication_prices, price_error, market_prices
    @tf.function
    def Transform_train_step(self, W, gbm = False, replication_regularizer = 10, unsupervised_regularizer = 100):
        with tf.GradientTape() as tape:
            E_hat = self.Generator(W, training = True)
            H_hat = self.Supervisor(E_hat, training = True)
            X_hat = self.Recovery(H_hat, training = True)
            y_fake = self.Discriminator(H_hat, training = True)
            y_fake_e = self.Discriminator(E_hat, training = True)
            unsupervised_gen_loss = self.GeneratorNet_UnsupervisedLoss(y_fake, y_fake_e)
            X_renorm = self.Renormalize_tf(X_hat, gbm)
            eta = self.Regression(X_renorm, training = True)
            replication_loss, replication_prices, price_error, market_prices = self.RegressionLoss(eta, X_renorm, gbm = gbm)
            transform_loss = price_error + replication_regularizer * replication_loss  + unsupervised_regularizer * unsupervised_gen_loss
        transform_vars =  self.Generator.trainable_variables + self.Supervisor.trainable_variables \
                            + self.Recovery.trainable_variables + self.Regression.trainable_variables
        transform_grads = tape.gradient(transform_loss, transform_vars)
        self.Regression_optimizer.apply_gradients(zip(transform_grads, transform_vars))
        return replication_loss, replication_prices, price_error, market_prices,transform_loss, unsupervised_gen_loss
    #@tf.function
    def sample_batch(self, data):
        idx = np.random.randint(low = 0, high = data.shape[0], size = self.batch_size)
        idx = tf.cast(idx, tf.int32)
        dta = tf.gather(data, idx, axis = 0)
        dta = tf.cast(dta, tf.float32)
        return dta
    def train(self, model_names = None, save = False, gbm = False):
        if gbm:
            print('Training on simulated GBM data (set False if you want to train on real data)')
            data = self.norm_GBM_data
            save_dir = 'model_saves/GTGAN_GBM'
        else:
            data = self.norm_data
            save_dir = 'model_saves/GTGAN'
        print('Initiating EmbedderNet Pre-training')
        for iteration in range(self.iterations):
            X_batch = self.sample_batch(data)
            _embedder_loss = self.Embedder_train_step(X_batch)
            if iteration % 1000 == 0:
                print('Batch training (embedder) loss at iteration %d: %.4f'% (iteration, float(np.sqrt(_embedder_loss))))
                print('---------------------------------------------------------------')
        for iteration in range(self.iterations):
            X_batch = self.sample_batch(data)
            _generator_loss = self.Supervised_Generator_train_step(X_batch)
            if iteration % 1000 == 0:
                print('Batch training supervised (generator) loss at iteration %d: %.4f'% (iteration, float(np.sqrt(_generator_loss))))
                print('---------------------------------------------------------------')
        print('GeneratorNet Pre-training Complete')
        print('Initiating Joint GAN Pre-Training')
        for iteration in range(self.iterations):
            for kk in range(2):
                X1 = self.sample_batch(data)
                W1 = self.BrownianMotion(X1.shape[0], X1.shape[1], X1.shape[2])
                G_loss_U, G_loss_S, G_loss_V, generator_loss, embedder_loss = self.Generator_train_step(X1, W1)
            X2 = self.sample_batch(data)
            W2 = self.BrownianMotion(X2.shape[0], X2.shape[1], X2.shape[2])
            discriminator_loss = self.Discriminator_train_step(X2, W2)
            if iteration % 1000 == 0:
                print('---------------------------------------------------------------')
                print('Batch training discriminator loss at iteration %d: %.4f'% (iteration, float(discriminator_loss)))
                print('Batch training generator loss at iteration %d: %.4f'% (iteration, float(generator_loss)))
                print('Batch training unsupervised loss at iteration %d: %.4f'% (iteration, float(G_loss_U)))
                print('Batch training supervised loss at iteration %d: %.4f'% (iteration, float(np.sqrt(G_loss_S))))
                print('Batch training embedder loss at iteration %d: %.4f'% (iteration, float(np.sqrt(embedder_loss))))
                print('---------------------------------------------------------------')
        if save:
            print('Saving models under P dynamics')
            model_dirs = [save_dir + '/{}_P1'.format(model_name) for model_name in model_names]
            # print('Saving model to directory: {}'.format(model_name) for model name in model_names)
            self.Generator.save(model_dirs[0])
            self.Supervisor.save(model_dirs[1])
            self.Recovery.save(model_dirs[2])
        print('Initiating Regression training on P dynamics')
        for iteration in range(self.iterations):
            W = self.BrownianMotion(self.batch_size, self.seq_len, self.W_dim)
            replication_loss, replication_prices, price_error, market_prices = self.Regression_train_step(W, gbm)
            if iteration % 1000 == 0:
                print('----------------------------------------------------------------------------')
                print('Over/under-pricing at iteration %d: %.4f'% (iteration, float(replication_loss)))
                print('Pricing error at iteration %d: %.4f'% (iteration, float(price_error)))
                print('Regression prices: {}'.format(replication_prices))
                print('Market prices: {}'.format(market_prices))
                print('----------------------------------------------------------------------------')
        if save:
            print('Saving regression model under P dynamics')
            model_dirs = [save_dir + '/{}_P'.format(model_name) for model_name in model_names]
            self.Regression.save(model_dirs[3])
        print('Initiating Girsanov Transform Training')
        for iteration in range(self.iterations):
            W = self.BrownianMotion(self.batch_size, self.seq_len, self.W_dim)
            replication_loss, replication_prices, price_error, market_prices,transform_loss, unsupervised_gen_loss = self.Transform_train_step(W, gbm)
            if iteration % 1000 == 0:
                print('----------------------------------------------------------------------------')
                print('Transform loss at iteration %d: %.4f'% (iteration, float(transform_loss)))
                print('Unsupervised loss at iteration %d: %.4f'% (iteration, float(unsupervised_gen_loss)))
                print('Replication Error at iteration %d: %.4f'% (iteration, float(replication_loss)))
                print('Pricing error at iteration %d: %.4f'% (iteration, float(price_error)))
                print('Regression prices: {}'.format(replication_prices))
                print('Market prices: {}'.format(market_prices))
                print('----------------------------------------------------------------------------')
        if save:
            print('Saving models under Q dynamics')
            model_dirs = [save_dir + '/{}_Q'.format(model_name) for model_name in model_names]
            self.Generator.save(model_dirs[0])
            self.Supervisor.save(model_dirs[1])
            self.Recovery.save(model_dirs[2])
            self.Regression.save(model_dirs[3])
    def generate_data(self, W, generator, supervisor, recovery):
        E_hat = generator(W, training = True)
        H_hat = supervisor(E_hat, training = True)
        X_hat = recovery(H_hat, training = True)
        return X_hat
    def Simulate_Data(self, W, generator, supervisor, recovery, gbm = False):
        X_hat = self.generate_data(W, generator, supervisor, recovery)
        X_hat = self.Renormalize_tf(X_hat, gbm)
        return X_hat
    def SimulateHedge_Prices(self, W, generator, supervisor, recovery, regression, gbm = False):
        X_hat = self.Simulate_Data(W, generator, supervisor, recovery, gbm)
        eta = self.Regression(X_hat, training = True)
        return X_hat, eta
    def Restore(self, model_names, gbm = False, P = False):
        base_dir = 'model_saves'
        if gbm:
            model_dir = base_dir + '/GTGAN_GBM'
        else:
            model_dir = base_dir + '/GTGAN'
        if P:
            model_names = [name + '_P' for name in model_names]
        else:
            model_names = [name + '_Q' for name in model_names]
        models = [tf.keras.models.load_model(model_dir + '/{}'.format(name), compile = False) for name in model_names]
        return models

if __name__ == '__main__':
    print(device_lib.list_local_devices())
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    symbol_list = ['^GSPC', '^VIX', '^NDX', '^RUT', '^DJI',
                '^FVX','^TNX', '^TYX', 'EURUSD=X','JPY=X']
    settings = {'module':'lstm',
                'hidden_dim': 24,
                'latent_dim': 2,
                'num_layers': 3,
                'iterations': 50,
                'batch_size': 128,
                'learning_rate':0.0001,
                'regression_features': 1,
                'regression_hidden_dim': 70,
                'regression_learning_rate': 0.0001}

    datafolder_path = 'datafolder'
    filename = 'quotedata_SPX.csv'
    filepath = datafolder_path + '/' + filename
    option_data = pd.read_csv(filepath, sep = ';')
    calendar = ql.UnitedStates()
    option_settings = {'start': datetime.date(2020,9,11),
                'day_count': ql.Actual365Fixed(),
                'calendar': ql.UnitedStates(),
                'strikes': [3000, 3100, 3200, 3300, 3400],
                'maturities':[10, 21, 31]}
    Model = GTGAN(symbol_list,settings, option_data ,option_settings)
    model_names = ['Generator', 'Supervisor', 'Recovery', 'Regression']
    Model.train(model_names, save = False, gbm = True)
    # [gen, sup, rec, reg] = Model.Restore(model_names, gbm = True, P = True)
