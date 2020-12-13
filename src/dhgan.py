import tensorflow as tf
from utils import Loader, Option_data
import pandas as pd
import numpy as np
import datetime
import QuantLib as ql
import matplotlib.pyplot as plt
from tensorflow.keras.layers import TimeDistributed
from tensorflow.python.client import device_lib

class DHGAN(Loader, Option_data):
    def __init__(self, symbol_list, settings, dh_settings, option_data, option_settings):
        Loader.__init__(self, symbol_list = symbol_list)
        Option_data.__init__(self, option_data ,option_settings)
        self.settings = settings
        self.dh_settings = dh_settings
        self.input_data = np.asarray(self.data)
        self.module = self.settings['module']
        self.hidden_dim = self.settings['hidden_dim']
        self.latent_dim = self.settings['latent_dim']
        self.num_layers = self.settings['num_layers']
        self.iterations = self.settings['iterations']
        self.batch_size = self.settings['batch_size']
        self.learning_rate = self.settings['learning_rate']
        self.no, self.seq_len, self.dim = self.input_data.shape
        # self.time, self.max_seq_len = extract_time(self.input_data)
        self.dh_sims = self.dh_settings['hedge_simulations']
        self.dh_features = self.dh_settings['hedge_features']
        self.dh_epochs = self.dh_settings['hedge_epochs']
        self.dh_batch_size = self.dh_settings['hedge_batch_size']
        self.dh_neurons = self.dh_settings['hedge_neurons']
        self.dh_learning_rate = self.dh_settings['hedge_learning_rate']
        self.dh_validation_proportion = self.dh_settings['hedge_validation_proportion']
        self.dh_test_proportion = self.dh_settings['hedge_test_proportion']
        self.dh_nb_test = int(self.dh_test_proportion * self.dh_sims)
        self.dh_nb_valid = int(self.dh_test_proportion * self.dh_sims)
        self.dh_nb_train = int(self.dh_sims - (self.dh_nb_test+ self.dh_nb_valid))
        self.alpha = self.dh_settings['alpha']
        self.epsilon = self.dh_settings['epsilon']
        self.K = self.dh_settings['strike']
        self.T = self.seq_len
        self.W_dim = self.dim
        # models
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
        self.dhmodel = self.DeepHedgeNet()
        self.dhmodel_optimizer = tf.keras.optimizers.Adam(self.dh_learning_rate)
        self.dhmodel.summary()
        # Training datasets


        #renorm
        # self.dta = self.norm_data * self.max_val
        # self.dta = self.dta + self.min_val
        # self.dta = np.exp(self.dta)
        # print(self.dta[:, :,:] * np.asarray(self.starting_values))
        # print(self.starting_values)

    def BrownianMotion(self,nb_sims, timesteps, W_dim, dt = 1/365):
        W = np.zeros(shape = (nb_sims, timesteps, W_dim))
        for t in range(1, timesteps):
            W[:,t,:] = W[:,t-1,:] + np.random.randn(nb_sims, W_dim)* np.sqrt(dt)
        return W
    def EmbedderNet(self):
        '''
        Latent space map from function space X (bounded random processes F-adapted)
        to target latent space representation
        '''
        model = tf.keras.Sequential(name = 'EmbedderNet')
        if self.module == 'lstm':
            model.add(tf.keras.layers.LSTM(self.dim, input_shape = (self.T, self.dim), activation = 'tanh',return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.LSTM(self.hidden_dim, activation = 'tanh', return_sequences = True))
        elif self.module == 'gru':
            model.add(tf.keras.layers.GRU(self.dim, input_shape = (self.T, self.dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.GRU(self.hidden_dim, activation = 'tanh', return_sequences = True))
        else:
            raise AssertionError('Please Choose gru or lstm as module')
        model.add(TimeDistributed(tf.keras.layers.Dense(self.latent_dim, activation = 'sigmoid')))
        return model
    def RecoveryNet(self):
        '''
        Inverse map from latent (HxT) space representation to state space of bounded F-adapted random processes
        '''
        model = tf.keras.Sequential(name = 'RecoveryNet')
        if self.module == 'lstm':
            model.add(tf.keras.layers.LSTM(self.hidden_dim, input_shape = (self.T, self.latent_dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.LSTM(self.hidden_dim, activation = 'tanh', return_sequences = True))
        elif self.module == 'gru':
            model.add(tf.keras.layers.GRU(self.hidden_dim, input_shape = (self.T, self.latent_dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.GRU(self.hidden_dim, activation = 'tanh', return_sequences = True))
        model.add(TimeDistributed(tf.keras.layers.Dense(self.dim, activation = 'sigmoid')))
        return model
    def SupervisorNet(self):
        '''
        map from latent space (HxT)to latent space: Generate next sequence based on previous
        '''
        model = tf.keras.Sequential(name = 'SupervisorNet')
        if self.module == 'lstm':
            model.add(tf.keras.layers.LSTM(self.hidden_dim, input_shape = (self.T, self.latent_dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.LSTM(self.hidden_dim, activation = 'tanh', return_sequences = True))
        elif self.module == 'gru':
            model.add(tf.keras.layers.GRU(self.hidden_dim, input_shape = (self.T, self.latent_dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.GRU(self.hidden_dim, activation = 'tanh', return_sequences = True))
        model.add(TimeDistributed(tf.keras.layers.Dense(self.latent_dim, activation = 'sigmoid')))
        return model
    def GeneratorNet(self):
        '''
        Map from d dimensional Brownian motion (W(omega)xT) to latent space
        '''
        model = tf.keras.Sequential(name = 'GeneratorNet')
        if self.module == 'lstm':
            model.add(tf.keras.layers.LSTM(self.hidden_dim, input_shape = (self.T, self.dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.LSTM(self.hidden_dim, activation = 'tanh', return_sequences = True))
        elif self.module == 'gru':
            model.add(tf.keras.layers.GRU(self.hidden_dim, input_shape = (self.T, self.dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.GRU(self.hidden_dim, activation = 'tanh', return_sequences = True))
        model.add(TimeDistributed(tf.keras.layers.Dense(self.latent_dim, activation = 'sigmoid')))
        return model
    def DiscriminatorNet(self):
        '''
        Map from latent space (HxT) representation to [0,1] classification

        '''
        model = tf.keras.Sequential(name = 'DiscriminatorNet')
        if self.module == 'lstm':
            model.add(tf.keras.layers.LSTM(self.hidden_dim, input_shape = (self.T, self.latent_dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.LSTM(self.hidden_dim, activation = 'tanh', return_sequences = True))
        elif self.module == 'gru':
            model.add(tf.keras.layers.GRU(self.hidden_dim, input_shape = (self.T, self.latent_dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.GRU(self.hidden_dim, activation = 'tanh', return_sequences = True))
        model.add(TimeDistributed(tf.keras.layers.Dense(self.latent_dim, activation = 'linear')))
        return model
    def DeepHedgeNet(self):
        inp = tf.keras.layers.Input(shape = (self.seq_len, self.dim))
        if self.module == 'lstm':
            X = tf.keras.layers.LSTM(self.dh_neurons, return_sequences = False)(inp)
            X = tf.keras.layers.BatchNormalization()(X)
            T1 = tf.keras.layers.RepeatVector(self.maturities[0]-1)(X)
            T2 = tf.keras.layers.RepeatVector(self.maturities[1]-1)(X)
            T3 = tf.keras.layers.RepeatVector(self.maturities[2]-1)(X)
            T1 = tf.keras.layers.LSTM(self.dh_neurons, activation = 'elu', return_sequences = True)(T1)
            T2 = tf.keras.layers.LSTM(self.dh_neurons, activation = 'elu', return_sequences = True)(T2)
            T3 = tf.keras.layers.LSTM(self.dh_neurons, activation = 'elu', return_sequences = True)(T3)
        elif self.module == 'gru':
            X = tf.keras.layers.GRU(self.dh_neurons, return_sequences = False)(inp)
            X = tf.keras.layers.BatchNormalization()(X)
            T1 = tf.keras.layers.RepeatVector(self.maturities[0]-1)(X)
            T2 = tf.keras.layers.RepeatVector(self.maturities[1]-1)(X)
            T3 = tf.keras.layers.RepeatVector(self.maturities[2]-1)(X)
            T1 = tf.keras.layers.GRU(self.dh_neurons, activation = 'elu', return_sequences = True)(T1)
            T2 = tf.keras.layers.GRU(self.dh_neurons, activation = 'elu', return_sequences = True)(T2)
            T3 = tf.keras.layers.GRU(self.dh_neurons, activation = 'elu', return_sequences = True)(T3)
        T1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.dh_features * len(self.strikes), activation = 'linear'))(T1)
        T2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.dh_features * len(self.strikes), activation = 'linear'))(T2)
        T3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.dh_features * len(self.strikes), activation = 'linear'))(T3)
        model = tf.keras.Model(inputs = inp, outputs = [T1, T2, T3], name = 'DeepHedgeNet')
        return model
    @tf.function
    def EmbedderNetLosst0(self, X, X_tilde):
        E_loss_t0 = tf.compat.v1.losses.mean_squared_error(X, X_tilde)
        return E_loss_t0
    @tf.function
    def EmbedderNetLoss(self, X, X_tilde, G_loss_S):
        E_loss_t0 = self.EmbedderNetLosst0(X, X_tilde)
        E_loss_0 = 10*tf.sqrt(E_loss_t0)
        E_loss = E_loss0 + 0.1 * G_loss_S
        return E_loss_t0, E_loss_0, E_loss
    def RecoveryNetLoss(self):
        pass
    def SupervisorNetLoss(self):
        pass
    @tf.function
    def GeneratorNetLoss(self, y_fake, y_fake_e, H, H_hat_supervise, X_hat, X, gamma = 1):
        G_loss_U = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(y_fake), y_fake)
        G_loss_U_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(y_fake_e), y_fake_e)
        G_loss_S = self.GeneratorNet_SupervisedLoss(H, H_hat_supervise)
        G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
        G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
        G_loss_V = G_loss_V1 + G_loss_V2
        G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V
        return G_loss_U, G_loss_U_e, G_loss_S, G_loss_V1, G_loss_V2, G_loss_V, G_loss
    @tf.function
    def GeneratorNet_SupervisedLoss(self, H, H_hat_supervise):
        G_loss_S = tf.compat.v1.losses.mean_squared_error(H[:, 1:, :], H_hat_supervise[:, :-1, :])
        return G_loss_S
    @tf.function
    def DiscriminatorNetLoss(self, y_real,y_fake, y_fake_e, gamma = 1):
        D_loss_real = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(y_real), y_real)
        D_loss_fake = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(y_fake), y_real)
        D_loss_fake_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(y_fake_e), y_fake_e)
        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
        return D_loss_real, D_loss_fake, D_loss_fake_e, D_loss
    @tf.function
    def ReplicationPortfolio(self, X, H):
        dX = X[:, 1:, :self.dh_features] - X[:, :-1, :self.dh_features]
        discrete_integral = tf.reduce_sum(tf.multiply(H, dX), axis = 1)
        trading_strategy = tf.reduce_sum(discrete_integral, axis = 1)
        return trading_strategy
    @tf.function
    def TransactionCosts(self, epsilon, X, H):
        X = X[:, :, :self.dh_features]
        zeros = tf.zeros(shape = (H.shape[0], 1, H.shape[2]))
        H = tf.concat([zeros, H, zeros], axis = 1)
        dH = H[:,1:,:]- H[:,:-1, :]
        dH = tf.abs(dH)
        transaction_costs = tf.multiply(tf.reduce_sum(tf.reduce_sum(tf.multiply(X, dH), axis = 1), axis = 1), epsilon)
        return transaction_costs
    @tf.function
    def DeepHedgeNetLoss1(self, y, trading_strategy, transaction_costs):
        loss,_ =tf.math.top_k(-(-tf.squeeze(y) + trading_strategy - transaction_costs), tf.cast(self.alpha * self.batch_size, tf.int32))
        loss = tf.reduce_mean(tf.abs(loss))
        return loss
    @tf.function
    def DeepHedgeNetLoss2(self, y, trading_strategy, transaction_costs):
        loss,_ =tf.math.top_k(-(-tf.squeeze(y) + trading_strategy - transaction_costs), tf.cast(self.alpha * self.batch_size, tf.int32))
        loss = tf.reduce_mean(loss)
        return loss
    @tf.function
    def DHTermStructureLoss(self, H, X, loss_type = 'type2'):
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
        X1 = X[:,:self.maturities[0], :]
        X2 = X[:,:self.maturities[1], :]
        X3 = X[:,:self.maturities[2], :]
        for (H1, K1),(H2, K2),(H3, K3) in zip(HKT1, HKT2, HKT3):
            strategy1 = self.ReplicationPortfolio(X1, H1)
            cost1 = self.TransactionCosts(self.epsilon, X1, H1)
            Y1 = tf.math.maximum(X1[:,-1:, 0] - K1, 0)
            loss1 = self.DeepHedgeNetLoss1(Y1, strategy1, cost1)
            p01 = strategy1 - cost1
            P01.append(p01[0, 0])
            L1.append(loss1)

            strategy2 = self.ReplicationPortfolio(X2, H2)
            cost2 = self.TransactionCosts(self.epsilon, X2, H2)
            Y2 = tf.math.maximum(X2[:,-1:, 0] - K2, 0)
            loss2 = self.DeepHedgeNetLoss1(Y2, strategy2, cost2)
            p02 = strategy2 - cost2
            P02.append(p02[0, 0])
            L2.append(loss2)

            strategy3 = self.ReplicationPortfolio(X3, H3)
            cost3 = self.TransactionCosts(self.epsilon, X3, H3)
            Y3 = tf.math.maximum(X3[:,-1:, 0] - K3, 0)
            loss3 = self.DeepHedgeNetLoss1(Y3, strategy3, cost3)
            p03 = strategy3 - cost3
            P03.append(p03[0, 0])
            L3.append(loss3)
        P01 = tf.convert_to_tensor(P01, dtype = tf.float32)
        P02 = tf.convert_to_tensor(P02, dtype = tf.float32)
        P03 = tf.convert_to_tensor(P03, dtype = tf.float32)
        L1 = tf.convert_to_tensor(L1, dtype = tf.float32)
        L2 = tf.convert_to_tensor(L2, dtype = tf.float32)
        L3 = tf.convert_to_tensor(L3, dtype = tf.float32)
        market_prices = tf.cast(self.price_array[:, 2], dtype = tf.float32)
        if loss_type == 'type1':
            replication_prices = tf.concat([P01, P02, P03], axis = 0)
            replication_loss = tf.keras.losses.mse(replication_prices, market_prices)
            price_error = None
        elif loss_type == 'type2':
            replication_prices = tf.concat([L1, L2, L3], axis = 0)
            replication_loss = tf.keras.losses.mse(replication_prices, market_prices)
            price_error = None
        elif loss_type == 'type3':
            losses = tf.concat([L1, L2, L3], axis = 0)
            replication_loss = tf.reduce_mean(tf.square(losses))
            replication_prices = tf.concat([P01, P02, P03], axis = 0)
            price_error = tf.keras.losses.mse(replication_prices, market_prices)
        else:
            raise NotImplementedError
        return replication_loss, replication_prices, price_error
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
    def Supervised_Generator_train_step(self, X, W):
        with tf.GradientTape() as tape:
            E_hat = self.Generator(W)
            H = self.Embedder(X)
            H_hat_supervise = self.Supervisor(H)
            loss = self.GeneratorNet_SupervisedLoss(H, H_hat_supervise)
        grads = tape.gradient(loss, self.Supervisor.trainable_variables + self.Generator.trainable_variables)
        self.Supervisor_optimizer.apply_gradients(zip(grads, self.Supervisor.trainable_variables + self.Generator.trainable_variables))
        return loss
    @tf.function
    def Unsupervised_Generator_train_step(self, X1, W1):
        with tf.GradientTape(persistent = True) as tape:
            H = self.Embedder(X1)
            E_hat = self.Generator(W1, training = True)
            H_hat_supervise = self.Supervisor(H)
            H_hat = self.Supervisor(E_hat)
            X_hat = self.Recovery(H_hat)
            y_fake = self.Discriminator(H_hat)
            y_fake_e = self.Discriminator(E_hat)
            G_loss_U, _, G_loss_S, _, _, G_loss_V,G_loss = self.GeneratorNetLoss(y_fake, y_fake_e, H, H_hat_supervise, X_hat, X1)
            # Embedder training
            X_tilde = self.Recovery(H, training = True)
            embedder_loss = self.EmbedderNetLosst0(X1, X_tilde)
        generator_grads = tape.gradient(G_loss, self.Generator.trainable_variables + self.Supervisor.trainable_variables)
        self.Generator_optimizer.apply_gradients(zip(generator_grads, self.Generator.trainable_variables + self.Supervisor.trainable_variables))
        embedder_grads = tape.gradient(embedder_loss, self.Embedder.trainable_variables + self.Recovery.trainable_variables)
        self.Embedder_optimizer.apply_gradients(zip(embedder_grads, self.Embedder.trainable_variables + self.Recovery.trainable_variables))
        return G_loss_U, G_loss_S, G_loss_V, G_loss, embedder_loss
    @tf.function
    def Discriminator_train_step(self, X2, W2):
        with tf.GradientTape() as tape:
            H2 = self.Embedder(X2)
            E_hat2 = self.Generator(W2)
            H_hat2 = self.Supervisor(E_hat2)
            y_fake2 = self.Discriminator(H_hat2)
            y_real2 = self.Discriminator(H2)
            y_fake_e2 = self.Discriminator(E_hat2)
            _,_,_, discriminator_loss = self.DiscriminatorNetLoss(y_fake2, y_real2, y_fake_e2)
        if (discriminator_loss > 0.15):
            discriminator_grads = tape.gradient(discriminator_loss, self.Discriminator.trainable_variables)
            self.Discriminator_optimizer.apply_gradients(zip(discriminator_grads, self.Discriminator.trainable_variables))
        return discriminator_loss
    @tf.function
    def DH_train_step(self, W):
        with tf.GradientTape()  as tape:
            E_hat = self.Generator(W, training = True)
            H_hat = self.Supervisor(E_hat, training = True)
            X_hat = self.Recovery(H_hat, training = True)
            X_renorm = self.Renormalize_tf(X_hat)
            hedge_weights = self.dhmodel(X_renorm, training = True)
            termstructure_replication_loss,replication_prices, price_error  = self.DHTermStructureLoss(hedge_weights, X_renorm, loss_type = 'type3')
        dh_grads = tape.gradient(termstructure_replication_loss, self.dhmodel.trainable_variables)
        self.dhmodel_optimizer.apply_gradients(zip(dh_grads, self.dhmodel.trainable_variables))
        return termstructure_replication_loss, replication_prices, price_error
    @tf.function
    def DHGAN_train_step(self,X, W):
        with tf.GradientTape(persistent = True)  as tape:
            H = self.Embedder(X, training = True)
            E_hat = self.Generator(W, training = True)
            H_hat_supervise = self.Supervisor(H, training = True)
            H_hat = self.Supervisor(E_hat, training = True)
            X_hat = self.Recovery(H_hat, training = True)
            y_fake = self.Discriminator(H_hat, training = True)
            y_fake_e = self.Discriminator(E_hat, training = True)
            G_loss_U, _, G_loss_S, _, _, G_loss_V,G_loss = self.GeneratorNetLoss(y_fake, y_fake_e, H, H_hat_supervise, X_hat, X)
            X_tilde = self.Recovery(H, training = True)
            embedder_loss = self.EmbedderNetLosst0(X, X_tilde)
            X_renorm = self.Renormalize_tf(X_hat)
            # X_renorm = X_renorm[:,:,:self.dh_features]
            hedge_weights = self.dhmodel(X_renorm, training = True)
            termstructure_replication_loss,replication_prices ,price_error  = self.DHTermStructureLoss(hedge_weights, X_renorm, loss_type = 'type3')
        generator_grads = tape.gradient(G_loss, self.Generator.trainable_variables + self.Supervisor.trainable_variables)
        self.Generator_optimizer.apply_gradients(zip(generator_grads, self.Generator.trainable_variables + self.Supervisor.trainable_variables))
        embedder_grads = tape.gradient(embedder_loss, self.Embedder.trainable_variables + self.Recovery.trainable_variables)
        self.Embedder_optimizer.apply_gradients(zip(embedder_grads, self.Embedder.trainable_variables + self.Recovery.trainable_variables))
        dhvars = self.Generator.trainable_variables + self.Supervisor.trainable_variables + self.Recovery.trainable_variables + self.dhmodel.trainable_variables
        dh_grads = tape.gradient(termstructure_replication_loss,dhvars)
        self.dhmodel_optimizer.apply_gradients(zip(dh_grads, dhvars))
        return termstructure_replication_loss, replication_prices,price_error ,G_loss, embedder_loss

    def train(self, model_names = None, save = False):
        save_dir = 'model_saves/DHGAN'
        print('Initiating EmbedderNet Pre-training')
        for iteration in range(self.iterations):
            X_batch = self.sample_batch(self.norm_data)
            _embedder_loss = self.Embedder_train_step(X_batch)
            if iteration % 1000 == 0:
                print('Batch training (embedder) loss at iteration %d: %.4f'% (iteration, float(np.sqrt(_embedder_loss))))
                print('---------------------------------------------------------------')
        print('EmbedderNet Pre-training Complete')
        print('Initiating Generator Pre-training (Supervised Loss Only)')
        for iteration in range(self.iterations):
            X_batch = self.sample_batch(self.norm_data)
            W_batch = self.BrownianMotion(X_batch.shape[0],X_batch.shape[1], X_batch.shape[2])
            _generator_loss = self.Supervised_Generator_train_step(X_batch, W_batch)
            if iteration % 1000 == 0:
                print('Batch training supervised (generator) loss at iteration %d: %.4f'% (iteration, float(np.sqrt(_generator_loss))))
                print('---------------------------------------------------------------')
        print('GeneratorNet Pre-training Complete')
        print('Initiating Joint GAN Pre-Training')
        for iteration in range(self.iterations):
            for kk in range(2):
                X1 = self.sample_batch(self.norm_data)
                W1 = self.BrownianMotion(X1.shape[0], X1.shape[1], X1.shape[2])
                G_loss_U, G_loss_S, G_loss_V, generator_loss, embedder_loss = self.Unsupervised_Generator_train_step(X1, W1)
            X2 = self.sample_batch(self.norm_data)
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
            model_dirs = [save_dir + '/{}'.format(model_name) for model_name in model_names]
            # print('Saving model to directory: {}'.format(model_name) for model name in model_names)
            self.Generator.save(model_dirs[0])
            self.Supervisor.save(model_dirs[1])
            self.Recovery.save(model_dirs[2])
        print('Joint GAN Pre-Training Complete')
        print('Initiating Deep Hedge Pre-Training')
        for iteration in range(self.iterations):
            W = self.BrownianMotion(self.batch_size, self.seq_len, self.W_dim)
            termstructure_replication_loss, replication_prices, price_error = self.DH_train_step(W)
            if iteration % 1000 == 0:
                print('---------------------------------------------------------------')
                print('Batch training termstructure replication loss at iteration %d: %.4f'% (iteration, float(termstructure_replication_loss)))
                print('Batch training pricing error at iteration %d: %.4f'% (iteration, float(price_error)))
                print('Batch training termstructure replication prices: {}'.format(replication_prices))
                print('---------------------------------------------------------------')
        if save:
            model_dirs = [save_dir + '/{}'.format(model_name) for model_name in model_names]
            self.dhmodel.save(model_dirs[3])
        print('Deep Hedge Pre-Training Complete')
        print('Initiating Joint DHGAN training')
        for iteration in range(self.iterations):
            for kk in range(2):
                X1 = self.sample_batch(self.norm_data)
                W1 = self.BrownianMotion(X1.shape[0], X1.shape[1], X1.shape[2])
                termstructure_replication_loss, replication_prices, price_error, generator_loss, embedder_loss = self.DHGAN_train_step(X1, W1)
            X2 = self.sample_batch(self.norm_data)
            W2 = self.BrownianMotion(X2.shape[0], X2.shape[1], X2.shape[2])
            discriminator_loss = self.Discriminator_train_step(X2, W2)
            if iteration % 1000 == 0:
                print('---------------------------------------------------------------')
                print('Batch training discriminator loss at iteration %d: %.4f'% (iteration, float(discriminator_loss)))
                print('Batch training generator loss at iteration %d: %.4f'% (iteration, float(generator_loss)))
                print('Batch training embedder loss at iteration %d: %.4f'% (iteration, float(np.sqrt(embedder_loss))))
                print('Batch training termstructure replication loss at iteration %d: %.4f'% (iteration, float(termstructure_replication_loss)))
                print('Batch training pricing error at iteration %d: %.4f'% (iteration, float(price_error)))
                print('Batch training termstructure replication prices: {}'.format(replication_prices))
                market_prices = tf.cast(self.price_array[:, 2], dtype = tf.float32)
                print('Actual Prices: {}'.format(market_prices))
                print('---------------------------------------------------------------')
        if save:
            model_dirs = [save_dir + '/{}'.format(model_name) for model_name in model_names]
            # print('Saving model to directory: {}'.format(model_name) for model name in model_names)
            self.Generator.save(model_dirs[0])
            self.Supervisor.save(model_dirs[1])
            self.Recovery.save(model_dirs[2])
            self.dhmodel.save(model_dirs[3])


    def sample_batch(self, data):
        idx = np.random.randint(low = 0, high = data.shape[0], size = self.batch_size)
        dta = data[idx, :, :]
        dta = tf.cast(dta, tf.float32)
        return dta
    def generate_data(self, W, generator, supervisor, recovery):
        E = generator(W, training = False)
        H = supervisor(E, training = False)
        X = recovery(H, training = False)
        return X
    def Renormalize(self, X):
        X = X * (self.max_val + 1e-7)
        X = X + self.min_val
        X = np.exp(X)
        X = X * np.asarray(self.starting_values)
        return X
    @tf.function
    def Renormalize_tf(self, X):
        max_val = tf.cast(self.max_val + 1e-7, tf.float32)
        min_val = tf.cast(self.min_val, tf.float32)
        X = X * max_val
        X = X + min_val
        X = tf.math.exp(X)
        X = X * tf.convert_to_tensor(np.asarray(self.starting_values), dtype = tf.float32)
        return X
    def Simulate_Hedge(self, W, generator, supervisor, recovery, dhmodel):
        X = self.generate_data(W, generator, supervisor, recovery)
        X = self.Renormalize(X)
        H = dhmodel(X)
        return X, H
    def Hedge_on_data(self, data):
        pass
    def Simulate_Data(self, W, generator, supervisor, recovery):
        X = self.generate_data(W, generator, supervisor, recovery)
        X = self.Renormalize(X)
        return X
    def restore_model(self, model_name):
        save_dir = 'model_saves/DHGAN'
        model = tf.keras.models.load_model(save_dir + '/{}'.format(model_name))
        return model
    def RestoreGAN(self, model_names):
        save_dir = 'model_saves/DHGAN'
        models = [tf.keras.models.load_model(save_dir + '/{}'.format(model_name)) for model_name in model_names]
        return models[0], models[1], models[2]
    def RestoreDHGAN(self, model_names):
        save_dir = 'model_saves/DHGAN'
        models = [tf.keras.models.load_model(save_dir + '/{}'.format(model_name)) for model_name in model_names]
        return models[0], models[1], models[2], models[3]
if __name__ == '__main__':
    print(device_lib.list_local_devices())
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    symbol_list = ['^GSPC', '^VIX', '^NDX', '^RUT', '^DJI',
                '^FVX','^TNX', '^TYX', 'EURUSD=X','JPY=X']
    settings = {'module':'lstm',
                'hidden_dim':24,
                'latent_dim': 5,
                'num_layers': 3,
                'iterations': 30000,
                'batch_size': 100,
                'learning_rate':0.001}
    DH_settings = {'hedge_simulations': 1000000,
                    'hedge_features': 2,
                    'hedge_epochs': 2,
                    'hedge_batch_size': 1000,
                    'hedge_neurons': 70,
                    'hedge_learning_rate': 0.00001,
                    'hedge_validation_proportion':0.1,
                    'hedge_test_proportion':0.001,
                    'alpha':0.99,
                    'epsilon':0.0,
                    'strike':3200}
    datafolder_path = 'datafolder'
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
    Model = DHGAN(symbol_list,settings, DH_settings, option_data ,option_settings)

    model_names = ['Generator', 'Supervisor', 'Recovery', 'dhmodel']
    Model.train(model_names, save = True)
    gen, sup, rec, dh = Model.RestoreDHGAN(model_names)
    W = Model.BrownianMotion(100, 31, 10)
    X, H = Model.Simulate_Hedge(W, gen, sup, rec, dh)
    print(X)
    print('aldkfjalskjfd')
    print(H)
    for i in range(15):
        plt.plot(X[i, :,0])
    plt.show()
    # Model.test(Model.Generator_data)
