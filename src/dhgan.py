import tensorflow as tf
from utils import Loader, Option_data
import pandas as pd
import numpy as np
import datetime
import QuantLib as ql
import matplotlib.pyplot as plt
from tensorflow.keras.layers import TimeDistributed
class DHGAN(Loader, Option_data):
    def __init__(self, symbol_list, settings, dh_settings, option_data, option_settings):
        Loader.__init__(self, symbol_list = symbol_list)
        Option_data.__init__(self, option_data ,option_settings)
        self.settings = settings
        self.dh_settings = dh_settings
        self.input_data = np.asarray(self.data)
        self.module = self.settings['module']
        self.hidden_dim = self.settings['hidden_dim']
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
        self.norm_strike = (np.log(self.K) - self.min_val[0])/(self.max_val[0] + 1e-7)
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
        tmp = tf.zeros(shape = (self.norm_data.shape[0], self.norm_data.shape[1], 1))
        self.Embedder_data, self.Generator_data, self.Joint_data = self.BuildPipeline(self.norm_data, tmp)

        # renorm
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
    def BuildPipeline(self, X, y):
        nb_sims, timesteps, W_dim = X.shape
        W = self.BrownianMotion(nb_sims,timesteps, W_dim)
        W1 = self.BrownianMotion(nb_sims,timesteps, W_dim)
        W2 = self.BrownianMotion(nb_sims,timesteps, W_dim)
        X = tf.cast(X, tf.float32)
        y = tf.cast(y, tf.float32)
        embedder_dataset = tf.data.Dataset.from_tensor_slices((X, y))
        embedder_dataset = embedder_dataset.shuffle(buffer_size = nb_sims).batch(self.batch_size, drop_remainder = True)
        generator_dataset = tf.data.Dataset.from_tensor_slices((X, W))
        generator_dataset = generator_dataset.shuffle(buffer_size = nb_sims).batch(self.batch_size, drop_remainder = True)
        joint_dataset1 = tf.data.Dataset.from_tensor_slices((X, W1))
        joint_dataset2 = tf.data.Dataset.from_tensor_slices((X, W2))
        joint_dataset1 = joint_dataset1.shuffle(buffer_size = nb_sims, reshuffle_each_iteration = True).batch(self.batch_size, drop_remainder = True)
        joint_dataset2 = joint_dataset2.shuffle(buffer_size = nb_sims, reshuffle_each_iteration = True).batch(self.batch_size, drop_remainder = True)
        joint_dataset = tf.data.Dataset.zip((joint_dataset1, joint_dataset2))
        return embedder_dataset, generator_dataset, joint_dataset
    def PostProcessGenerated_data(self, X):
        features = X[:,:self.seq_len, :self.dh_features]
        labels = tf.math.maximum(features[:,-1:,0] - self.norm_strike, 0)
        return features, labels
    def BuildPipelineDH(self, X, y):
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size = 10000).batch(batch_size = self.dh_batch_size, drop_remainder = True)
        return dataset
    def PreProcessTermStructure(self):
        termstructure = self.price_array
        print(termstructure.shape)

    def EmbedderNet(self):
        '''
        Latent space map from function space X (bounded random processes F-adapted)
        to target latent space representation
        '''
        model = tf.keras.Sequential(name = 'EmbedderNet')
        if self.module == 'lstm':
            model.add(tf.keras.layers.LSTM(self.dim, input_shape = (self.T, self.dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.LSTM(self.hidden_dim, activation = 'tanh', return_sequences = True))
        elif self.module == 'gru':
            model.add(tf.keras.layers.GRU(self.dim, input_shape = (self.T, self.dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.GRU(self.hidden_dim, activation = 'tanh', return_sequences = True))
        else:
            raise AssertionError('Please Choose gru or lstm as module')
        model.add(TimeDistributed(tf.keras.layers.Dense(self.hidden_dim, activation = 'sigmoid')))
        return model
    def RecoveryNet(self):
        '''
        Inverse map from latent (HxT) space representation to state space of bounded F-adapted random processes
        '''
        model = tf.keras.Sequential(name = 'RecoveryNet')
        if self.module == 'lstm':
            model.add(tf.keras.layers.LSTM(self.hidden_dim, input_shape = (self.T, self.hidden_dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.LSTM(self.hidden_dim, activation = 'tanh', return_sequences = True))
        elif self.module == 'gru':
            model.add(tf.keras.layers.GRU(self.hidden_dim, input_shape = (self.T, self.hidden_dim), activation = 'tanh', return_sequences = True))
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
            model.add(tf.keras.layers.LSTM(self.hidden_dim, input_shape = (self.T, self.hidden_dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.LSTM(self.hidden_dim, activation = 'tanh', return_sequences = True))
        elif self.module == 'gru':
            model.add(tf.keras.layers.GRU(self.hidden_dim, input_shape = (self.T, self.hidden_dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.GRU(self.hidden_dim, activation = 'tanh', return_sequences = True))
        model.add(TimeDistributed(tf.keras.layers.Dense(self.hidden_dim, activation = 'sigmoid')))
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
        model.add(TimeDistributed(tf.keras.layers.Dense(self.hidden_dim, activation = 'sigmoid')))
        return model
    def DiscriminatorNet(self):
        '''
        Map from latent space (HxT) representation to [0,1] classification

        '''
        model = tf.keras.Sequential(name = 'DiscriminatorNet')
        if self.module == 'lstm':
            model.add(tf.keras.layers.LSTM(self.hidden_dim, input_shape = (self.T, self.hidden_dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.LSTM(self.hidden_dim, activation = 'tanh', return_sequences = True))
        elif self.module == 'gru':
            model.add(tf.keras.layers.GRU(self.hidden_dim, input_shape = (self.T, self.hidden_dim), activation = 'tanh', return_sequences = True))
            for layer in range(self.num_layers):
                model.add(tf.keras.layers.GRU(self.hidden_dim, activation = 'tanh', return_sequences = True))
        model.add(TimeDistributed(tf.keras.layers.Dense(self.hidden_dim, activation = 'linear')))
        return model
    def DeepHedgeNet(self):
        model = tf.keras.Sequential(name = 'DeepHedgeNet')
        if self.module == 'lstm':
            model.add(tf.keras.layers.LSTM(self.dh_neurons, activation = None, input_shape = (self.seq_len, self.dh_features)))
            model.add(tf.keras.layers.RepeatVector(self.seq_len -1))
            model.add(tf.keras.layers.LSTM(self.dh_neurons, activation = None, recurrent_activation = 'sigmoid', return_sequences = True, recurrent_dropout = 0.2))
            model.add(tf.keras.layers.LSTM(self.dh_neurons, activation = None, recurrent_activation = 'sigmoid',return_sequences = True))
        elif self.module == 'gru':
            model.add(tf.keras.layers.GRU(self.dh_neurons, activation = None, input_shape = (self.seq_len, self.dh_features)))
            model.add(tf.keras.layers.RepeatVector(self.seq_len -1))
            model.add(tf.keras.layers.GRU(self.dh_neurons, activation = None, recurrent_activation = 'sigmoid', return_sequences = True, recurrent_dropout = 0.2))
            model.add(tf.keras.layers.GRU(self.dh_neurons, activation = None, recurrent_activation = 'sigmoid',return_sequences = True))
            model.add(TimeDistributed(tf.keras.layers.Dense(self.dh_features, activation = 'linear')))
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
    def GeneratorNet_SupervisedLoss(self, H, H_hat_supervise):
        G_loss_S = tf.compat.v1.losses.mean_squared_error(H[:, 1:, :], H_hat_supervise[:, :-1, :])
        return G_loss_S
    def DiscriminatorNetLoss(self, y_real,y_fake, y_fake_e, gamma = 1):
        D_loss_real = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(y_real), y_real)
        D_loss_fake = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(y_fake), y_real)
        D_loss_fake_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(y_fake_e), y_fake_e)
        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
        return D_loss_real, D_loss_fake, D_loss_fake_e, D_loss
    @tf.function
    def ReplicationPortfolio(self, X, H):
        dX = X[:, 1:, :] - X[:, :-1, :]
        discrete_integral = tf.reduce_sum(tf.multiply(H, dX), axis = 1)
        trading_strategy = tf.reduce_sum(discrete_integral, axis = 1)
        return trading_strategy
    @tf.function
    def TransactionCosts(self, epsilon, X, H):
        zeros = tf.zeros(shape = (H.shape[0], 1, H.shape[2]))
        H = tf.concat([zeros, H, zeros], axis = 1)
        dH = H[:,1:,:]- H[:,:-1, :]
        dH = tf.abs(dH)
        transaction_costs = tf.multiply(tf.reduce_sum(tf.reduce_sum(tf.multiply(X, dH), axis = 1), axis = 1), epsilon)
        return transaction_costs
    @tf.function
    def DeepHedgeNetLoss1(self, y, trading_strategy, transaction_costs):
        loss,_ =tf.math.top_k(-(-tf.squeeze(y) + trading_strategy - transaction_costs), tf.cast(self.alpha * self.dh_batch_size, tf.int32))
        loss = tf.reduce_mean(tf.abs(loss))
        return loss
    # def DeepHedgeNetLoss2(self, y, trading_strategy, transaction_costs):
    #     loss = tf.reduce_mean(tf.square(-tf.squeeze(y) + trading_strategy - transaction_costs))
    #     return loss

    @tf.function
    def Embedder_train_step(self, X):
        with tf.GradientTape() as embedder_tape, \
                tf.GradientTape() as recovery_tape:
            H = self.Embedder(X, training = True)
            X_tilde = self.Recovery(H, training = True)
            loss = self.EmbedderNetLosst0(X, X_tilde)
        Embedder_grads = embedder_tape.gradient(loss, self.Embedder.trainable_variables)
        self.Embedder_optimizer.apply_gradients(zip(Embedder_grads, self.Embedder.trainable_variables))
        Recovery_grads = recovery_tape.gradient(loss, self.Recovery.trainable_variables)
        self.Recovery_optimizer.apply_gradients(zip(Recovery_grads, self.Recovery.trainable_variables))
        return loss
    @tf.function
    def Supervised_Generator_train_step(self, X, W):
        with tf.GradientTape() as generator_tape, \
                tf.GradientTape() as supervisor_tape:
            E = self.Generator(W)
            H = self.Embedder(X)
            H_hat_supervise = self.Supervisor(H)
            loss = self.GeneratorNet_SupervisedLoss(H, H_hat_supervise)
        # generator_grads = generator_tape.gradient(loss, self.Generator.trainable_variables)
        # self.Generator_optimizer.apply_gradients(zip(generator_grads, self.Generator.trainable_variables))
        supervisor_grads = supervisor_tape.gradient(loss, self.Supervisor.trainable_variables)
        self.Supervisor_optimizer.apply_gradients(zip(supervisor_grads, self.Supervisor.trainable_variables))
        return loss
    @tf.function
    def Joint_train_step(self, X1, X2, W1, W2):
        with tf.GradientTape() as embedder_tape, \
                tf.GradientTape() as recovery_tape, \
                tf.GradientTape() as generator_tape, \
                tf.GradientTape() as supervisor_tape, \
                tf.GradientTape() as discriminator_tape:
            # Generator training
            H = self.Embedder(X1, training = True)
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
            # Discriminator training
            H2 = self.Embedder(X2)
            E_hat2 = self.Generator(W2)
            H_hat2 = self.Supervisor(E_hat2)
            y_fake2 = self.Discriminator(H_hat2)
            y_real2 = self.Discriminator(H2)
            y_fake_e2 = self.Discriminator(E_hat2)
            _,_,_, discriminator_loss = self.DiscriminatorNetLoss(y_fake2, y_real2, y_fake_e2)
        generator_grads = generator_tape.gradient(G_loss, self.Generator.trainable_variables)
        supervisor_grads = supervisor_tape.gradient(G_loss_S, self.Supervisor.trainable_variables)
        self.Generator_optimizer.apply_gradients(zip(generator_grads, self.Generator.trainable_variables))
        self.Supervisor_optimizer.apply_gradients(zip(supervisor_grads, self.Supervisor.trainable_variables))
        embedder_grads = embedder_tape.gradient(embedder_loss, self.Embedder.trainable_variables)
        recovery_grads = recovery_tape.gradient(embedder_loss, self.Recovery.trainable_variables)
        self.Embedder_optimizer.apply_gradients(zip(embedder_grads, self.Embedder.trainable_variables))
        self.Recovery_optimizer.apply_gradients(zip(recovery_grads, self.Recovery.trainable_variables))
        if (discriminator_loss > 0.15):
            discriminator_grads = discriminator_tape.gradient(discriminator_loss, self.Discriminator.trainable_variables)
            self.Discriminator_optimizer.apply_gradients(zip(discriminator_grads, self.Discriminator.trainable_variables))
        return G_loss_U, G_loss_S, G_loss, embedder_loss, discriminator_loss
    @tf.function
    def DeepHedge_train_step(self, X, y):
        with tf.GradientTape() as tape:
            ypred = self.dhmodel(X, training = True)
            strategy = self.ReplicationPortfolio(X, ypred)
            costs = self.TransactionCosts(epsilon = self.epsilon, X = X, H = ypred)
            loss = self.DeepHedgeNetLoss1(y, strategy, costs)
        grads = tape.gradient(loss, self.dhmodel.trainable_variables)
        self.dhmodel_optimizer.apply_gradients(zip(grads, self.dhmodel.trainable_variables))
        return loss
    @tf.function
    def DeepHedge_test_step(self, X, y):
        ypred = self.dhmodel(X, training = False)
        strategy = self.ReplicationPortfolio(X, ypred)
        costs = self.TransactionCosts(epsilon = self.epsilon, X = X, H = ypred)
        loss = self.DeepHedgeNetLoss1(y, strategy, costs)
        return loss
    def DeepHedgePrice(self):
        '''
        cumulative hedging strategy, extract initial capital
        '''
        pass
    def PayoffFunction(self, X):
        return tf.math.maximum(X[:,-1, 0] - self.norm_strike, 0)
    def TermStructureLoss(self, X, y):
        '''
        Loss function of monte carlo prices relative to observed prices
        '''
        payoffs = self.PayoffFunction(X)
        loss = None
        pass

    def train(self, model_names = None, save = False, dhgan_training = False):
        val_loss = []
        save_dir = 'model_saves/DHGAN'
        print('Initiating EmbedderNet Pre-training')
        for iteration in range(self.iterations):
            print("\nStart of iteration %d" % (iteration,))
            for step, (X_batch, _) in enumerate(self.Embedder_data):
                _embedder_loss = self.Embedder_train_step(X_batch)
                if step % 200 == 0:
                    print('Batch training (embedder) loss at step %d: %.4f'% (step, float(np.sqrt(_embedder_loss))))
                    print("Seen so far: %s samples" % ((step + 1) * 64))
        print('EmbedderNet Pre-training Complete')
        print('Initiating Generator Pre-training (Supervised Loss Only)')
        for iteration in range(self.iterations):
            print("\nStart of iteration %d" % (iteration,))
            for step, (X_batch, W_batch) in enumerate(self.Generator_data):
                _generator_loss = self.Supervised_Generator_train_step(X_batch, W_batch)
                if step % 200 == 0:
                    print('Batch training supervised (generator) loss at step %d: %.4f'% (step, float(np.sqrt(_generator_loss))))
                    print("Seen so far: %s samples" % ((step + 1) * 64))
        print('GeneratorNet Pre-training Complete')
        print('Initiating Joint GAN Training')
        for iteration in range(self.iterations):
            print("\nStart of iteration %d" % (iteration,))
            for step,((X1_batch, W1_batch), (X2_batch, W2_batch)) in enumerate(self.Joint_data):
                G_loss_U, G_loss_S, generator_loss, embedder_loss, discriminator_loss = self.Joint_train_step(X1_batch, X2_batch, W1_batch, W2_batch)
                if step % 200 == 0:
                    print('Batch training discriminator loss at step %d: %.4f'% (step, float(discriminator_loss)))
                    print('Batch training generator loss at step %d: %.4f'% (step, float(generator_loss)))
                    print('Batch training unsupervised loss at step %d: %.4f'% (step, float(G_loss_U)))
                    print('Batch training supervised loss at step %d: %.4f'% (step, float(np.sqrt(G_loss_S))))
                    print('Batch training embedder loss at step %d: %.4f'% (step, float(np.sqrt(embedder_loss))))
                    print("Seen so far: %s samples" % ((step + 1) * 64))
        print('Joint GAN training Complete')
        if save:
            model_dirs = [save_dir + '/{}'.format(model_name) for model_name in model_names]
            # print('Saving model to directory: {}'.format(model_name) for model name in model_names)
            self.Generator.save(model_dirs[0])
            self.Supervisor.save(model_dirs[1])
            self.Recovery.save(model_dirs[2])
        print('Initiating Post-processing of generated data')
        W_train = self.BrownianMotion(self.dh_nb_train, self.seq_len, self.W_dim)
        W_valid = self.BrownianMotion(self.dh_nb_valid, self.seq_len, self.W_dim)
        W_test = self.BrownianMotion(self.dh_nb_test, self.seq_len, self.W_dim)

        E_train = self.Generator(W_train, training = False)
        H_train = self.Supervisor(E_train, training = False)
        X_train = self.Recovery(H_train, training = False)
        X_train = X_train[:,:, 0:self.dh_features]
        train_features, train_labels = self.PostProcessGenerated_data(X_train)
        train_dataset = self.BuildPipelineDH(train_features, train_labels)

        E_valid = self.Generator(W_valid, training = False)
        H_valid = self.Supervisor(E_valid, training = False)
        X_valid = self.Recovery(H_valid, training = False)
        X_valid = X_valid[:,:, 0:self.dh_features]
        valid_features, valid_labels = self.PostProcessGenerated_data(X_valid)
        valid_dataset = self.BuildPipelineDH(valid_features, valid_labels)

        E_test = self.Generator(W_test, training = False)
        H_test = self.Supervisor(E_test, training = False)
        X_test = self.Recovery(H_test, training = False)
        X_test = X_test[:,:, 0:self.dh_features]
        test_features, test_labels = self.PostProcessGenerated_data(X_test)
        test_dataset = self.BuildPipelineDH(test_features, test_labels)

        print('Completed Post-processing of generated data')
        print('Initiating DeepHedge Training')
        for epoch in range(self.dh_epochs):
            print("\nStart of epoch %d" % (epoch,))
            for step, (X_batch, y_batch) in enumerate(train_dataset):
                dh_loss = self.DeepHedge_train_step(X_batch, y_batch)
                if step % 200 == 0:
                    print('Batch training loss at step %d: %.4f'% (step, float(np.sqrt(dh_loss*1000))))
                    print("Seen so far: %s samples" % ((step + 1) * 64))
                    tp1 = self.dhmodel(X_batch, training = False)
            for X_batch_valid, y_batch_valid in valid_dataset:
                vloss = self.DeepHedge_test_step(X_batch_valid, y_batch_valid)
                val_loss.append(float(np.sqrt(vloss)*1000))
            avg_vloss = np.mean(np.asarray(val_loss))
            print('Average Validation loss (scaled)%.4f' %(avg_vloss))
            if save:
                model_dir = save_dir + '/{}'.format(model_names[3])
        print('DeepHedge Training Complete')
        if dhgan_training:
            print('Initiating DHGAN training')
            '''
            joint training loop of GAN and dhmodel such that funding for replication strategy matches market prices (MSE)
                - self.epsilon has to be zero
            '''
            pass

        #
        # print('DeepHedge Training Complete')
        # print('Initiating DeepHedgeGAN joint training')
        # # to be implemented
    def generate_data(self, W, generator, supervisor, recovery):
        E = generator(W, training = False)
        H = supervisor(E, training = False)
        X = recovery(H, training = False)
        return X
    def Renormalize(self, X):
        X = X * self.max_val
        X = X + self.min_val
        X = np.exp(X)
        X = X * np.asarray(self.starting_values)
        return X

    def Simulate_Hedge(self, W, generator, supervisor, recovery, dhmodel):
        X = generate_data(W, generator, supervisor, recovery)
        X_h = X[:,:,:self.dh_features]
        H = dhmodel(X_h)
        X = self.Renormalize(X)
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

if __name__ == '__main__':
    symbol_list = ['^GSPC', '^VIX', '^NDX', '^RUT', '^DJI',
                '^FVX','^TNX', '^TYX', 'EURUSD=X','JPY=X']
    settings = {'module':'gru',
                'hidden_dim':24,
                'num_layers': 3,
                'iterations': 20000,
                'batch_size': 100,
                'learning_rate':0.001}
    DH_settings = {'hedge_simulations': 1000000,
                    'hedge_features': 2,
                    'hedge_epochs': 1000,
                    'hedge_batch_size': 1000,
                    'hedge_neurons': 70,
                    'hedge_learning_rate': 0.0001,
                    'hedge_validation_proportion':0.1,
                    'hedge_test_proportion':0.001,
                    'alpha':0.99,
                    'epsilon':0.00001,
                    'strike':3200}
    datafolder_path = 'datafolder'
    filename = 'quotedata_SPX.csv'
    filepath = datafolder_path + '/' + filename
    option_data = pd.read_csv(filepath, sep = ';')
    calendar = ql.UnitedStates()
    Ks = [3000, 3100, 3200, 3300, 3400]
    ttms = [0, 10, 21, 31, 62]
    option_settings = {'start': datetime.date(2020,9,11),
                'day_count': ql.Actual365Fixed(),
                'calendar': ql.UnitedStates(),
                'strikes': Ks,
                'maturities':ttms}
    Model = DHGAN(symbol_list,settings, DH_settings, option_data ,option_settings)
    model_names = ['Generator', 'Supervisor', 'Recovery', 'dhmodel']
    Model.train(model_names, save = True)
    W = Model.BrownianMotion(10, 21, 10)
    X = Model.Simulate_Data(W, Model.Generator, Model.Supervisor, Model.Recovery)
    plt.plot(X[0, :,0])
    plt.plot(X[1, :,0])
    plt.plot(X[2, :,0])
    plt.show()
    # Model.test(Model.Generator_data)
