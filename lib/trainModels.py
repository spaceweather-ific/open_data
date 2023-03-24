import sys
from trainUtils import *

class ConcreteDropout(Wrapper):
    """This wrapper allows to learn the dropout probability for any given input Dense layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers which have 2D
    kernels, not just `Dense`. However, Conv2D layers require different
    weighing of the regulariser (use SpatialConcreteDropout instead).
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(self, layer, weight_regularizer=0, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()

        # initialise p
        self.p_logit = self.add_weight(name='p_logit',
                                       shape=(1,),
                                       initializer=tf.random_uniform_initializer(self.init_min, self.init_max),
                                       dtype=tf.dtypes.float32,
                                       trainable=True)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x, p):
        """
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        """
        eps = 1e-07
        temp = 0.1

        unif_noise = tf.random.uniform(shape=tf.shape(x))
        drop_prob = (
            tf.math.log(p + eps)
            - tf.math.log(1. - p + eps)
            + tf.math.log(unif_noise + eps)
            - tf.math.log(1. - unif_noise + eps)
        )
        drop_prob = tf.math.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob
        
        retain_prob = 1. - p
       # print(retain_prob)
#        tf.print(p)
        
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        p = tf.math.sigmoid(self.p_logit)

        # initialise regulariser / prior KL term
        input_dim = inputs.shape[-1]  # last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(weight)) / (1. - p)
        dropout_regularizer = p * tf.math.log(p) + (1. - p) * tf.math.log(1. - p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs, p))#, regularizer
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs, p))#, regularizer

            return tf.keras.backend.in_train_phase(relaxed_dropped_inputs,
                                                   self.layer.call(inputs),
                                                   training=True)#, regularizer

# activates dropout at inference for the LSTM layer
class MonteCarloLSTM(tf.keras.layers.LSTM):
   def call(self, inputs):
      return super().call(inputs, training=True)

# ====================================================================
#  Build DNN
# ====================================================================
def build_dnn(dtype, inputshape, outputshape, lr, nlayers, nD, kC=5, sC=2, concrete_dropout = False, dropout_p = 0):

  if dtype == 0:
    return build_cnn(inputshape, outputshape, lr, nlayers, nD, kC, sC, dropout_p, concrete_dropout)
  if dtype == 1 :
    return build_lstm(inputshape, outputshape, lr, nlayers, nD, concrete_dropout, dropout_p)
  if dtype == 2 :
    return build_lstm_cnn(inputshape, outputshape, lr, nlayers, nD, kC, sC, dropout_p, concrete_dropout)
  if dtype == 3 :
    return build_attention_lstm(inputshape, outputshape, lr, nlayers, nD, dropout_p, concrete_dropout)

def build_attention_lstm(inputshape, outputshape, lr, nlayers, nD, dropout_p = 0, concrete_dropout = False):

#    print(inputshape)
    # Create the model
    model = models.Sequential()
    print('hahahaha',inputshape)
#    model.add(MonteCarloLSTM(nD, input_shape=inputshape,kernel_initializer='glorot_uniform', dropout = dropout_p))
   
    model.add(LSTM(nD, input_shape=inputshape,kernel_initializer='glorot_uniform', return_sequences=True))

    model.add(Attention(nD))

    for x in range(0,nlayers) :
      if concrete_dropout:# and x == nlayers-1:
#          model.add(Dense(nD, activation='relu'))
#          model.add(Dropout(dropout_p))

          model.add(ConcreteDropout(Dense(nD, activation = 'relu')))
      else:
          model.add(Dense(nD, activation = 'relu'))

    model.add(Dense(outputshape))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=['mse'])

    model.summary()

    return model


# ====================================================================
#  Build and set up model
#
#  LSTM
# ====================================================================
def build_lstm(inputshape, outputshape, lr, nlayers, nD, concrete_dropout = False, dropout_p = 0):

#    print(inputshape)
    # Create the model

    model = models.Sequential()
    
    model.add(MonteCarloLSTM(nD, input_shape=inputshape,kernel_initializer='glorot_uniform', dropout = dropout_p))

    for x in range(0,nlayers) :
      if concrete_dropout:# and x == nlayers-1:
#          model.add(Dense(nD, activation='relu'))
#          model.add(Dropout(dropout_p))

          model.add(ConcreteDropout(Dense(nD, activation = 'relu')))
      else:
          model.add(Dense(nD, activation = 'relu'))

    model.add(Dense(outputshape))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=['mse'])

#    model.summary()

    return model


# ====================================================================
#  Build and set up
#
#  LSTM + CNN -- TO BE CHECKED
# ====================================================================
def build_lstm_cnn(inputshape, outputshape, lr, nlayers, nD, kC=5, sC=2, dropout_p = False, concrete_dropout = False):
  # input of shape: (lb, 1, H)

#  print(inputshape)
  model = Sequential()
 
  # convolution layers
  model.add(Conv2D(filters=nD, kernel_size=(3,1), input_shape=inputshape, kernel_initializer='glorot_uniform'))
  model.add(Activation("relu"))
 
  # pooling layers
  #model.add(AveragePooling2D((2,1)))
  model.add(MaxPooling2D((2,1)))
 
  # Flatten
  model.add(Flatten())
  model.add(RepeatVector(1))

  # LSTM
  model.add(LSTM(nD, activation='tanh', return_sequences=True))
  model.add(TimeDistributed(Dense(100, activation='relu')))

  # Dense layers
  for x in range(0,nlayers) :
      if concrete_dropout and x == nlayers - 1:
          model.add(ConcreteDropout(Dense(nD, activation = 'relu')))
      else:
          model.add(Dense(nD, activation='relu'))
#          if dropout_p >= 0:
#            model.add(Dropout(dropout_p))

  #model.add(TimeDistributed(Dense(100, activation='relu')))
  #model.add(Dense(100, activation='relu'))
  model.add(Dense(int(outputshape)))

#  model.summary()

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=['mse'])

  return model

# ====================================================================
#  Build and set up model
#
#  CNN (Kernels defined by default as in Siciliano et al)
# ====================================================================
def build_cnn(inputshape, outputshape, lr, nlayers, nD,  kC=5, sC=2, dropout_p = False, concrete_dropout = False):
  # input of shape: (lb, 1, H)

#  print(inputshape)
  model = Sequential()
  
  # convolution layers
  model.add(Conv2D(filters=nD, kernel_size=(kC,1), strides=(sC, 1), input_shape=inputshape, kernel_initializer='glorot_uniform'))
  model.add(Activation("relu"))
  
  # pooling layers
  model.add(AveragePooling2D((kC,1), strides=(sC, 1)))
  model.add(MaxPooling2D((kC,1), strides=(sC, 1)))

  # Flatten
  model.add(Flatten())

  # Dense layers
  for x in range(0,nlayers) :
      if concrete_dropout and x == nlayers - 1:
          model.add(ConcreteDropout(Dense(nD, activation = 'relu')))
      else:
          model.add(Dense(nD, activation='relu'))
#          if dropout_p >= 0:
#            model.add(Dropout(dropout_p))

  model.add(Dense(int(outputshape)))

#  model.summary()

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=['mse'])

  return model


import wandb

def objective(trial,config,data,stat_data,dnn_type,selected_features,es_callback,weigh_peaks):
    """ for optuna """
    global BEST_VALUE
    global BEST_HISTORY
    global lb
    
    parameters = readConfig(config)	
    lf =  int(parameters['lf'])
    OptunaParms = eval(parameters['OptunaParms'])
    batch_size = int(parameters['batch_size'])
    epochs = int(parameters['epochs'])
    doWandb = parameters['wandb']
    step = int(parameters['step'])
    
    
    # Layers
    n_layers = trial.suggest_int('n_layers',OptunaParms['n_layers'][0],OptunaParms['n_layers'][1])
    layers = []
    # Dropout p
    p = trial.suggest_float('p', 0, 1)
    # Units
    n_units = trial.suggest_int('n_unit', OptunaParms['n_unit'][0], OptunaParms['n_unit'][1])
    # Learning rate
    lr = trial.suggest_float('lr', OptunaParms['lr'][0], OptunaParms['lr'][1], log = True)
    # Look back
    lookback = trial.suggest_categorical("lb",  OptunaParms['lb'])

    # Dropout rate
    # dropout_rate = trial.suggest_discrete_uniform('droupout_rate', 0, 0.5, 0.05)

    x_train, y_train = create_dataset(data['Train'], dnn_type, lookback, lf, selected_features, len(selected_features), interface=stat_data['Train']['breaks'], step=step) # 4 for ace only
    x_val, y_val = create_dataset(data['Validation'], dnn_type, lookback, lf, selected_features, len(selected_features), interface=stat_data['Validation']['breaks'], step=step)
    x_test, y_test = create_dataset(data['Test'], dnn_type, lookback, lf, selected_features, len(selected_features),interface=stat_data['Test']['breaks'],step=step)

    n_itoptuna = 3
    mse = np.zeros(n_itoptuna)
    lr_callback = keras.callbacks.LearningRateScheduler(custom_lr_cycle) # Cyclical Learning Rate (triangular, centered around lr, stepsize 10)
    for i in range(n_itoptuna):
      model = build_dnn(dnn_type,x_train.shape[1:], y_train.shape[1],  lr=lr, nlayers=n_layers, nD=n_units, dropout_p = p, concrete_dropout = True)
      if weigh_peaks:
          sample_weight = np.absolute(y_train)
      else:
          sample_weight = np.ones(y_train.shape)
      history = model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          callbacks=[es_callback, lr_callback],
                          verbose=0,
                          validation_data=(x_val, y_val),
                          sample_weight = sample_weight)

      y_predict = model.predict(x_val)
      y_predict = np.reshape(y_predict, (y_predict.shape[0],1))
      mse_i = mean_squared_error(y_predict, y_val)
      mse[i] = mse_i

    config = dict(trial.params)
    config["trial.number"] = trial.number
    if doWandb=="True":
        wandb.init(project = "optunaSpaceWeather",
                entity = "daniel-condex",
                config = config,
                group = "SPACE_WEATHER",
                reinit = True
    )

    mse_mean = np.mean(mse)
    mse_std = np.std(mse)
    
    if doWandb=="True":
        trial.report(mse_mean, step = step)
        wandb.log(data = {"mse": mse_mean}, step = step)

    print(mse_mean, " +- ", mse_std)

    return mse_mean





