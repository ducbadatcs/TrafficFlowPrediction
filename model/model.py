"""
Defination of NN model
"""
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, GRU, Conv1D, GlobalAveragePooling1D, MultiHeadAttention
from keras.models import Sequential
from keras import Model, Input

def get_lstm(units: list[int]):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def get_gru(units: list[int]):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def _get_sae(inputs: int, hidden: int, output: int) -> Sequential:
    """SAE(Auto-Encoders)
    Build SAE Model.

    # Arguments
        inputs: Integer, number of input units.
        hidden: Integer, number of hidden units.
        output: Integer, number of output units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential([
        Input(shape=(inputs,)),
        Dense(hidden, name="hidden"),
        Dropout(0.2),
        Dense(output, activation="sigmoid")
    ])

    return model


def get_saes(layers: list[int]):
    """SAEs(Stacked Auto-Encoders)
    Build SAEs Model.

    # Arguments
        layers: List(int), number of input, output and hidden units.
    # Returns
        models: List(Model), List of SAE and SAEs.
    """
    sae1 = _get_sae(layers[0], layers[1], layers[-1])
    sae2 = _get_sae(layers[1], layers[2], layers[-1])
    sae3 = _get_sae(layers[2], layers[3], layers[-1])

    saes = Sequential([
        Input(shape=(layers[0],)),
        Dense(layers[1], name="hidden1"), Activation("sigmoid"),
        Dense(layers[2], name="hidden2"), Activation("sigmoid"),
        Dense(layers[3], name="hidden3"), Activation("sigmoid"),
        Dropout(0.2),
        Dense(layers[4], activation="sigmoid")
    ])
    # saes.add(Dense(layers[1], input_dim=layers[0], name='hidden1'))
    # saes.add(Activation('sigmoid'))
    # saes.add(Dense(layers[2], name='hidden2'))
    # saes.add(Activation('sigmoid'))
    # saes.add(Dense(layers[3], name='hidden3'))
    # saes.add(Activation('sigmoid'))
    # saes.add(Dropout(0.2))
    # saes.add(Dense(layers[4], activation='sigmoid'))

    models = [sae1, sae2, sae3, saes]

    return models

# extra models


def get_cnn(units: list[int]) -> Sequential:
    model = Sequential([
        Conv1D(32, 3, padding="causal", activation="relu", input_shape=(units[0], 1)),
        Conv1D(64, 3, padding='causal', activation='relu'),
        GlobalAveragePooling1D(),
        Dropout(0.2),
        Dense(units[3], activation='sigmoid')
    ])
    
    return model