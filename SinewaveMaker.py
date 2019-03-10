import numpy as np
import os
import pandas as pd
import math
import csv
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt


def make_sinewave():
    with open('sinewave.csv', 'w') as file:
        writer = csv.writer(file)
        inc = 0
        wave = []
        while inc <= 5000:
            x = math.sin(inc)
            writer.writerow([math.sin(inc)])
            wave.append(x)
            inc+=0.05
        plt.plot(wave[0:1000])
        plt.show()
def make_double_sinewave():
    with open('double_sinewave.csv', 'w') as file:
        writer = csv.writer(file)
        inc = 0
        l = []
        while inc < 5000:
            x = math.sin(inc)*math.sin(2*inc)
            l.append(x)
            writer.writerow([x])
            inc+=0.05
        plt.plot(l[0:1000])
        plt.show()
#make_sinewave()
#make_double_sinewave()

def load_data(filename):
    df = pd.read_csv(filename)
    i_split = int(len(df) * 0.75)
    train_data = df.values[:i_split]
    test_data = df.values[i_split:]
    return train_data, test_data

def create_windows(data, len_window):
    X_train = []
    y_train = []
    for i in range(len(data) - len_window):
        window = data[i:i+len_window]
        X_train.append(window[:-1])
        y_train.append(window[-1])
    return np.array(X_train), np.array(y_train)



    
    
    
def build_lstm_model(base_neurons, window):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(base_neurons, input_shape=(window-1, 1), return_sequences=True ),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(1, "linear")
    ])  
    model.compile(optimizer="adam",
                  loss="mean_squared_error",
                  metrics=['accuracy'])
    return model


def train_lstm_model(model, X_train, y_train, epochs=2, callback=None):
    if callback==None:
        model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=2)
    else:
        model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=2,
                  callbacks=callback)

    return model

def predict_point_by_point(model, X_test):
    predicted = model.predict(X_test)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence(model, X_test):
    print("do_ml//predict_sequence")
    print(len(X_test))
    sequence = []
    window = X_test[0]
    window_len = len(window)
    for i in range(len(X_test)):

        sequence.append(model.predict(window[np.newaxis,:,:])[0,0])
        window = window[1:]
        window = np.insert(window, [window_len-2], sequence[-1], axis=0)
    return sequence
    
    
    

def create_callback(checkpoint_path):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    return cp_callback


def do_ml(data_type, base_neurons, window, epochs=2, from_cp=False):
    if data_type == "sinewave":
        train, test = load_data("sinewave.csv")
    if data_type == "double_sinewave":
        train, test = load_data("double_sinewave.csv")
    X_train, y_train = create_windows(train, window)
    X_test, y_test = create_windows(test, window)
    model = build_lstm_model(base_neurons, window)
    cp_path="{}/base_neurons{}/window{}/epochs{}.ckpt".format(data_type,
    if from_cp:
        model.load_weights("training/cp.ckpt")
        print("loaded model from weights")
    else:
        print("making model from scratch")
        callback = create_callback("sinewave/lower_dropout/3_epochs.ckpt")
        train_lstm_model(model, X_train, y_train, epochs=epochs, callback=[callback])
    
    predicted = predict_sequence(model, X_test[0:2000])
    f = open("results.csv", 'w')
    writer = csv.writer(f)
    writer.writerow([ "y", "Predicted"])
    for i in range(100):
        writer.writerow([y_test[i], predicted[i]])

    plt.plot(y_test[0:500], color = 'blue')
    plt.plot(predicted[0:500], color = 'orange')
    plt.show()





do_ml(from_cp=False)



