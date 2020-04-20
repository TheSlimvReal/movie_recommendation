from keras import backend as K
import matplotlib.pyplot as plt


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def plot_loss(history, title=None):
    # "Loss"
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    t = 'model loss'
    if title:
        t += ': ' + title
    plt.title(t)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()