import numpy as np
from keras.layers import Embedding, Reshape, Dot, Concatenate, Dense
from keras import Input, Model, optimizers
import time
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l1
import pickle
from shared import plot_loss, rmse


learn_files = ['u1.base', 'u2.base', 'u3.base', 'u4.base', 'u5.base']
test_files = ['u1.test', 'u2.test', 'u3.test', 'u4.test', 'u5.test']
embedding = 50
param = [0.05, 0.6]
# factors for the L1 regularization
# factors = [0.1, 0.5, 0.9]
layer_configs = [(128, 64), (128, 128), (128, 64, 32), (64, 64, 64)]
for config in layer_configs:
    for i in range(5):
        learn_set = np.loadtxt(learn_files[i], dtype='int32')
        users_l = (learn_set[:, 0] - 1).reshape(None)
        movies_l = (learn_set[:, 1] - 1).reshape(None)
        ratings_l = (learn_set[:, 2]).reshape(None)

        test_set = np.loadtxt(test_files[i], dtype='int32')
        users_t = (test_set[:, 0] - 1).reshape(None)
        movies_t = (test_set[:, 1] - 1).reshape(None)
        ratings_t = (test_set[:, 2]).reshape(None)

        num_users = 943
        num_movies = 1682

        user_id_input = Input(shape=[1], name='user')
        movie_id_input = Input(shape=[1], name='movie')

        user_embedding = Embedding(output_dim=embedding, input_dim=num_users, input_length=1, name='user_embedding')(user_id_input)
        movie_embedding = Embedding(output_dim=embedding, input_dim=num_movies, input_length=1, name='movie_embedding')(movie_id_input)

        user_vecs = Reshape([embedding])(user_embedding)
        movie_vecs = Reshape([embedding])(movie_embedding)

        input_vecs = Concatenate()([user_vecs, movie_vecs])
        # Add L1 regularization
        # x = Dense(128, activation='relu', kernel_regularizer=l1(0.1))(input_vecs)
        x = Dense(config[0], activation='relu')(input_vecs)
        for nodes in config[1:]:
            x = Dense(nodes, activation='relu')(x)
        y = Dense(1)(x)

        model = Model(inputs=[user_id_input, movie_id_input], outputs=y)

        sgd = optimizers.SGD(lr=param[0], momentum=param[1], decay=0.0, nesterov=False)
        model.compile(loss=rmse, optimizer=sgd, metrics=['mae'])

        mytime = time.strftime("%Y_%m_%d_%H_%M")
        modname = 'embedding_reg-' + str(config) + '_' + str(i) + '_' + mytime
        # modname = 'embedding100_' + str(embedding) + '_' + str(i) + '_' + mytime
        thename = modname + '.h5'
        mcheck = ModelCheckpoint('models/' + thename, monitor='val_loss', save_best_only=True)
        earlyStop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        history = model.fit([users_l, movies_l], ratings_l, batch_size=64, epochs=100,
                            validation_data=([users_t, movies_t], ratings_t), callbacks=[mcheck, earlyStop])

        with open('histories/' + modname + '.pk1', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        plot_loss(history.history, title=str(config) + '-' + str(i))
