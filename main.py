import time
import numpy as np
import pandas
from tensorflow import convert_to_tensor
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from keras import backend as K


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(convert_to_tensor(y_true))))
    return 1 - SS_res / (SS_tot + K.epsilon())


def get_weihgts(row):
    if (row[0] < row[1]) & (row[0] < row[2]):
        return [0.4, 0.3, 0.3]
    if row[1] < row[2]:
        return [0.3, 0.4, 0.3]
    return [0.3, 0.3, 0.4]


def plots(predictions, y_data, title):
    step = list(range(1, len(predictions) + 1))

    plt.clf()
    plt.plot(step, predictions, 'bo', label='Prediction')
    plt.plot(step, predictions, 'b')
    plt.plot(step, y_data, 'ro', label='Correct value')
    plt.plot(step, y_data, 'r')

    plt.title(title)
    plt.xlabel('step')
    plt.ylabel('value')
    plt.legend()
    plt.show()


def plot_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs = range(1, len(loss) + 1)

    # Построение графика точности
    plt.clf()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def get_ensemble_predictions(models, x_data):
    predictions = []
    for model in models:
        curr_prediction = model.predict(x_data[:, 0:11])
        predictions.append(curr_prediction)
    predictions = np.asarray(predictions)
    return predictions


def evaluate_ensemble_avr(models, x_data, y_data, row):
    predictions = get_ensemble_predictions(models, x_data)
    predictions = np.mean(predictions, 0)
    predictions = predictions.flatten()

    mae = np.absolute(np.round((y_data - predictions) * 100, 2))
    cdeterm = coeff_determination(y_data, predictions)

    row.append(np.round(np.mean(mae), 2))
    row.append(np.round(cdeterm, 4))

    print("MAE: %.2f" % np.mean(mae) + " %")
    print("Coeff Determ: %.4f" % np.round(cdeterm, 4))


def evaluate_ensemble_vote(models, x_data, y_data, weights, row):
    predictions = get_ensemble_predictions(models, x_data)
    predictions = np.reshape(predictions, (len(models), 39))

    predictions[0] = predictions[0] * weights[0]
    predictions[1] = predictions[1] * weights[1]
    predictions[2] = predictions[2] * weights[2]

    predictions = np.sum(predictions, 0)
    predictions = predictions.flatten()

    mae = np.absolute(np.round((y_data - predictions) * 100, 2))
    cdeterm = coeff_determination(y_data, predictions)

    row.append(np.round(np.mean(mae), 2))
    row.append(np.round(cdeterm, 4))
    print("MAE: %.2f" % np.mean(mae) + " %")
    print("Coeff Determ: %.4f" % np.round(cdeterm, 4))


def evaluate_ensemble_median(models, x_data, y_data, row):
    predictions = get_ensemble_predictions(models, x_data)
    predictions = np.reshape(predictions, (3, 39))
    predictions = np.median(predictions, axis=0)
    predictions = predictions.flatten()

    mae = np.absolute(np.round((y_data - predictions) * 100, 2))
    cdeterm = coeff_determination(y_data, predictions)

    row.append(np.round(np.mean(mae), 2))
    row.append(np.round(cdeterm, 4))
    print("MAE: %.2f" % np.mean(mae) + " %")
    print("Coeff Determ: %.4f" % np.round(cdeterm, 4))


def get_model_two(param):
    model = Sequential()

    model.add(layers.Dense(10, input_dim=param, activation='tanh'))
    model.add(layers.Dense(9, activation='tanh'))
    model.add(layers.Dense(5, activation='tanh'))
    model.add(layers.Dense(1, kernel_initializer='normal', activation='ELU'))

    model.compile(loss='mse', optimizer='Nadam', metrics=['mae', coeff_determination])
    return model


def educate(model, train_data, train_labels, test_x, test_y, epochs, row):
    history = model.fit(train_data, train_labels, validation_data=(test_x, test_y), epochs=epochs, batch_size=3,
                        verbose=0)

    scores = model.evaluate(test_x, test_y, verbose=0)
    row.append(np.round((scores[1]) * 100, 2))
    row.append(np.round(scores[2], 4))
    print("MAE: %.2f" % np.round((scores[1]) * 100, 2))
    print("Coeff Determ: %.4f" % np.round(scores[2], 4))

    return model


def get_meta(models, x_data, y_data, row):
    predictions = get_ensemble_predictions(models, x_data)
    predictions = np.reshape(predictions, (3, 39))
    predictions = predictions.T

    model = get_model_two(3)
    train_batch_len = len(y_data) // 3
    x_test = predictions[0:2 * train_batch_len]
    x_valid = predictions[2 * train_batch_len:3 * train_batch_len]
    y_test = y_data[0:2 * train_batch_len]
    y_valid = y_data[2 * train_batch_len:3 * train_batch_len]

    educate(model, x_test, y_test, x_valid, y_valid, 1000, row)

    results = model.predict(predictions)


def get_ansamble(param, datax, datay, data_x_1, data_x_2, data_x_3, data_y_1, data_y_2, data_y_3, data_x_12, data_x_13,
                 data_x_23, data_y_12, data_y_13, data_y_23, epoch1, epoch2, epoch3, row):
    all_models = [get_model_two(11 - param), get_model_two(11 - param), get_model_two(11 - param)]

    all_models[0] = educate(all_models[0], data_x_13, data_y_13, data_x_2, data_y_2, epoch1, row)
    all_models[1] = educate(all_models[1], data_x_12, data_y_12, data_x_3, data_y_3, epoch2, row)
    all_models[2] = educate(all_models[2], data_x_23, data_y_23, data_x_1, data_y_1, epoch3, row)

    return all_models


def prep_data(row, key=0):
    dataframe = pandas.read_csv("data.csv", header=0, sep=';')
    dataset = dataframe.values

    param = 1
    datax = dataset[:, param:11]
    datay = dataset[:, 11]
    train_batch_len = len(datay) // 3

    if key == 0:
        # k-folds
        data_x_1 = datax[0: train_batch_len]
        data_x_2 = datax[train_batch_len: 2 * train_batch_len]
        data_x_3 = datax[2 * train_batch_len: 3 * train_batch_len]

        data_x_12 = datax[0 * train_batch_len: 2 * train_batch_len]
        data_x_23 = datax[train_batch_len: 3 * train_batch_len]
        data_x_13 = np.concatenate((data_x_1, data_x_3), axis=0)

        data_y_1 = datay[0: train_batch_len]
        data_y_2 = datay[train_batch_len: 2 * train_batch_len]
        data_y_3 = datay[2 * train_batch_len: 3 * train_batch_len]

        data_y_12 = datay[0 * train_batch_len: 2 * train_batch_len]
        data_y_23 = datay[train_batch_len: 3 * train_batch_len]
        data_y_13 = np.concatenate((data_y_1, data_y_3), axis=0)

        all_models = get_ansamble(param, datax, datay, data_x_1, data_x_2, data_x_3, data_y_1, data_y_2, data_y_3,
                                  data_x_12, data_x_13, data_x_23, data_y_12, data_y_13, data_y_23, 1000, 500, 500, row)

        get_meta(all_models, datax, datay, row)
        evaluate_ensemble_avr(all_models, datax, datay, row)
        evaluate_ensemble_vote(all_models, datax, datay, get_weihgts(row), row)
        evaluate_ensemble_median(all_models, datax, datay, row)

    if key == 1:
        # bagging
        dataset = (dataframe.sample(n=13, replace=True, axis=0)).values
        data_x_1 = dataset[:, param:11]
        data_y_1 = dataset[:, 11]

        dataset = (dataframe.sample(n=13, replace=True, axis=0)).values
        data_x_2 = dataset[:, param:11]
        data_y_2 = dataset[:, 11]

        dataset = (dataframe.sample(n=13, replace=True, axis=0)).values
        data_x_3 = dataset[:, param:11]
        data_y_3 = dataset[:, 11]

        dataset = (dataframe.sample(n=26, replace=True, axis=0)).values
        data_x_12 = dataset[:, param:11]
        data_y_12 = dataset[:, 11]

        dataset = (dataframe.sample(n=26, replace=True, axis=0)).values
        data_x_13 = dataset[:, param:11]
        data_y_13 = dataset[:, 11]

        dataset = (dataframe.sample(n=26, replace=True, axis=0)).values
        data_x_23 = dataset[:, param:11]
        data_y_23 = dataset[:, 11]

        all_models = get_ansamble(param, datax, datay, data_x_1, data_x_2, data_x_3, data_y_1, data_y_2, data_y_3,
                                  data_x_12, data_x_13, data_x_23, data_y_12, data_y_13, data_y_23, 100, 100, 100, row)

        get_meta(all_models, datax, datay, row)
        evaluate_ensemble_avr(all_models, datax, datay, row)
        evaluate_ensemble_median(all_models, datax, datay, row)
        evaluate_ensemble_vote(all_models, datax, datay, get_weihgts(row), row)

    if key == 2:
        # boosting
        data_x_1 = datax[0: train_batch_len]
        data_x_2 = datax[train_batch_len: 2 * train_batch_len]
        data_x_3 = datax[2 * train_batch_len: 3 * train_batch_len]

        data_x_13 = np.concatenate((data_x_1, data_x_3), axis=0)

        data_y_1 = datay[0: train_batch_len]
        data_y_2 = datay[train_batch_len: 2 * train_batch_len]
        data_y_3 = datay[2 * train_batch_len: 3 * train_batch_len]

        data_y_13 = np.concatenate((data_y_1, data_y_3), axis=0)

        all_models = [get_model_two(11 - param), get_model_two(11 - param), get_model_two(11 - param)]

        all_models[0] = educate(all_models[0], data_x_13, data_y_13, data_x_2, data_y_2, 1000, row)
        predictions = get_ensemble_predictions([all_models[0]], datax)
        datay1 = datay - predictions.reshape(39)
        data_y_1 = datay1[0: train_batch_len]
        data_y_2 = datay1[train_batch_len: 2 * train_batch_len]
        data_y_3 = datay1[2 * train_batch_len: 3 * train_batch_len]
        data_y_13 = np.concatenate((data_y_1, data_y_3), axis=0)

        all_models[1] = educate(all_models[1], data_x_13, data_y_13, data_x_2, data_y_2, 1000, row)
        predictions = get_ensemble_predictions([all_models[1]], datax)
        datay2 = datay1 - predictions.reshape(39)
        data_y_1 = datay2[0: train_batch_len]
        data_y_2 = datay2[train_batch_len: 2 * train_batch_len]
        data_y_3 = datay2[2 * train_batch_len: 3 * train_batch_len]
        data_y_13 = np.concatenate((data_y_1, data_y_3), axis=0)

        all_models[2] = educate(all_models[2], data_x_13, data_y_13, data_x_2, data_y_2, 1000, row)
        print(time.time())
        get_meta(all_models, datax, datay, row)
        evaluate_ensemble_vote(all_models, datax, datay, [1, 1, 1], row)


def main(key=0):
    f = open('results.csv', 'w')
    for i in range(0, 21):
        print("Запуск ", i)
        row = []
        prep_data(row, key)

        for el in row:
            f.write(f'{el};')
        f.write('\n')
    f.write('\n')
    f.write('\n')
    f.close()


main(1)
