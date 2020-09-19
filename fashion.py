# General imports
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(sys.argv[0]))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# sklearn imports
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

#PCA and LR
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# SVM
from sklearn.svm import SVC

# Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasClassifier

# # TensorFlow version
# import tensorflow as tf
# print('TensorFlow version: ', tf.__version__)


# Already implemented
def get_data(datafile, to_shuffle = False):
    dataframe = pd.read_csv(datafile)
    if to_shuffle:
        dataframe = shuffle(dataframe)
    data = list(dataframe.values)
    labels, images = [], []
    for line in data:
        labels.append(line[0])
        images.append(line[1:])
    labels = np.array(labels)
    images = np.array(images).astype('float32')
    images /= 255
    return images, labels


# Already implemented
def visualize_weights(trained_model, num_to_display=10, save=True, hot=True):
    layer1 = trained_model.layers[0]
    weights = layer1.get_weights()[0]

    # Feel free to change the color scheme
    colors = 'hot' if hot else 'binary'

    for i in range(num_to_display):
        wi = weights[:,i].reshape(28, 28)
        plt.imshow(wi, cmap=colors, interpolation='nearest')
        plt.show()


# Already implemented
def output_predictions(predictions, method):
    with open('predictions_%s.txt' % method, 'w+') as f:
        for pred in predictions:
            f.write(str(pred) + '\n')


"""
Implement the following method to generate plots of the train and validation accuracy and loss vs epochs. 
"""

def plot_history(history, method):
    train_loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']

    train_acc_history = history.history['accuracy']
    val_acc_history = history.history['val_accuracy']

    outputFileFolder = "./data_" + method + "/"
    if not os.path.exists(outputFileFolder):
        os.makedirs(outputFileFolder)

    # save data
    nx = len(train_loss_history)
    epochs = [i for i in range(len(train_loss_history))]

    loss_vs_epoch = []
    loss_vs_epoch.append(epochs)
    loss_vs_epoch.append(train_loss_history)
    loss_vs_epoch.append(val_loss_history)
    np.savetxt(outputFileFolder + 'loss_vs_epoch' + '.csv', loss_vs_epoch, delimiter=',')

    acc_vs_epoch = []
    acc_vs_epoch.append(epochs)
    acc_vs_epoch.append(train_acc_history)
    acc_vs_epoch.append(val_acc_history)    
    np.savetxt(outputFileFolder + 'acc_vs_epoch' + '.csv', acc_vs_epoch, delimiter=',')

    # plot
    plt.plot(epochs,train_loss_history, label='training loss')
    plt.plot(epochs,val_loss_history, label='testing loss')
    # plt.scatter(val_loss_history.index(min(val_loss_history)), min(val_loss_history),
    #             c='r', marker='o', label='minimum_val_loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.xticks(np.linspace(0, nx, 5))
    plt.legend(loc='best')
    plt.grid(True)
    plt.title('training loss and testing loss vs. epoch number')
    plt.savefig(outputFileFolder + 'loss_vs_epoch.png')
    plt.show()

    plt.plot(epochs,train_acc_history, label='training accuracy')
    plt.plot(epochs,val_acc_history, label='testing accuracy')
    plt.scatter(val_acc_history.index(max(val_acc_history)), max(val_acc_history),
                c='r', marker='o', label='maximum_testing_accuracy')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.xticks(np.linspace(0, nx, 5))
    plt.legend(loc='best')
    plt.grid(True)
    plt.title('training accuracy and testing accuracy vs. epoch number')
    plt.savefig(outputFileFolder + 'accuracy_vs_epoch.png')   
    plt.show()


"""Code for defining and training your MLP models"""

def create_mlp(args):
    # You can use args to pass parameter values to this method

    # Define model architecture
    model = Sequential()
 
    # 1st dense layer
    model.add(Dense(units=512, activation='relu', input_dim=28*28))
 
    # 2nd dense layer
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=512, activation='relu'))

    # softmax layer
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=10, activation='softmax'))
    # add more layers...

    # Define Optimizer
    # optimizer = keras.optimizers.SGD(**args)
    optimizer = keras.optimizers.Adam(**args)

    # Compile
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def train_mlp(x, y, x_val=None, y_val=None, val_split=1/3, epochs = 80, x_vali=None, y_vali=None, args=None):
    # You can use args to pass parameter values to this method
    y = keras.utils.to_categorical(y, num_classes=10)
    model = create_mlp(args)
    history = model.fit(x, y, validation_split=val_split, batch_size=512, epochs=epochs, shuffle=False, verbose=2)

    return model, history


"""Code for defining and training your CNN models"""

def create_cnn(args=None):
    # You can use args to pass parameter values to this method

    # 28x28 images with 1 color channel
    input_shape = (28, 28, 1)

    # Define model architecture
    model = Sequential()
 
    # 1st conv layer
    model.add(Conv2D(filters=32, activation='relu', kernel_size=3, strides=1, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # can add more layers here...

    # 2nd conv layer
    model.add(Dropout(rate=0.25))
    # model.add(Conv2D(filters=64, activation='relu', kernel_size=3, strides=1))
    model.add(MaxPooling2D(pool_size=2, strides=1))
    # model.add(MaxPooling2D(pool_size=2))
 
    # 3rd conv layer
    model.add(Dropout(rate=0.25))
    # model.add(Conv2D(filters=128, activation='relu', kernel_size=3, strides=1))
    model.add(MaxPooling2D(pool_size=2, strides=1))
    # model.add(MaxPooling2D(pool_size=2))
  
    # flatten layer
    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    # can add more layers here...

    # 1st dense layer
    model.add(Dense(units=1024, activation='relu'))
 
    # 2nd dense layer
    model.add(Dropout(rate=0.25))
    model.add(Dense(units=512, activation='relu'))
 
    # 3rd dense layer
    model.add(Dropout(rate=0.25))
    model.add(Dense(units=128, activation='relu'))
 
    # softmax layer
    model.add(Dropout(rate=0.25))
    model.add(Dense(units=10, activation='softmax'))

    # Optimizer
    optimizer = keras.optimizers.Adam(**args)

    # Compile
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def train_cnn(x, y, x_val=None, y_val=None, val_split=1/3, epochs = 50, x_vali=None, y_vali=None, args=None):
    # You can use args to pass parameter values to this method
    x = x.reshape(-1, 28, 28, 1)
    y = keras.utils.to_categorical(y, num_classes=10)
    model = create_cnn(args)
    history = model.fit(x, y, validation_split=val_split, batch_size=512, epochs=epochs, shuffle=False, verbose=1)
    return model, history



"""An optional method you can use to repeatedly call create_mlp, train_mlp, create_cnn, or train_cnn. 
You can use it for performing cross validation or parameter searching.
"""

def train_and_select_model(file_name, method = 'cnn'):
    """Optional method. You can write code here to perform a 
    parameter search, cross-validation, etc. """

    x, y = get_data(file_name)
    _, x_test, _, _ = train_test_split(x, y, test_size=1/3, shuffle=False)   

    best_acc = 0
    best_epoch = 0
    best_params = 0
    best_model = 0
    best_history = 0

    # sgd args
    # args = {
    #     'lr': 0.01, 'decay': 1e-6, 'momentum': 0.9, 'nesterov': True,
    # }

    # Adam args
    # args = [
    #         {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-7, 'decay': 1e-2, 'amsgrad': False},
    #         {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-7, 'decay': 1e-3, 'amsgrad': False},
    #         {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-7, 'decay': 1e-4, 'amsgrad': False},
    #         {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-7, 'decay': 1e-5, 'amsgrad': False},
    #         {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-7, 'decay': 1e-6, 'amsgrad': False},
    #         {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-7, 'decay': 1e-7, 'amsgrad': False},
    #         {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-7, 'decay': 0, 'amsgrad': False},
    #         {'lr': 0.003, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-7, 'decay': 0.0, 'amsgrad': False},
    #         {'lr': 0.005, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-7, 'decay': 0.0, 'amsgrad': False},
    #         {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-7, 'decay': 1e-3, 'amsgrad': True},
    #         ]
    # After testing, the best params for mlp is {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-7, 'decay': 1e-2, 'amsgrad': False}

    args = [{'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-7, 'decay': 1e-2, 'amsgrad': False}]

    for params in args: 
        if method == 'mlp':       
            print("MLP training...\nusing Adam optimizer parameter: \n", params)
            model, history = train_mlp(x, y, x_vali=None, y_vali=None, args=params)
        else:
            print("CNN training...\nusing Adam optimizer parameter: \n", params)
            model, history = train_cnn(x, y, x_vali=None, y_vali=None, args=params)

        validation_accuracy = history.history['val_accuracy']
        max_acc = max(validation_accuracy)
        idx_max_acc = validation_accuracy.index(max_acc)+1

        print("\nmaximum accuracy of %f training on the epoch %d" % (max_acc, idx_max_acc))
        print("using Adam optimizer parameter: \n", params)
        print("================================================================")

        if max_acc > best_acc:
            best_acc = max_acc
            best_epoch = idx_max_acc
            best_params = params
            best_model = model
            best_history = history

    print("================================================================")
    print("final parameters:\n")
    print("best_acc: %f" % best_acc)
    print("best_epoch: %d" % best_epoch)
    print("Adam optimizer parameter: \n", best_params)
    print("================================================================")

    # model, _ = train_mlp(x, y, x_vali=None, y_vali=None, epochs=best_epoch, args=best_params)
    if method == 'cnn':
        x_test = x_test.reshape(-1, 28, 28, 1)
    predictions = best_model.predict_classes(x_test)
    # print(predictions[:10])

    return best_model, best_history, predictions


def write_conf_matrix(predictions, method, save = False):
    x, y = get_data(file_name)
    _, _, _, y_test = train_test_split(x, y, test_size=1/3, shuffle=False)   

    outputFileFolder = "./data_" + method + "/"
    conf_matrix = metrics.confusion_matrix(y_test, predictions)
    np.savetxt(outputFileFolder + 'conf_matrix' + '.csv', conf_matrix, delimiter=',')
    
    if save:
        output_predictions(predictions, method)


"""PCA + LR."""

def train_single_pca_lr(x_train, y_train, x_test, y_test, pcs, verbose = 1):
    # pca
    pca = PCA(n_components = pcs, random_state=100)
    pca.fit(x_train)
    n_components = pca.n_components_

    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    
    # lr
    print("start logistic regression training...")
    logisticRegr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200, n_jobs=-1, random_state=100)
    logisticRegr.fit(x_train, y_train)

    # model evaluation
    train_predicted = logisticRegr.predict(x_train)
    test_predicted = logisticRegr.predict(x_test)

    train_loss = metrics.zero_one_loss(train_predicted, y_train)
    test_loss = metrics.zero_one_loss(test_predicted, y_test)
    train_acc = metrics.accuracy_score(train_predicted, y_train)
    test_acc = metrics.accuracy_score(test_predicted, y_test)
    conf_matrix = metrics.confusion_matrix(y_test, test_predicted)

    if verbose:
        print("x_train shape before PCA: ", x_train.shape)
        print("x_test shape before PCA: ", x_test.shape)      
        print("x_train shape after PCA: ", x_train.shape)
        print("x_test shape after PCA: ", x_test.shape)

        ratio = sum(pca.explained_variance_ratio_)
        print("n_components: %d" % n_components)
        print("variance retained: %.4f" % ratio)

    return train_loss, test_loss, train_acc, test_acc, conf_matrix


def show_result_with_pca_lr(file_name, show = True, verbose = 1):
    
    x, y = get_data(file_name)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, shuffle=False)

    max_n_components = x_train.shape[1]
    print("\n***\tmax_n_components: %d\t***\n" % max_n_components)

    pcs = [int(max_n_components*0.1*(i+1)) for i in range(10)]
    # print(pcs)
    # pcs = pcs[:2]

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    conf_matrices = []
    for pc in pcs:
        train_loss, test_loss, train_acc, test_acc, conf_matrix = train_single_pca_lr(x_train, y_train, x_test, y_test, pc, verbose)

        if verbose:
            print("training loss: %.4f" % train_loss)
            print("test loss: %.4f" % test_loss)
            print("training accuracy: %.4f" % train_acc)
            print("test accuracy: %.4f" % test_acc)
            print(conf_matrix)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        conf_matrices.append(conf_matrix)

        print("================================================================\n")

    if show:

        outputFileFolder = "./data_lr/"
        if not os.path.exists(outputFileFolder):
            os.makedirs(outputFileFolder)

        plt.plot(pcs, train_losses, label = 'training loss')
        plt.plot(pcs, test_losses, label = 'test loss')
        plt.xlabel("PCs")
        plt.ylabel("loss")
        plt.legend(loc='best')
        plt.grid(True)
        plt.title('loss vs. PCs')
        plt.savefig(outputFileFolder + 'loss_vs_pcs.png')
        plt.show()

        plt.plot(pcs, train_accs, label = 'training accuracy')
        plt.plot(pcs, test_accs, label = 'test accuracy')
        plt.xlabel("PCs")
        plt.ylabel("accuracy")
        plt.legend(loc='best')
        plt.grid(True)
        plt.title('accuracy vs. PCs')
        plt.savefig(outputFileFolder + 'acc_vs_pcs.png')
        plt.show()

    loss_vs_pcs_data = []
    loss_vs_pcs_data.append(pcs)
    loss_vs_pcs_data.append(train_losses)
    loss_vs_pcs_data.append(test_losses)
    acc_vs_pcs_data =[]
    acc_vs_pcs_data.append(pcs)
    acc_vs_pcs_data.append(train_accs)
    acc_vs_pcs_data.append(test_accs)

    index_highest_score = np.argmax(test_accs)
    print("index with highest score: %d" % index_highest_score)

    return loss_vs_pcs_data, acc_vs_pcs_data, conf_matrices[index_highest_score]


def train_and_test_model(file_name, method):
    
    if method == 'mlp':
        best_model, best_history, predictions = train_and_select_model(file_name, method)
        # print(best_model.summary())
        plot_history(best_history, method)

        # visualize how mlp learn weights
        # visualize_weights(best_model)

        # save predictions
        write_conf_matrix(predictions, method, save=False)

    elif method == 'cnn':
        best_model, best_history, predictions = train_and_select_model(file_name, method)
        # print(best_model.summary())
        plot_history(best_history, method)

        # save predictions
        write_conf_matrix(predictions, method, save=False)

    elif method ==' logistic':
        # pca and lr
        verbose = 1
        loss_vs_pcs_data, acc_vs_pcs_data, conf_matrices = show_result_with_pca_lr(file_name, show=True, verbose=verbose)

        save_list = [loss_vs_pcs_data, acc_vs_pcs_data, conf_matrices]
        save_name = ['loss_vs_pcs_data', 'acc_vs_pcs_data', 'conf_matrices']

        outputFileFolder = "./data_lr/"
        for i, item in enumerate(save_list):
            filename = outputFileFolder + save_name[i] + '.csv'
            np.savetxt(filename, item, delimiter=',')

    elif method == 'svm':
        acc, conf_matrix = train_with_svm(file_name)

    else:
        print("Incorrect method chosen!")



def train_with_svm(file_name):

    x, y = get_data(file_name)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, shuffle=False)    

    # pca
    print("starting pca...")
    pca = PCA(n_components = 156, random_state=100)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    # grid search
    # print("starting grid search...")
    # tuned_parameters = [{'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 
    #                     'gamma': ['auto'], 'C': [0.1, 1, 10]}]
    
    # clf = GridSearchCV(SVC(), tuned_parameters, scoring='accuracy')
    # clf.fit(x_train, y_train)
    # print(clf.best_params_)
    # best_svm = clf.best_estimator_
    # acc = best_svm.score(x_test, y_test)
    # print("score: %.4f" % acc)

    # After grid search, the best params for svm is {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}
    print("starting svm...")
    args = {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}

    clf = SVC(**args)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)

    return acc, conf_matrix


"""Main method. Make sure the file paths here point to the correct place in your file."""

if __name__ == '__main__':
    # Edit the following two lines if your paths are different
    file_name = './fashion.csv'

    # method shoul be chose from 'mlp', 'cnn', 'logistic', 'svm'
    train_and_test_model(file_name, method = 'cnn')