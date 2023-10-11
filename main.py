import cnn2d
import dataload
import tensorflow as tf
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score
import numpy as np
import pandas as pd
import seaborn as sns

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_devices())

physical_device = tf.config.experimental.list_physical_devices("GPU")

#tf.config.experimental.set_memory_growth(physical_device[0], True)

EmoDB_file_path = "D:/FYP/archive"


def train(train_data_x, train_data_y, validation_data_x, validation_data_y):
    model = cnn2d.model2d(input_shape=(128, 251, 1), num_classes=7)
    model.summary()
    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       verbose=0,
                       patience=20)

    mc = ModelCheckpoint('model.h5',
                         monitor='val_categorical_accuracy',
                         mode='max',
                         verbose=0,
                         save_best_only=True)

    model.fit(train_data_x, train_data_y,
              validation_data=(validation_data_x, validation_data_y),
              epochs=100,
              batch_size=4,
              verbose=2,
              callbacks=[es, mc])


def test(test_data_x, test_data_y ):
    new_model = load_model('model.h5')
    #new_model.evaluate(test_data_x, test_data_y, batch_size=1)
    # Predict probabilities for testing set using TensorFlow model
    y_pred_proba = new_model.predict(test_data_x)
    temp_array=y_pred_proba.tolist()
    temp_data=test_data_y.tolist()
    new_data=[]
    for i in  temp_data:
        temp=[0,0,0,0,0,0,0]
        a=i.index(max(i))
        temp[a]=1
        new_data.append(temp)
    c=np.array(new_data)
    new_data=c.reshape((111,7))
    new_pred=[]
    for i in  temp_array:
        temp=[0,0,0,0,0,0,0]
        a=i.index(max(i))
        temp[a]=1
        new_pred.append(temp)
    print(new_data.shape)
    b=np.array(new_pred)
    final_pred=b.reshape((111,7))
    print(final_pred.shape)
    cm = confusion_matrix(new_data.argmax(axis=1), final_pred.argmax(axis=1))
    conf_matrix = pd.DataFrame(cm, index=['anger','boredom','disgust','anxiety','fear','happiness','sadness'], columns=['anger','boredom','disgust','anxiety','fear','happiness','sadness'])
    fig, ax = plt.subplots(figsize = (10,10))
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 19}, cmap="Blues")
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('confusion.pdf', bbox_inches='tight')
    plt.show()
    print('------Weighted------')
    print('Weighted precision', precision_score(new_data.argmax(axis=1), final_pred.argmax(axis=1), average='weighted'))
    print('Weighted recall', recall_score(new_data.argmax(axis=1), final_pred.argmax(axis=1), average='weighted'))
    print('Weighted f1-score', f1_score(new_data.argmax(axis=1), final_pred.argmax(axis=1), average='weighted'))
    print('------Macro------')
    print('Macro precision', precision_score(new_data.argmax(axis=1), final_pred.argmax(axis=1), average='macro'))
    print('Macro recall', recall_score(new_data.argmax(axis=1), final_pred.argmax(axis=1), average='macro'))
    print('Macro f1-score', f1_score(new_data.argmax(axis=1), final_pred.argmax(axis=1), average='macro'))
    print('------Micro------')
    print('Micro precision', precision_score(new_data.argmax(axis=1), final_pred.argmax(axis=1), average='micro'))
    print('Micro recall', recall_score(new_data.argmax(axis=1), final_pred.argmax(axis=1), average='micro'))
    print('Micro f1-score', f1_score(new_data.argmax(axis=1), final_pred.argmax(axis=1), average='micro'))

if __name__ == '__main__':

    train_data_x, train_data_y, validation_data_x, validation_data_y, test_data_x, test_data_y = dataload.load_data(EmoDB_file_path)

    train_data_x = normalize(train_data_x)
    validation_data_x = normalize(validation_data_x)
    test_data_x = normalize(test_data_x)

    train_data_y = to_categorical(train_data_y)
    validation_data_y = to_categorical(validation_data_y)
    test_data_y = to_categorical(test_data_y)

    #train(train_data_x, train_data_y, validation_data_x, validation_data_y)

    test(test_data_x, test_data_y)
