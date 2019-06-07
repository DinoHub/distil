import os
import keras
import pandas as pd

import keras.backend as K

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

def custom_loss(ytrue, ypred):
    return 0.5*K.categorical_crossentropy(ytrue[:,:5], ypred) #+ 0.5*K.categorical_crossentropy(ytrue[:,5:], ypred)

def folder_to_df(data_folder):
    classes = sorted(os.listdir(data_folder))
    label_dict = {}
    for i,c in enumerate(classes):
        label_dict[c] = i

    rows = []

    for cl in classes:
        cl_folder = os.path.join(data_folder, cl)
        for im in os.listdir(cl_folder):
            rel_im_path = os.path.join( cl, im )
            # print('{},{}'.format(rel_im_path, label_dict[cl]))
            rows.append( [rel_im_path, label_dict[cl]] )

    df = pd.DataFrame(rows, columns=('im_path', 'labelidx'))

    df = pd.concat( [df,pd.get_dummies( df['labelidx'] )], axis=1 )

    df = df.drop( ['labelidx'], axis=1 )

    return df

if __name__ == '__main__':
    # data_folder = 'data'
    # df = folder_to_df( data_folder )
    # df.to_csv( 'data.csv', index=False )
    # exit()

    batch_size = 1

    df = pd.read_csv( './data2.csv' )
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1)

    train_generator = datagen.flow_from_dataframe( 
        df, 
        directory='./data2/',
        x_col='im_path',
        y_col=['0','1','2','3','4'],
        class_mode='other',
        batch_size=batch_size,
        target_size=(224,224),
        subset='training' )

    val_generator = datagen.flow_from_dataframe( 
        df, 
        directory='./data2/',
        x_col='im_path',
        y_col=['0','1','2','3','4'],
        class_mode='other',
        batch_size=batch_size,
        target_size=(224,224),
        subset='validation' )

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.compile(loss=custom_loss,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size + 1,
            epochs=10,
            validation_data=val_generator,
            validation_steps=val_generator.samples // batch_size + 1)
    model.save_weights('first_try.h5')
