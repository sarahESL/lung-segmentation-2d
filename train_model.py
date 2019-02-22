from image_gen import ImageDataGenerator
from load_data import loadDataMontgomery, loadDataJSRT
from build_model import build_UNet2D_4L
import keras
from tensorflow import convert_to_tensor
import tensorflow as tf
import pandas as pd
# from keras.utils import plot_model
import numpy as np
from keras.callbacks import ModelCheckpoint
from tensorflow import InteractiveSession

# if __name__ == '__main__':
def train():
    keras.backend.clear_session()

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    csv_path = '/home/sedigheh/lung_segmentation/dataset/JSRT/preprocessed_org_with_mask/idx.csv'
    # csv_path = '/home/sedigheh/lung_playground/dataset/JSRT/augmented/AugmentedImages/preprocessed/idx.csv'
    # Path to the folder with images. Images will be read from path + path_from_csv
    path = csv_path[:csv_path.rfind('/')] + '/'

    df = pd.read_csv(csv_path)
    # Shuffle rows in dataframe. Random state is set for reproducibility.
    df = df.sample(frac=1, random_state=23)
    n_train = int(0.8 * len(df))
    df_train = df[:n_train]
    df_val = df[n_train:]

    # Load training and validation data
    im_shape = (256, 256)
    X_train, y_train = loadDataJSRT(df_train, path, im_shape)
    X_val, y_val = loadDataJSRT(df_val, path, im_shape)

    X_train_temp = X_train[:50, :, :, :]
    y_train_temp = y_train[:50, :, :, :]

    # Build model
    org_shape = X_train[0].shape
    inp_shape = (org_shape[0], org_shape[1], org_shape[2])
    UNet = build_UNet2D_4L(inp_shape)
    UNet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    y_tensor = convert_to_tensor(y_train_temp, dtype=tf.float32)
    print("y tensor: ", y_tensor)
    loss = keras.losses.binary_crossentropy(y_tensor, UNet.output)
    sess = InteractiveSession()
    init = tf.global_variables_initializer()
    #print("Perturbation: ", sess.run(perturbation))
    sess.run(init)
    gradients = keras.backend.gradients(loss, UNet.input)
    gradients = sess.run(gradients[0], feed_dict={UNet.input:X_train_temp})

    print("Gradients0: ", gradients[0])
    print("Loss shape: ", loss.shape)
    print("UNet Input shape: ", UNet.input.shape)

    print("Gradients shape: ", gradients[0].shape)
    gradients_sign = sess.run(keras.backend.sign(gradients))
    print("Gradient0 sign: ", gradients_sign)
    print("Gradient sign shape: ", gradients_sign.shape)

    # sess = tf.InteractiveSession()
    # print(sess.run(gradients[0]))
    epsilon = 0.1
    perturbation_4dim = epsilon * gradients_sign
    perturbation = perturbation_4dim[-1, :, :, :]
    print("Perturbation shape: ", perturbation.shape)
    print("Perturbation ", perturbation)



    # Visualize model
    # plot_model(UNet, 'model.png', show_shapes=True)

    ##########################################################################################
    """
    model_file_format = 'model.{epoch:03d}.hdf5'
    print (model_file_format)
    checkpointer = ModelCheckpoint(model_file_format, period=10)

    train_gen = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rescale=1.,
                                   zoom_range=0.2,
                                   fill_mode='nearest',
                                   cval=0)

    test_gen = ImageDataGenerator(rescale=1.)

    batch_size = 8
    UNet.fit_generator(train_gen.flow(X_train, y_train, batch_size),
                       steps_per_epoch=(X_train.shape[0] + batch_size - 1) // batch_size,
                       epochs=100,
                       callbacks=[checkpointer],
                       validation_data=test_gen.flow(X_val, y_val),
                       validation_steps=(X_val.shape[0] + batch_size - 1) // batch_size)
    """
    return perturbation
