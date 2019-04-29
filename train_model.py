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
    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    csv_path = '/home/sedigheh/lung_segmentation/dataset/JSRT/preprocessed_org_with_mask/idx.csv'
    # csv_path = '/home/sedigheh/lung_segmentation/dataset/CT-kaggle-lung/mixed/idx.csv'
    # csv_path = '/home/sedigheh/lung_playground/dataset/JSRT/augmented/AugmentedImages/preprocessed/idx.csv'
    # Path to the folder with images. Images will be read from path + path_from_csv
    path = csv_path[:csv_path.rfind('/')] + '/'

    df = pd.read_csv(csv_path)
    # Shuffle rows in dataframe. Random state is set for reproducibility.
    df = df.sample(frac=1, random_state=23)
    n_train = int(0.8 * len(df))
    df_train = df[:n_train]
    df_val = df[n_train:]

    df_first = df.head(1)
    print(df_first)
    # print(df_first.shape)
    # print(type(df_first))


    # Load training and validation data
    im_shape = (256, 256)
    X_train, y_train = loadDataJSRT(df_train, path, im_shape)
    X_val, y_val = loadDataJSRT(df_val, path, im_shape)

    # X, y = loadDataJSRT(df, path, im_shape)
    X_first, y_first = loadDataJSRT(df_first, path, im_shape)

    # print("Total number of images: ", X.shape)
    # Build model
    # org_shape = X[0].shape
    org_shape = X_train[0].shape
    inp_shape = (org_shape[0], org_shape[1], org_shape[2])
    UNet = build_UNet2D_4L(inp_shape)
    UNet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    # y_tensor = convert_to_tensor(y_train_temp, dtype=tf.float32)
    y_first_tensor = convert_to_tensor(y_first, dtype=tf.float32)
    print("y tensor: ", y_first_tensor)
    # loss = keras.losses.binary_crossentropy(y_tensor, UNet.output)
    sess = InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    # gradients = keras.backend.gradients(loss, UNet.input)
    # print("**************Computing gradients****************")
    # gradients = sess.run(gradients[0], feed_dict={UNet.input:X_train_temp})
    # gradients = sess.run(gradients[0], feed_dict={UNet.input:X})

    # print("Gradients0: ", gradients[0])
    #print("Loss shape: ", loss.shape)
    #print("UNet Input shape: ", UNet.input.shape)

    #print("Gradients shape: ", gradients[0].shape)
    #gradients_sign = sess.run(keras.backend.sign(gradients))
    #print("Gradient0 sign: ", gradients_sign)
    #print("Gradient sign shape: ", gradients_sign.shape)

    # sess = tf.InteractiveSession()
    # print(sess.run(gradients[0]))
    # epsilon = 0.6

    #perturbation = epsilon * gradients_sign
    # print("first pertubartion: ", perturbation_4dim[0])
    # print("first pertubartion shape: ", perturbation_4dim[0].shape)
    # perturbation = perturbation_4dim[-1, :, :, :]
    #print("Perturbation shape: ", perturbation.shape)
    #print("Perturbation ", perturbation)
    #print(type(perturbation))
    #np.save("/home/sedigheh/lung_segmentation/perturbation_grad_wrt_x0", perturbation)
    # input("wait")



    # Visualize model
    # plot_model(UNet, 'model.png', show_shapes=True)

    ##########################################################################################
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
    UNet.fit_generator(train_gen.flow(X_val, y_val, batch_size),
                       steps_per_epoch=(X_train.shape[0] + batch_size - 1) // batch_size,
                       epochs=100,
                       callbacks=[checkpointer],
                       validation_data=test_gen.flow(X_val, y_val),
                       validation_steps=(X_val.shape[0] + batch_size - 1) // batch_size)


    loss = keras.losses.binary_crossentropy(y_first_tensor, UNet.output)
    # sess = InteractiveSession()
    # init = tf.global_variables_initializer()
    # sess.run(init)
    print("**************Computing gradients****************")
    gradients = keras.backend.gradients(loss, UNet.input)
    # gradients = sess.run(gradients[0], feed_dict={UNet.input:X_train_temp})
    gradients = sess.run(gradients[0], feed_dict={UNet.input:X_first})
    gradients_sign = sess.run(keras.backend.sign(gradients))

    np.save("/home/sedigheh/lung_segmentation/gradient_signs/grad_wrt_x0", gradients_sign)

    # print("difference: ", gradients[0]-gradients2[0])
    # return perturbation


if __name__ == "__main__":
    train()
