from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Flatten, Dense, MaxPooling2D, Conv2D, Input, Add#, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import BinaryAccuracy, Accuracy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam



#Convolution for CNN (convolution neural network)
def Convolution(input_tensor, filters, kernel_size, pool_size, strides):
    
    x = Conv2D(filters, kernel_size, padding="same", strides=strides, kernel_regularizer=l2(0.001))(input_tensor)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(pool_size=pool_size)(x)

    return x


def model(RES=32, COLOR_TYPE="bgr", COLORS=3, CONVS=4, CONVSTART=32, CONVMULT=2, CONVADD=0, KERNELSIZE=(3, 3), POOLSIZE=(2, 2), STRIDES=(1, 1)): #COLOR_TYPE bgr or bw

    #define inputshape and inputs
    input_shape = (RES, RES, COLORS)
    inputs = Input((input_shape))


    #add the disired amount of convolutions
    conv_size = CONVSTART
    convs = []
    next_layer = inputs
    for i in range(CONVS):
        convs.append(Convolution(next_layer, conv_size, KERNELSIZE, POOLSIZE, STRIDES))
        next_layer = convs[i]
        conv_size = conv_size * CONVMULT + CONVADD

    
    #add flatten layer and dense layers that split into a part that trains 
    flatten = Flatten()(convs[-1])

    #gender
    dense_1 = Dense(128, activation="relu")(flatten)
    drop_1 = Dropout(0.15)(dense_1)
    output_1 = Dense(1, activation="sigmoid", name="gender_out")(drop_1)
    
    #define the model and compile it
    outputs = [output_1]
    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer="Adam", loss=["binary_crossentropy"], metrics={"output" : "accuracy"})
    

    #return the model and it's paramaters
    return [model, RES, COLORS, COLOR_TYPE]


#method to get a list of different models
def get_models():

    #return a list of models
    return [
        model()
    ]


#if program is ran by itself, print the model summaries
if __name__ == "__main__":

    for model in get_models():
        model = model[0]
        model.summary()
