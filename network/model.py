# from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Flatten, Dense, MaxPooling2D, Conv2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

class MyModel:

    def assemble_model(self):
        #input layer
        input_shape = (10, 10, 2) #can be basically any amount of dimensions
        inputs = Input((input_shape))

        #hidden layers
        hidden_layers = self.hidden_layers(inputs)

        #output layer
        outputs = Dense(3, activation="tanh", name="output")(hidden_layers)

        #compile
        model = Model(inputs=[inputs], output=[outputs])
        model.compile(optimizers="Adam", loss=["binary_crossentropy"], metrics={"output":"accuracy"})

        return model
    

    #Convolution for CNN (convolution neural network)
    def Convolution(input_tensor, filters, kernel_size=(3, 3), pool_size=(2, 2), strides=(1, 1)):
        
        x = Conv2D(filters, kernel_size, padding="same", strides=strides, kernel_regularizer=l2(0.001))(input_tensor)
        x = Dropout(0.1)(x)
        x = MaxPooling2D(pool_size=pool_size)(x)

        return x

    def hidden_layers(self, inputs):
        #hidden layers
        x = self.Convolution(inputs, 128)
        x = self.Convolution(x, 86)

        x = Flatten()(x)

        x = Dense(128, activation="relu")(x)
        x = Dropout(0.15)(x)
        x = Dense(8, activation="sigmoid")(x)
        x = Dropout(0.15)(x)

        return x


#method to get a list of different models
def get_models():

    #return a list of models
    model = MyModel()
    return [
        model.assemble_model()
    ]


#if program is ran by itself, print the model summaries
if __name__ == "__main__":

    for model in get_models():
        model = model[0]
        model.summary()
