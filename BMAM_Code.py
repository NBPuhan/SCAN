import tensorflow as tf
from tensorflow.keras.layers import Activation, Multiply, Dense, GlobalAveragePooling2D, Add
from GFGA_Code import GFGA, GateLayer

def Spatial(X_input, l, c, base):

    X = GFGA(X_input, l, c, c, int(c/8), 8, base+'/self_att_1') # Dimension: (H,W,C)

    X = GFGA(X, l, c, 1, 1, 1, base+'/self_att_2') # Dimension: (H,W,1)
    
    X = Activation('sigmoid')(X) # Activation operation for attention

    X = Multiply()([X,X_input])  # Dimension: (H,W,C)

    # Perform the gating operation

    # Find gate value

    T = Dense(c, activation = 'sigmoid', name = base + '/Dense_1')(X_input)  # Dimension: (H,W,C)
    
    # The Gating mechanism

    Output = GateLayer(X_input,X,T,c)

    return Output


def Channel(X_input, l, c, base):

    Y = GlobalAveragePooling2D()(X_input) # Dimension: C

    Y = Dense(int(c/4), activation = 'relu', name = base + '/dense_3')(Y)

    Y = Dense(c, activation = 'sigmoid', name = base + '/dense_4')(Y)

    Y = Multiply()([Y,X_input]) # Dimension: (H,W,C)

    # Perform the gating operation

    # Find transform

    T = Dense(c, activation = 'sigmoid', name = base + '/Dense_6')(X_input)  # Dimension: (H,W,C)

    # The Gating mechanism
    
    Output = GateLayer(X_input,Y,T,c)
    
    return Output

def BMAM(X_input, l, c, base):

    channel_val = Channel(X_input, l, c, base + '/channel')
    spatial_val = Spatial(X_input, l, c, base + '/spatial')

    Output = Add()([channel_val, spatial_val])

    return Output
