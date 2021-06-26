import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, Add
from GFGA_Code import GFGA

width = 96

def Stem(X, base):

    # Path 1, Gated Self 1
    L = GFGA(X, width, 3, 32, 4, 8, base+'/self_att_1')   # Dimension changes from (96,96,3) to (96,96,32)

    # Path 1, Gated Self 2
    L = GFGA(L, width, 32, 32, 4, 8, base+'/self_att_2')   # Dimension changes from (96,96,32) to (96,96,32)

    # Path 2, Gated Self 1
    R = GFGA(X, width, 3, 32, 4, 8, base+'/self_att_3')   # Dimension changes from (96,96,3) to (96,96,32)

    # Path 2, Max Pooling
    R = GFGA(pool_size = (2,2), strides = (1,1), padding = 'same', name = base + '/max_pool_1')(R) # Dimension:(96,96,32)

    Out = Add(name=base + '/Addition')([L,R])

    # Max pooling operation
    Out = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = base + '/max_pool_2')(Out) # Dimension:(48,48,32)

    return Out


def Transition(X, l, c, base):

    channel_1 = int(2*c)
    channel_2 = int(c/2)

    # Path 1, Gated Self 1
    P1 = GFGA(X, l, c, c, int(c/8), 8, base+'/self_att_1') # Dimension: (H,W,C)

    # Path 1, Gated Self 2
    P1 = GFGA(P1, l, c, channel_1, int(channel_1/8), 8, base+'/self_att_2') # Dimension: (H,W,2C)

    # Path 1, Gated Self 3
    P1 = GFGA(P1, l, channel_1, c, int(c/8), 8, base+'/self_att_3') # Dimension: (H,W,C)

    # Path 2, Gated Self 1
    P2 = GFGA(X, l, c, channel_2, int(channel_2/8), 8, base+'/self_att_4') # Dimension: (H,W,C/2)

    # Path 2, Gated Self 2
    P2 = GFGA(P2, l, channel_2, c, int(c/8), 8, base+'/self_att_5') # Dimension: (H,W,C)

    # Path 3, Gated Self 1
    P3 = GFGA(X, l, c, c, int(c/8), 8, base+'/self_att_6') # Dimension: (H,W,C)

    # Path 3, Max Pooling
    P3 = MaxPooling2D(pool_size = (2,2), strides = (1,1), padding = 'same', name = base + '/max_pool_1')(P3) # Dimension:(H,W,C)

    # Concatenate them together
    Out = Concatenate()([P1,P2,P3]) # Dimension: (H,W,3C)

    # Final max pooling for spatial aggregation
    Out = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = base + '/max_pool_2')(Out) # Dimension:(H/2,W/2,3C)

    return Out
