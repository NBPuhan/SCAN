import tensorflow as tf
from tensorflow.keras.layers import
from GFGA_Code import GFGA
from BMAM_Code import BMAM

def MSAM_1(X, l, c, base):

    # Here l = 48, H,W = 48, C = 32

    X_2 = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = base + '/max_pool_1')(X) # Dimension:(H/2,W/2,C)
    X_4 = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = base + '/max_pool_2')(X_2) # Dimension:(H/4,W/4,C)
    X_8 = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = base + '/max_pool_3')(X_4) # Dimension:(H/8,W/8,C)


    # Path 1
    P1 = BMAM(X, l, c, base+'/path_1') # Dimension: (H,W,C)
    
    # Path 2
    P2 = BMAM(X_2, int(l/2), c, base+'/path_2') # Dimension: (H/2,W/2,C)

    P2 = UpSampling2D(size = (2,2), interpolation = "bilinear")(P2) # Dimension: (H,W,C)

    # Path 3
    P3 = BMAM(X_4, int(l/4), c, base+'/path_3') # Dimension: (H/4,W/4,C)
    
    P3 = UpSampling2D(size = (2,2), interpolation = "bilinear")(P3) # Dimension: (H/2,W/2,C)
    P3 = UpSampling2D(size = (2,2), interpolation = "bilinear")(P3) # Dimension: (H,W,C)

    # Path 4
    P4 = BMAM(X_8, int(l/8), c, base+'/path_4') # Dimension: (H/8,W/8,C)
    
    P4 = UpSampling2D(size = (2,2), interpolation = "bilinear")(P4) # Dimension: (H/4,W/4,C)
    P4 = UpSampling2D(size = (2,2), interpolation = "bilinear")(P4) # Dimension: (H/2,W/2,C)
    P4 = UpSampling2D(size = (2,2), interpolation = "bilinear")(P4) # Dimension: (H,W,C)

    # Local Feature extraction
    P1 = GatedSelf(P1, l, c, c, int(c/8), 8, base+'/self_att_1') # Dimension: (H,W,C)
    P2 = GatedSelf(P2, l, c, c, int(c/8), 8, base+'/self_att_2') # Dimension: (H,W,C)
    P3 = GatedSelf(P3, l, c, c, int(c/8), 8, base+'/self_att_3') # Dimension: (H,W,C)
    P4 = GatedSelf(P4, l, c, c, int(c/8), 8, base+'/self_att_4') # Dimension: (H,W,C)


    # Finally concatenated together
    Out = Add()([P1,P2,P3,P4]) # Dimension: (H,W,C)

    return Out


def MSAM_2(X, l, c, base):

    # Here l = 24, H,W = 24, C = 128

    X_2 = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = base + '/max_pool_1')(X) # Dimension:(H/2,W/2,C)
    X_4 = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = base + '/max_pool_2')(X_2) # Dimension:(H/4,W/4,C)


    # Path 1
    P1 = BMAM(X, l, c, base+'/path_1') # Dimension: (H,W,C)

    # Path 2
    P2 = BMAM(X_2, int(l/2), c, base+'/path_2') # Dimension: (H/2,W/2,C)

    P2 = UpSampling2D(size = (2,2), interpolation = "bilinear")(P2) # Dimension: (H,W,C)

    # Path 3
    P3 = BMAM(X_4, int(l/4), c, base+'/path_3') # Dimension: (H/4,W/4,C)

    P3 = UpSampling2D(size = (2,2), interpolation = "bilinear")(P3) # Dimension:(H/2,W/2,C)
    P3 = UpSampling2D(size = (2,2), interpolation = "bilinear")(P3) # Dimension: (H,W,C)

    # Local Feature extraction
    P1 = GatedSelf(P1, l, c, c, int(c/8), 8, base+'/self_att_1') # Dimension: (H,W,C)
    P2 = GatedSelf(P2, l, c, c, int(c/8), 8, base+'/self_att_2') # Dimension: (H,W,C)
    P3 = GatedSelf(P3, l, c, c, int(c/8), 8, base+'/self_att_3') # Dimension: (H,W,C)

    # Finally concatenated together
    Out = Add()([P1,P2,P3]) # Dimension: (H,W,C)
    return Out

def MSAM_3(X, l, c, base):

    # Here l = 12, H,W = 12, C = 384

    X_2 = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = base + '/max_pool_1')(X) # Dimension:(H/2,W/2,C)

    # Add them together
    P1 = BMAM(X, l, c, base + '/path_1') # Dimension: (H,W,C)

    # Path 2
    P2 = BMAM(X_2, int(l/2), c, base+'/path_2') # Dimension: (H/2,W/2,C)

    P2 = UpSampling2D(size = (2,2), interpolation = "bilinear")(P2) # Dimension: (H,W,C)

    # Local Feature extraction
    P1 = GatedSelf(P1, l, c, c, int(c/8), 8, base+'/self_att_1') # Dimension: (H,W,C)
    P2 = GatedSelf(P2, l, c, c, int(c/8), 8, base+'/self_att_2') # Dimension: (H,W,C)

    # Finally concatenated together
    Out = Add()([P1,P2]) # Dimension: (H,W,C)
    return Out
