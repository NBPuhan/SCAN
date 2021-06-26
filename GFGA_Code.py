# Import libraries

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda, Multiply, Add, Reshape, Dense, Activation

# Define the Gate Layer

def GateLayer(inval,transval,gate,layers):

    # inval: input to any block, transval: transformed value, mainly attention here, gate: gate value, layers: no. of output layers
    
    dim = K.int_shape(inval)[-1]
    
    for i in range(layers):
        negated_gate = Lambda(
            lambda x:1.0 - x,
            output_shape = (dim,))(gate) # Negation of gate values for this operation
        transformed = Multiply()([gate,transval])
        identity = Multiply()([negated_gate,inval])
        Output = Add()([identity,transformed])

    return Output

# Define the self-attention layer

def Self(x, l, c, d, dv, nv, base):

    # x: 3-d input tensor, l: height/width of the tensor, e.g. 96, 48, 24 etc. c: Number of input channels, d: Number of output channels
    # d = dv*nv, if d = 1, then both are 1, else generally nv = 8
    
    x = Reshape([l*l,c])(x)     # Dimension of tensor changes from (l,l,c) to (l*l,c)
    
    v = Dense(d, activation = 'relu', name = base + '/Dense_1')(x)   # Dimension changes from (l*l,c) to (l*l,d)
    k = Dense(d, activation = 'relu', name = base + '/Dense_2')(x)   # Dimension changes from (l*l,c) to (l*l,d)
    q = Dense(d, activation = 'relu', name = base + '/Dense_3')(x)   # Dimension changes from (l*l,c) to (l*l,d)
   
    v = Reshape([l*l,dv,nv], name = base + '/Reshape_1')(v)  # Dimension changes from (l*l,d) to (l*l,dv,nv)
    k = Reshape([l*l,dv,nv], name = base + '/Reshape_2')(k)  # Dimension changes from (l*l,d) to (l*l,dv,nv)  
    q = Reshape([l*l,dv,nv], name = base + '/Reshape_3')(q)  # Dimension changes from (l*l,d) to (l*l,dv,nv)
   
    new = Multiply()([q,k])     # Dimension is (l*l,dv,nv)
    new = Activation('softmax')(new)     # Dimension is (l*l,dv,nv)
   
    new1 = Multiply()([new,v])  # Dimension is (l*l,dv,nv)
    new1 = Reshape([l*l,d],  name = base + '/Reshape_4')(new1) # Dimension changes from (l*l,dv,nv) to (l*l,d)

    X = Reshape([l,l,d])(new1)  # Dimension chnages from (l*l,d) to (l,l,d)
     
    return X

# Define the Gated fine-grained attention unit

def GFGA(x, l, c, d, dv, nv, base):

    # This block performs two-fold operation. If c = d, then the process will move like highway network. If d != c, then the input tensor
    # is first transformed to compare.

    if (d == c):

        # First use self-attention
        H = Self(x, l, c, d, dv, nv, base+'/self_att')   # Dimension changes from (l,l,c) to (l,l,d) where d = c

        # Then find gate value
        T = Dense(c, activation = 'sigmoid', name = base + '/Dense_1')(x)  # Dimension is (l,l,c)

        # Add the gated part
        output = GateLayer(x,H,T,d)

    else:

        # First use self-attention
        H = Self(x, l, c, d, dv, nv, base+'/self_att')   # Dimension changes from (l,l,c) to (l,l,d) where d != c

        # Then find gate value
        T = Dense(d, activation = 'sigmoid', name = base + '/Dense_1')(x)  # Dimension is (l,l,d)

        # Then find linear transform of input
        X = Dense(d, name = base + '/Dense_2')(x)   # Dimension is (l,l,d)

        # Add the gated part
        output = GateLayer(X,H,T,d)

    return output
