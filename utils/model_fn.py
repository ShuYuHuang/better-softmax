

## Embedding ##
class Encoder(keras.Model):
    def __init__(self,n_layers=2,latent_features=2,hidden_featrues=64):
        super().__init__()
        self.latent_features=latent_features
        # First Layer
        self.conv_0=keras.layers.Conv2D(hidden_featrues, kernel_size=3, activation="relu",padding='same')
        self.dropout_0=keras.layers.Dropout(0.2)
        # Middle Layers
        self.convs=[keras.layers.Conv2D(hidden_featrues, kernel_size=3, activation="relu",padding='same') for _ in range(n_layers)]
        self.mxpools = [keras.layers.MaxPooling2D(pool_size=2) for _ in range(n_layers)]
        self.dropouts = [keras.layers.Dropout(0.2) for _ in range(n_layers)]
        # Integration Layer(Output Layer)
        WIDTH=2
        self.gap=keras.layers.AveragePooling2D(pool_size=WIDTH,)
        self.flatten=keras.layers.Flatten()
        self.dense=keras.layers.Dense(latent_features)
      
    def call(self,x):
        x=self.dropout_0(self.conv_0(x))
        for conv,mxp,drp in zip(self.convs,self.mxpools,self.dropouts):
            x=drp(mxp(conv(x)))
        x=self.gap(x)
        x=self.flatten(x)
        y=self.dense(x)
        return y
    

## Special Layers ##
class MatMul:
    def __init__(self,mode="paired"):
        self.mode=mode
        assert mode in ["paired","cross"]
    def __call__(self,q,s):
        if self.mode=="paired":
            return tf.reduce_sum(tf.multiply(q,s),axis=-1,keepdims=True)
        if self.mode=="cross":
            return tf.matmul(q,s,transpose_b=True)
        
class EuclideanDist:
    def __init__(self,mode="paired"):
        self.mode=mode
        assert mode in ["paired","cross"]
        
    def __call__(self,q,s):
        if self.mode=="paired":
            dist=tf.math.reduce_euclidean_norm(q-s,axis=-1,keepdims=True)
        if self.mode=="cross":
            dist=-tf.math.reduce_euclidean_norm(
                tf.stack([q[:,qq:qq+1]-s for qq in range(q.shape[1])],axis=1)
                ,axis=-1,keepdims=False)
        return dist

class CosSim:
    def __init__(self,mode="paired"):
        assert mode in ["paired","cross"]
        self.mode=mode
        self.matmul=MatMul(mode)
    def __call__(self,q,s):
        query_n=tf.math.l2_normalize(q, axis=-1)
        support_n=tf.math.l2_normalize(s, axis=-1)
        return self.matmul(query_n,support_n)
    
class CosineLayer(keras.layers.Layer):
    def __init__(self, out_features=10,mode="cross"):
        super().__init__()
        self.out_features = out_features
        self.cos_sim=CosSim(mode)
    def build(self, input_shape):
        self.w = self.add_weight(shape=(self.out_features,input_shape[-1]),
                               initializer='glorot_uniform',
                               trainable=True)
    def call(self, inputs):
        return self.cos_sim(inputs,self.w)
    
## Loss function ##
class ContrastiveLoss:
    def __init__(self,m=1,mode="dist"):
        
        self.mode=mode
        if self.mode=="dist":
            self.m=m
        if self.mode=="cos" or self.mode=="matmul":
            self.activation=keras.activations.sigmoid
            self.loss_fn=keras.losses.BinaryCrossentropy()
        assert mode in ["dist","cos","matmul"]
        self.__name__="loss"
    def __call__(self,y_true, y_pred):
        if self.mode=="dist":
            loss=tf.reduce_mean(y_true * tf.square(y_pred) +
                                 (1 - y_true)* tf.square(tf.maximum(self.m - y_pred, 0)),
                               axis=-1,keepdims=True)
        if self.mode=="cos" or self.mode=="matmul":
            loss=self.loss_fn(y_true,self.activation(y_pred))
        return loss
    
class FocalLoss:
    def __init__(self, gamma=3, eps=1e-10):
        self.gamma = gamma
        self.eps = tf.constant(eps)
        self.cce=keras.losses.CategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
        self.__name__="loss"
    def __call__(self,y_true, y_pred):
        logp = self.cce(y_true,y_pred)
        p = tf.exp(-logp)
        loss = (tf.constant(1.) - p) ** self.gamma * logp
        return tf.reduce_mean(loss, axis=-1)
    
class AddMarginLoss():
    def __init__(self,s=5,m=0.3,gamma=3):
        self.s = s
        self.m = m
        self.lossfn=FocalLoss(gamma)
        self.__name__="loss"
    def __call__(self, y_true, y_pred):
        cosine=y_pred
        phi = cosine - self.m
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        arc = (y_true * phi) + ((1.0 - y_true) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        arc *= self.s
        return self.lossfn(y_true,arc)
    
class ArcMarginLoss():
    def __init__(self,s=32,m=0.3,easy_margin=False,gamma=3, eps=1e-6):
        self.s = s
        self.m = m
        self.eps=tf.constant(eps)# This is very, very, very important
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
        self.lossfn=FocalLoss(gamma)
        self.__name__="loss"
    def __call__(self, y_true, y_pred):
        cosine=y_pred
        
        sine = tf.sqrt(tf.constant(1.) - tf.pow(cosine, 2) + self.eps)
        
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where((cosine - self.th) > 0, phi, cosine - self.mm)
        
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        arc = (y_true * phi) + ((1.0 - y_true) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        arc *= self.s
        return self.lossfn(y_true,arc)