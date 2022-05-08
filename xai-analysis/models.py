import tensorflow
from tensorflow import keras
import keras.backend as K
from keras.utils import to_categorical
from keras.models import Input, Model   
from keras.layers import concatenate
from keras.layers import Add, add, Reshape, BatchNormalization, Input, Dense, Dropout, Flatten, multiply
from keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, AveragePooling2D, AveragePooling3D, GlobalAveragePooling2D, GlobalAveragePooling3D 
from keras.layers import ReLU, PReLU, Activation 
from keras.layers.advanced_activations import LeakyReLU 
from keras import optimizers 
from keras import regularizers  
from keras.callbacks import Callback 
from keras.optimizers import Adam, SGD 
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from operator import truediv
from scipy.io import loadmat
#from keras.layers import Input
#from tensorflow.keras.models import Model

class FogNet():
    def __init__(self, input_nam_G1_shape,input_nam_G2_shape,input_nam_G3_shape,input_nam_G4_shape,input_nam_mixed_shape, input_mur_shape, filters, dropout, num_classes): 

        self.num_classes = num_classes
        self.filters = filters
        self.dropout = dropout
        self.input_nam_G1_shape    = input_nam_G1_shape       # 32*32*81
        self.input_nam_G2_shape    = input_nam_G2_shape       # 32*32*72
        self.input_nam_G3_shape    = input_nam_G3_shape       # 32*32*81
        self.input_nam_G4_shape    = input_nam_G4_shape       # 32*32*45
        self.input_nam_mixed_shape = input_nam_mixed_shape    # 32*32*9 
        self.input_mur_shape       = input_mur_shape


    def BuildModel(self):
        # Mur Data processing: 
        MUR_Conv1 = FogNetConfig.SSTDilationBlock(self.input_mur_shape)

        MUR_Conv1 = MaxPooling3D(pool_size=(2, 2, 1), strides = (3,3,1), name = 'MaxPooiling_MUR_Conv1')(MUR_Conv1) 
        MUR_Conv2 = Conv3D(filters = 32, kernel_size=(3, 3, 1), padding ='same', name = 'MUR_Conv2')(MUR_Conv1)
        MUR_Conv2 = BatchNormalization()(MUR_Conv2)
        MUR_Conv2 = PReLU()(MUR_Conv2) 
        MUR_Conv2 = MaxPooling3D(pool_size=(2, 2, 1), name = 'MaxPooiling_MUR_Conv2')(MUR_Conv2) 

        MUR_Conv3 = Conv3D(filters = 1, kernel_size=(3, 3, 1), padding ='same', name = 'MUR_Conv3')(MUR_Conv2)
        MUR_Conv3 = BatchNormalization()(MUR_Conv3) 
        MUR_Conv3 = PReLU()(MUR_Conv3) 
        MUR_Conv3 = MaxPooling3D(pool_size=(2, 2, 1), name= 'MaxPooiling_MUR_Conv3')(MUR_Conv3)   
        #======================================================================================
        # ===================================Group1 ===========================================
        #NAM_G1_Depth      = self.input_nam_G1_shape._keras_shape[3]

        NAM_G1_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G1_shape)
        NAM_Spectral_A_G1 = Conv3D(filters=self.filters, kernel_size=(1, 1, 108))(NAM_G1_Dense) 
        NAM_Spectral_A_G1 = BatchNormalization()(NAM_Spectral_A_G1)
        NAM_Spectral_A_G1 = PReLU()(NAM_Spectral_A_G1)
        NAM_Spectral_A_G1 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G1, self.filters) 


        NAM_Spatial_G1    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G1)
        NAM_Spatial_A_G1  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G1) 
        # ===================================Group2 ===========================================
        #NAM_G2_Depth      = self.input_nam_G2_shape._keras_shape[3]

        NAM_G2_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G2_shape)
        NAM_Spectral_A_G2 = Conv3D(filters=self.filters, kernel_size=(1, 1, 96))(NAM_G2_Dense) 
        NAM_Spectral_A_G2 = BatchNormalization()(NAM_Spectral_A_G2)
        NAM_Spectral_A_G2 = PReLU()(NAM_Spectral_A_G2)
        NAM_Spectral_A_G2 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G2, self.filters)

     
        NAM_Spatial_G2    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G2)
        NAM_Spatial_A_G2  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G2) 

        # ===================================Group3 ===========================================
        #NAM_G3_Depth      = self.input_nam_G3_shape._keras_shape[3]

        NAM_G3_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G3_shape)
        NAM_Spectral_A_G3 = Conv3D(filters=self.filters, kernel_size=(1, 1, 108))(NAM_G3_Dense) 
        NAM_Spectral_A_G3 = BatchNormalization()(NAM_Spectral_A_G3)
        NAM_Spectral_A_G3 = PReLU()(NAM_Spectral_A_G3)
        NAM_Spectral_A_G3 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G3, self.filters) 

        NAM_Spatial_G3    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G3)
        NAM_Spatial_A_G3  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G3) 
        # ===================================Group4 ===========================================
        #NAM_G4_Depth      = self.input_nam_G4_shape._keras_shape[3]

        NAM_G4_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G4_shape)
        NAM_Spectral_A_G4 = Conv3D(filters=self.filters, kernel_size=(1, 1, 60))(NAM_G4_Dense) 
        NAM_Spectral_A_G4 = BatchNormalization()(NAM_Spectral_A_G4)
        NAM_Spectral_A_G4 = PReLU()(NAM_Spectral_A_G4)
        NAM_Spectral_A_G4 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G4, self.filters) 

        NAM_Spatial_G4    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G4)
        NAM_Spatial_A_G4  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G4) 
        # ===================================Group5 ===========================================
        MIXED             = concatenate([self.input_nam_mixed_shape, MUR_Conv3], axis = 3)  

        #NAM_MIXED_Depth      = MIXED.shape[3]
        NAM_MIXED_Dense      = FogNetConfig.SpectralDenseBlock(MIXED)
        NAM_Spectral_A_MIXED = Conv3D(filters=self.filters, kernel_size=(1, 1, 13))(NAM_MIXED_Dense) 
        NAM_Spectral_A_MIXED = BatchNormalization()(NAM_Spectral_A_MIXED)
        NAM_Spectral_A_MIXED = PReLU()(NAM_Spectral_A_MIXED)
        NAM_Spectral_A_MIXED = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_MIXED, self.filters)

        NAM_Spatial_MIXED    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_MIXED)
        NAM_Spatial_A_MIXED  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_MIXED) 

        # ===================================Group1 ===========================================
        NAM_Spectral = concatenate([NAM_Spectral_A_G1, NAM_Spectral_A_G2, NAM_Spectral_A_G3, NAM_Spectral_A_G4, NAM_Spectral_A_MIXED], axis = 3) 
        NAM_Spectral_Multi = FogNetConfig.NAMDilationBlock(NAM_Spectral) 
        GLOBAL_NAM_Spectral_Multi = GlobalAveragePooling3D()(NAM_Spectral_Multi) 
        GLOBAL_NAM_Spectral_Multi = Dropout(self.dropout)(GLOBAL_NAM_Spectral_Multi)                       # defult = 0.4 


        NAM_Spatial = concatenate([NAM_Spatial_A_G1, NAM_Spatial_A_G2, NAM_Spatial_A_G3, NAM_Spatial_A_G4, NAM_Spatial_A_MIXED], axis = 3) 
        NAM_Spatial_Multi = FogNetConfig.NAMDilationBlock(NAM_Spatial) 
        GLOBAL_NAM_Satial_Multi = GlobalAveragePooling3D()(NAM_Spatial_Multi)
        GLOBAL_NAM_Satial_Multi = Dropout(self.dropout)(GLOBAL_NAM_Satial_Multi) # defult = 0.4 
        # Concatenate Spectral ana Spatial features
        FinalConcat = concatenate([GLOBAL_NAM_Spectral_Multi, GLOBAL_NAM_Satial_Multi], axis = 1)
        #print("The size of input flatten for classification: ", FinalConcat._keras_shape) 
        # prediction
        prediction = Dense(units = self.num_classes, activation='softmax')(FinalConcat)
        model = Model(inputs = [self.input_nam_G1_shape, self.input_nam_G2_shape, self.input_nam_G3_shape, self.input_nam_G4_shape, self.input_nam_mixed_shape, self.input_mur_shape], 
            outputs=prediction)
        #model.summary()

        return model 

#===================================================================================
class FogNetB():
    def __init__(self, categ, input_nam_G1_shape,input_nam_G2_shape,input_nam_G3_shape,input_nam_G4_shape,input_nam_mixed_shape, input_mur_shape, filters, dropout, num_classes): 
        self.categ = categ
        self.num_classes = num_classes
        self.filters = filters
        self.dropout = dropout
        self.input_nam_G1_shape    = input_nam_G1_shape       # 32*32*81
        self.input_nam_G2_shape    = input_nam_G2_shape       # 32*32*72
        self.input_nam_G3_shape    = input_nam_G3_shape       # 32*32*81
        self.input_nam_G4_shape    = input_nam_G4_shape       # 32*32*45
        self.input_nam_mixed_shape = input_nam_mixed_shape    # 32*32*9 
        self.input_mur_shape       = input_mur_shape


    def BuildModel(self):
        # Mur Data processing: 
        MUR_Conv1 = FogNetConfig.SSTDilationBlock(self.input_mur_shape)

        MUR_Conv1 = MaxPooling3D(pool_size=(2, 2, 1), strides = (3,3,1), name = 'MaxPooiling_MUR_Conv1')(MUR_Conv1) 
        MUR_Conv2 = Conv3D(filters = 32, kernel_size=(3, 3, 1), padding ='same', name = 'MUR_Conv2')(MUR_Conv1)
        MUR_Conv2 = BatchNormalization()(MUR_Conv2)
        MUR_Conv2 = PReLU()(MUR_Conv2) 
        MUR_Conv2 = MaxPooling3D(pool_size=(2, 2, 1), name = 'MaxPooiling_MUR_Conv2')(MUR_Conv2) 

        MUR_Conv3 = Conv3D(filters = 1, kernel_size=(3, 3, 1), padding ='same', name = 'MUR_Conv3')(MUR_Conv2)
        MUR_Conv3 = BatchNormalization()(MUR_Conv3) 
        MUR_Conv3 = PReLU()(MUR_Conv3) 
        MUR_Conv3 = MaxPooling3D(pool_size=(2, 2, 1), name= 'MaxPooiling_MUR_Conv3')(MUR_Conv3)   
        #======================================================================================
        # ===================================Group1 ===========================================
        NAM_G1_Depth      = self.input_nam_G1_shape._keras_shape[3]

        NAM_G1_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G1_shape)
        NAM_Spectral_A_G1 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G1_Depth))(NAM_G1_Dense) 
        NAM_Spectral_A_G1 = BatchNormalization()(NAM_Spectral_A_G1)
        NAM_Spectral_A_G1 = PReLU()(NAM_Spectral_A_G1)
        NAM_Spectral_A_G1 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G1, self.filters) 


        NAM_Spatial_G1    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G1)
        NAM_Spatial_A_G1  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G1) 
        # ===================================Group2 ===========================================
        NAM_G2_Depth      = self.input_nam_G2_shape._keras_shape[3]

        NAM_G2_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G2_shape)
        NAM_Spectral_A_G2 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G2_Depth))(NAM_G2_Dense) 
        NAM_Spectral_A_G2 = BatchNormalization()(NAM_Spectral_A_G2)
        NAM_Spectral_A_G2 = PReLU()(NAM_Spectral_A_G2)
        NAM_Spectral_A_G2 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G2, self.filters)

     
        NAM_Spatial_G2    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G2)
        NAM_Spatial_A_G2  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G2) 

        # ===================================Group3 ===========================================
        NAM_G3_Depth      = self.input_nam_G3_shape._keras_shape[3]

        NAM_G3_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G3_shape)
        NAM_Spectral_A_G3 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G3_Depth))(NAM_G3_Dense) 
        NAM_Spectral_A_G3 = BatchNormalization()(NAM_Spectral_A_G3)
        NAM_Spectral_A_G3 = PReLU()(NAM_Spectral_A_G3)
        NAM_Spectral_A_G3 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G3, self.filters) 

        NAM_Spatial_G3    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G3)
        NAM_Spatial_A_G3  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G3) 
        # ===================================Group4 ===========================================
        NAM_G4_Depth      = self.input_nam_G4_shape._keras_shape[3]

        NAM_G4_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G4_shape)
        NAM_Spectral_A_G4 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G4_Depth))(NAM_G4_Dense) 
        NAM_Spectral_A_G4 = BatchNormalization()(NAM_Spectral_A_G4)
        NAM_Spectral_A_G4 = PReLU()(NAM_Spectral_A_G4)
        NAM_Spectral_A_G4 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G4, self.filters) 

        NAM_Spatial_G4    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G4)
        NAM_Spatial_A_G4  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G4) 
        # ===================================Group5 ===========================================
        MIXED             = concatenate([self.input_nam_mixed_shape, MUR_Conv3], axis = 3)  

        NAM_MIXED_Depth      = MIXED._keras_shape[3]
        NAM_MIXED_Dense      = FogNetConfig.SpectralDenseBlock(MIXED)
        NAM_Spectral_A_MIXED = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_MIXED_Depth))(NAM_MIXED_Dense) 
        NAM_Spectral_A_MIXED = BatchNormalization()(NAM_Spectral_A_MIXED)
        NAM_Spectral_A_MIXED = PReLU()(NAM_Spectral_A_MIXED)
        NAM_Spectral_A_MIXED = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_MIXED, self.filters)

        NAM_Spatial_MIXED    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_MIXED)
        NAM_Spatial_A_MIXED  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_MIXED) 

        # ===================================concat ===========================================
        if self.categ is 'G1':
            NAM_Spectral = concatenate([NAM_Spectral_A_G2, NAM_Spectral_A_G3, NAM_Spectral_A_G4, NAM_Spectral_A_MIXED], axis = 3) 
            NAM_Spectral_Multi = FogNetConfig.NAMDilationBlock(NAM_Spectral) 
            GLOBAL_NAM_Spectral_Multi = GlobalAveragePooling3D()(NAM_Spectral_Multi) 
            GLOBAL_NAM_Spectral_Multi = Dropout(self.dropout)(GLOBAL_NAM_Spectral_Multi)                       # defult = 0.4 


            NAM_Spatial = concatenate([NAM_Spatial_A_G2, NAM_Spatial_A_G3, NAM_Spatial_A_G4, NAM_Spatial_A_MIXED], axis = 3) 
            NAM_Spatial_Multi = FogNetConfig.NAMDilationBlock(NAM_Spatial) 
            GLOBAL_NAM_Satial_Multi = GlobalAveragePooling3D()(NAM_Spatial_Multi)
            GLOBAL_NAM_Satial_Multi = Dropout(self.dropout)(GLOBAL_NAM_Satial_Multi) # defult = 0.4 
            # Concatenate Spectral ana Spatial features
            FinalConcat = concatenate([GLOBAL_NAM_Spectral_Multi, GLOBAL_NAM_Satial_Multi], axis = 1)
                #print("The size of input flatten for classification: ", FinalConcat._keras_shape) 
            # prediction
            prediction = Dense(units = self.num_classes, activation='softmax')(FinalConcat)
            model = Model(inputs = [self.input_nam_G1_shape, self.input_nam_G2_shape, self.input_nam_G3_shape, self.input_nam_G4_shape, 
            	self.input_nam_mixed_shape, self.input_mur_shape], outputs=prediction)

        elif self.categ is 'G2': 
            NAM_Spectral = concatenate([NAM_Spectral_A_G1, NAM_Spectral_A_G3, NAM_Spectral_A_G4, NAM_Spectral_A_MIXED], axis = 3) 
            NAM_Spectral_Multi = FogNetConfig.NAMDilationBlock(NAM_Spectral) 
            GLOBAL_NAM_Spectral_Multi = GlobalAveragePooling3D()(NAM_Spectral_Multi) 
            GLOBAL_NAM_Spectral_Multi = Dropout(self.dropout)(GLOBAL_NAM_Spectral_Multi)                       # defult = 0.4 


            NAM_Spatial = concatenate([NAM_Spatial_A_G1, NAM_Spatial_A_G3, NAM_Spatial_A_G4, NAM_Spatial_A_MIXED], axis = 3) 
            NAM_Spatial_Multi = FogNetConfig.NAMDilationBlock(NAM_Spatial) 
            GLOBAL_NAM_Satial_Multi = GlobalAveragePooling3D()(NAM_Spatial_Multi)
            GLOBAL_NAM_Satial_Multi = Dropout(self.dropout)(GLOBAL_NAM_Satial_Multi) # defult = 0.4 
            # Concatenate Spectral ana Spatial features
            FinalConcat = concatenate([GLOBAL_NAM_Spectral_Multi, GLOBAL_NAM_Satial_Multi], axis = 1)
            #print("The size of input flatten for classification: ", FinalConcat._keras_shape) 
            # prediction
            prediction = Dense(units = self.num_classes, activation='softmax')(FinalConcat)
            model = Model(inputs = [self.input_nam_G1_shape, self.input_nam_G2_shape, self.input_nam_G3_shape, self.input_nam_G4_shape, self.input_nam_mixed_shape, self.input_mur_shape], 
            outputs=prediction)

        elif self.categ is 'G3': 
            NAM_Spectral = concatenate([NAM_Spectral_A_G1, NAM_Spectral_A_G2,  NAM_Spectral_A_G4, NAM_Spectral_A_MIXED], axis = 3) 
            NAM_Spectral_Multi = FogNetConfig.NAMDilationBlock(NAM_Spectral) 
            GLOBAL_NAM_Spectral_Multi = GlobalAveragePooling3D()(NAM_Spectral_Multi) 
            GLOBAL_NAM_Spectral_Multi = Dropout(self.dropout)(GLOBAL_NAM_Spectral_Multi)                       # defult = 0.4 


            NAM_Spatial = concatenate([NAM_Spatial_A_G1, NAM_Spatial_A_G2,  NAM_Spatial_A_G4, NAM_Spatial_A_MIXED], axis = 3) 
            NAM_Spatial_Multi = FogNetConfig.NAMDilationBlock(NAM_Spatial) 
            GLOBAL_NAM_Satial_Multi = GlobalAveragePooling3D()(NAM_Spatial_Multi)
            GLOBAL_NAM_Satial_Multi = Dropout(self.dropout)(GLOBAL_NAM_Satial_Multi) # defult = 0.4 
            # Concatenate Spectral ana Spatial features
            FinalConcat = concatenate([GLOBAL_NAM_Spectral_Multi, GLOBAL_NAM_Satial_Multi], axis = 1)
            #print("The size of input flatten for classification: ", FinalConcat._keras_shape) 
            # prediction
            prediction = Dense(units = self.num_classes, activation='softmax')(FinalConcat)
            model = Model(inputs = [self.input_nam_G1_shape, self.input_nam_G2_shape, self.input_nam_G3_shape, self.input_nam_G4_shape, self.input_nam_mixed_shape, self.input_mur_shape], 
            outputs=prediction)

        elif self.categ is 'G4': 
            NAM_Spectral = concatenate([NAM_Spectral_A_G1, NAM_Spectral_A_G2, NAM_Spectral_A_G3,  NAM_Spectral_A_MIXED], axis = 3) 
            NAM_Spectral_Multi = FogNetConfig.NAMDilationBlock(NAM_Spectral) 
            GLOBAL_NAM_Spectral_Multi = GlobalAveragePooling3D()(NAM_Spectral_Multi) 
            GLOBAL_NAM_Spectral_Multi = Dropout(self.dropout)(GLOBAL_NAM_Spectral_Multi)                       # defult = 0.4 


            NAM_Spatial = concatenate([NAM_Spatial_A_G1, NAM_Spatial_A_G2, NAM_Spatial_A_G3, NAM_Spatial_A_MIXED], axis = 3) 
            NAM_Spatial_Multi = FogNetConfig.NAMDilationBlock(NAM_Spatial) 
            GLOBAL_NAM_Satial_Multi = GlobalAveragePooling3D()(NAM_Spatial_Multi)
            GLOBAL_NAM_Satial_Multi = Dropout(self.dropout)(GLOBAL_NAM_Satial_Multi) # defult = 0.4 
            # Concatenate Spectral ana Spatial features
            FinalConcat = concatenate([GLOBAL_NAM_Spectral_Multi, GLOBAL_NAM_Satial_Multi], axis = 1)
            #print("The size of input flatten for classification: ", FinalConcat._keras_shape) 
            # prediction
            prediction = Dense(units = self.num_classes, activation='softmax')(FinalConcat)
            model = Model(inputs = [self.input_nam_G1_shape, self.input_nam_G2_shape, self.input_nam_G3_shape, self.input_nam_G4_shape, self.input_nam_mixed_shape, self.input_mur_shape], 
            outputs=prediction)


        elif self.categ is 'G5': 
            NAM_Spectral = concatenate([NAM_Spectral_A_G1, NAM_Spectral_A_G2, NAM_Spectral_A_G3, NAM_Spectral_A_G4,], axis = 3) 
            NAM_Spectral_Multi = FogNetConfig.NAMDilationBlock(NAM_Spectral) 
            GLOBAL_NAM_Spectral_Multi = GlobalAveragePooling3D()(NAM_Spectral_Multi) 
            GLOBAL_NAM_Spectral_Multi = Dropout(self.dropout)(GLOBAL_NAM_Spectral_Multi)                       # defult = 0.4 


            NAM_Spatial = concatenate([NAM_Spatial_A_G1, NAM_Spatial_A_G2, NAM_Spatial_A_G3,  NAM_Spatial_A_G4], axis = 3) 
            NAM_Spatial_Multi = FogNetConfig.NAMDilationBlock(NAM_Spatial) 
            GLOBAL_NAM_Satial_Multi = GlobalAveragePooling3D()(NAM_Spatial_Multi)
            GLOBAL_NAM_Satial_Multi = Dropout(self.dropout)(GLOBAL_NAM_Satial_Multi) # defult = 0.4 
            # Concatenate Spectral ana Spatial features
            FinalConcat = concatenate([GLOBAL_NAM_Spectral_Multi, GLOBAL_NAM_Satial_Multi], axis = 1)
            #print("The size of input flatten for classification: ", FinalConcat._keras_shape) 
            # prediction
            prediction = Dense(units = self.num_classes, activation='softmax')(FinalConcat)
            model = Model(inputs = [self.input_nam_G1_shape, self.input_nam_G2_shape, self.input_nam_G3_shape, self.input_nam_G4_shape, self.input_nam_mixed_shape, self.input_mur_shape], 
            outputs=prediction)

        #model.summary()

        return model 
#====================================================================================
class FogNetC():
    def __init__(self, input_nam_G1_shape,input_nam_G2_shape,input_nam_G3_shape,input_nam_G4_shape,input_nam_mixed_shape, input_mur_shape, filters, dropout, num_classes): 

        self.num_classes = num_classes
        self.filters = filters
        self.dropout = dropout
        self.input_nam_G1_shape    = input_nam_G1_shape       # 32*32*81
        self.input_nam_G2_shape    = input_nam_G2_shape       # 32*32*72
        self.input_nam_G3_shape    = input_nam_G3_shape       # 32*32*81
        self.input_nam_G4_shape    = input_nam_G4_shape       # 32*32*45
        self.input_nam_mixed_shape = input_nam_mixed_shape    # 32*32*9 
        self.input_mur_shape       = input_mur_shape


    def BuildModel(self):
        # Mur Data processing: 
        MUR_Conv1 = FogNetConfig.SSTDilationBlock(self.input_mur_shape)

        MUR_Conv1 = MaxPooling3D(pool_size=(2, 2, 1), strides = (3,3,1), name = 'MaxPooiling_MUR_Conv1')(MUR_Conv1) 
        MUR_Conv2 = Conv3D(filters = 32, kernel_size=(3, 3, 1), padding ='same', name = 'MUR_Conv2')(MUR_Conv1)
        MUR_Conv2 = BatchNormalization()(MUR_Conv2)
        MUR_Conv2 = PReLU()(MUR_Conv2) 
        MUR_Conv2 = MaxPooling3D(pool_size=(2, 2, 1), name = 'MaxPooiling_MUR_Conv2')(MUR_Conv2) 

        MUR_Conv3 = Conv3D(filters = 1, kernel_size=(3, 3, 1), padding ='same', name = 'MUR_Conv3')(MUR_Conv2)
        MUR_Conv3 = BatchNormalization()(MUR_Conv3) 
        MUR_Conv3 = PReLU()(MUR_Conv3) 
        MUR_Conv3 = MaxPooling3D(pool_size=(2, 2, 1), name= 'MaxPooiling_MUR_Conv3')(MUR_Conv3)   


        # ===================================Group5 ===========================================
        MIXED             = concatenate([self.input_nam_G1_shape, self.input_nam_G2_shape, self.input_nam_G3_shape, self.input_nam_G4_shape, self.input_nam_mixed_shape, MUR_Conv3], axis = 3)  

        NAM_MIXED_Depth      = MIXED._keras_shape[3]
        NAM_MIXED_Dense      = FogNetConfig.SpectralDenseBlock(MIXED)
        NAM_Spectral_A_MIXED = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_MIXED_Depth))(NAM_MIXED_Dense) 
        NAM_Spectral_A_MIXED = BatchNormalization()(NAM_Spectral_A_MIXED)
        NAM_Spectral_A_MIXED = PReLU()(NAM_Spectral_A_MIXED)
        NAM_Spectral_A_MIXED = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_MIXED, self.filters)

        NAM_Spatial_MIXED    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_MIXED)
        NAM_Spatial_A_MIXED  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_MIXED) 

        # ===================================Group1 ===========================================
        
        NAM_Spectral_Multi = FogNetConfig.NAMDilationBlock(NAM_Spectral_A_MIXED) 
        GLOBAL_NAM_Spectral_Multi = GlobalAveragePooling3D()(NAM_Spectral_Multi) 
        GLOBAL_NAM_Spectral_Multi = Dropout(self.dropout)(GLOBAL_NAM_Spectral_Multi)                       # defult = 0.4 

        NAM_Spatial_Multi = FogNetConfig.NAMDilationBlock(NAM_Spatial_A_MIXED) 
        GLOBAL_NAM_Satial_Multi = GlobalAveragePooling3D()(NAM_Spatial_Multi)
        GLOBAL_NAM_Satial_Multi = Dropout(self.dropout)(GLOBAL_NAM_Satial_Multi) # defult = 0.4 
        # Concatenate Spectral ana Spatial features
        FinalConcat = concatenate([GLOBAL_NAM_Spectral_Multi, GLOBAL_NAM_Satial_Multi], axis = 1)
        #print("The size of input flatten for classification: ", FinalConcat._keras_shape) 
        # prediction
        prediction = Dense(units = self.num_classes, activation='softmax')(FinalConcat)
        model = Model(inputs = [self.input_nam_G1_shape, self.input_nam_G2_shape, self.input_nam_G3_shape, self.input_nam_G4_shape, self.input_nam_mixed_shape, self.input_mur_shape], 
            outputs=prediction)
        model.summary()

        return model 

#===================================================================================
#===================================================================================


class FogNet2D():
    def __init__(self, input_nam_G1_shape,input_nam_G2_shape,input_nam_G3_shape,input_nam_G4_shape,input_nam_mixed_shape, input_mur_shape, filters, dropout, num_classes): 

        self.num_classes = num_classes
        self.filters = filters
        self.dropout = dropout
        self.input_nam_G1_shape    = input_nam_G1_shape       # 32*32*81
        self.input_nam_G2_shape    = input_nam_G2_shape       # 32*32*72
        self.input_nam_G3_shape    = input_nam_G3_shape       # 32*32*81
        self.input_nam_G4_shape    = input_nam_G4_shape       # 32*32*45
        self.input_nam_mixed_shape = input_nam_mixed_shape    # 32*32*9 
        self.input_mur_shape       = input_mur_shape


    def BuildModel(self):
        # Mur Data processing: 
        MUR_Conv1 = FogNetConfig.SSTDilationBlock2D(self.input_mur_shape)

        MUR_Conv1 = MaxPooling2D(pool_size=(2, 2), strides = (3,3), name = 'MaxPooiling_MUR_Conv1')(MUR_Conv1) 
        MUR_Conv2 = Conv2D(filters = 32, kernel_size=(3, 3), padding ='same', name = 'MUR_Conv2')(MUR_Conv1)
        MUR_Conv2 = BatchNormalization()(MUR_Conv2)
        MUR_Conv2 = PReLU()(MUR_Conv2) 
        MUR_Conv2 = MaxPooling2D(pool_size=(2, 2), name = 'MaxPooiling_MUR_Conv2')(MUR_Conv2) 

        MUR_Conv3 = Conv2D(filters = 1, kernel_size=(3, 3), padding ='same', name = 'MUR_Conv3')(MUR_Conv2)
        MUR_Conv3 = BatchNormalization()(MUR_Conv3) 
        MUR_Conv3 = PReLU()(MUR_Conv3) 
        MUR_Conv3 = MaxPooling2D(pool_size=(2, 2), name= 'MaxPooiling_MUR_Conv3')(MUR_Conv3)   
        #======================================================================================
        # ===================================Group1 ===========================================
        NAM_G1_Depth      = self.input_nam_G1_shape._keras_shape[3]

        NAM_G1_Dense      = FogNetConfig.SpectralDenseBlock2D(self.input_nam_G1_shape)
        NAM_Spectral_A_G1 = Conv2D(filters=self.filters, kernel_size=(1, 1))(NAM_G1_Dense) 
        NAM_Spectral_A_G1 = BatchNormalization()(NAM_Spectral_A_G1)
        NAM_Spectral_A_G1 = PReLU()(NAM_Spectral_A_G1)
        NAM_Spectral_A_G1 = FogNetConfig.SpectralAttentionBlock2D(NAM_Spectral_A_G1, self.filters) 


        NAM_Spatial_G1    = FogNetConfig.SpatialDenseBlock2D(NAM_Spectral_A_G1)
        NAM_Spatial_A_G1  = FogNetConfig.SpatialAttentionBlock2D(NAM_Spatial_G1) 
        # ===================================Group2 ===========================================
        NAM_G2_Depth      = self.input_nam_G2_shape._keras_shape[3]

        NAM_G2_Dense      = FogNetConfig.SpectralDenseBlock2D(self.input_nam_G2_shape)
        NAM_Spectral_A_G2 = Conv2D(filters=self.filters, kernel_size=(1, 1))(NAM_G2_Dense) 
        NAM_Spectral_A_G2 = BatchNormalization()(NAM_Spectral_A_G2)
        NAM_Spectral_A_G2 = PReLU()(NAM_Spectral_A_G2)
        NAM_Spectral_A_G2 = FogNetConfig.SpectralAttentionBlock2D(NAM_Spectral_A_G2, self.filters)

     
        NAM_Spatial_G2    = FogNetConfig.SpatialDenseBlock2D(NAM_Spectral_A_G2)
        NAM_Spatial_A_G2  = FogNetConfig.SpatialAttentionBlock2D(NAM_Spatial_G2) 

        # ===================================Group3 ===========================================
        NAM_G3_Depth      = self.input_nam_G3_shape._keras_shape[3]

        NAM_G3_Dense      = FogNetConfig.SpectralDenseBlock2D(self.input_nam_G3_shape)
        NAM_Spectral_A_G3 = Conv2D(filters=self.filters, kernel_size=(1, 1))(NAM_G3_Dense) 
        NAM_Spectral_A_G3 = BatchNormalization()(NAM_Spectral_A_G3)
        NAM_Spectral_A_G3 = PReLU()(NAM_Spectral_A_G3)
        NAM_Spectral_A_G3 = FogNetConfig.SpectralAttentionBlock2D(NAM_Spectral_A_G3, self.filters) 

        NAM_Spatial_G3    = FogNetConfig.SpatialDenseBlock2D(NAM_Spectral_A_G3)
        NAM_Spatial_A_G3  = FogNetConfig.SpatialAttentionBlock2D(NAM_Spatial_G3) 
        # ===================================Group4 ===========================================
        NAM_G4_Depth      = self.input_nam_G4_shape._keras_shape[3]

        NAM_G4_Dense      = FogNetConfig.SpectralDenseBlock2D(self.input_nam_G4_shape)
        NAM_Spectral_A_G4 = Conv2D(filters=self.filters, kernel_size=(1, 1))(NAM_G4_Dense) 
        NAM_Spectral_A_G4 = BatchNormalization()(NAM_Spectral_A_G4)
        NAM_Spectral_A_G4 = PReLU()(NAM_Spectral_A_G4)
        NAM_Spectral_A_G4 = FogNetConfig.SpectralAttentionBlock2D(NAM_Spectral_A_G4, self.filters) 

        NAM_Spatial_G4    = FogNetConfig.SpatialDenseBlock2D(NAM_Spectral_A_G4)
        NAM_Spatial_A_G4  = FogNetConfig.SpatialAttentionBlock2D(NAM_Spatial_G4) 
        # ===================================Group5 ===========================================
        MIXED             = concatenate([self.input_nam_mixed_shape, MUR_Conv3], axis = 3)  

        NAM_MIXED_Depth      = MIXED._keras_shape[3]
        NAM_MIXED_Dense      = FogNetConfig.SpectralDenseBlock2D(MIXED)
        NAM_Spectral_A_MIXED = Conv2D(filters=self.filters, kernel_size=(1, 1))(NAM_MIXED_Dense) 
        NAM_Spectral_A_MIXED = BatchNormalization()(NAM_Spectral_A_MIXED)
        NAM_Spectral_A_MIXED = PReLU()(NAM_Spectral_A_MIXED)
        NAM_Spectral_A_MIXED = FogNetConfig.SpectralAttentionBlock2D(NAM_Spectral_A_MIXED, self.filters)

        NAM_Spatial_MIXED    = FogNetConfig.SpatialDenseBlock2D(NAM_Spectral_A_MIXED)
        NAM_Spatial_A_MIXED  = FogNetConfig.SpatialAttentionBlock2D(NAM_Spatial_MIXED) 

        # ===================================Group1 ===========================================
        NAM_Spectral = concatenate([NAM_Spectral_A_G1, NAM_Spectral_A_G2, NAM_Spectral_A_G3, NAM_Spectral_A_G4, NAM_Spectral_A_MIXED], axis = 3) 
        NAM_Spectral_Multi = FogNetConfig.NAMDilationBlock2D(NAM_Spectral) 
        GLOBAL_NAM_Spectral_Multi = GlobalAveragePooling2D()(NAM_Spectral_Multi) 
        GLOBAL_NAM_Spectral_Multi = Dropout(self.dropout)(GLOBAL_NAM_Spectral_Multi)                       # defult = 0.4 


        NAM_Spatial = concatenate([NAM_Spatial_A_G1, NAM_Spatial_A_G2, NAM_Spatial_A_G3, NAM_Spatial_A_G4, NAM_Spatial_A_MIXED], axis = 3) 
        NAM_Spatial_Multi = FogNetConfig.NAMDilationBlock2D(NAM_Spatial) 
        GLOBAL_NAM_Satial_Multi = GlobalAveragePooling2D()(NAM_Spatial_Multi)
        GLOBAL_NAM_Satial_Multi = Dropout(self.dropout)(GLOBAL_NAM_Satial_Multi) # defult = 0.4 
        # Concatenate Spectral ana Spatial features
        FinalConcat = concatenate([GLOBAL_NAM_Spectral_Multi, GLOBAL_NAM_Satial_Multi], axis = 1)
        #print("The size of input flatten for classification: ", FinalConcat._keras_shape) 
        # prediction
        prediction = Dense(units = self.num_classes, activation='softmax')(FinalConcat)
        model = Model(inputs = [self.input_nam_G1_shape, self.input_nam_G2_shape, self.input_nam_G3_shape, self.input_nam_G4_shape, self.input_nam_mixed_shape, self.input_mur_shape], 
            outputs=prediction)
        model.summary()

        return model 






class seq():
    def __init__(self, input_nam_G1_shape,input_nam_G2_shape,input_nam_G3_shape,input_nam_G4_shape,input_nam_mixed_shape, input_mur_shape, filters, dropout, num_classes): 

        self.num_classes = num_classes
        self.filters = filters
        self.dropout = dropout
        self.input_nam_G1_shape    = input_nam_G1_shape       # 32*32*81
        self.input_nam_G2_shape    = input_nam_G2_shape       # 32*32*72
        self.input_nam_G3_shape    = input_nam_G3_shape       # 32*32*81
        self.input_nam_G4_shape    = input_nam_G4_shape       # 32*32*45
        self.input_nam_mixed_shape = input_nam_mixed_shape    # 32*32*9 
        self.input_mur_shape       = input_mur_shape


    def BuildModel(self):
        # Mur Data processing: 
        MUR_Conv1 = FogNetConfig.SSTDilationBlock(self.input_mur_shape)

        MUR_Conv1 = MaxPooling3D(pool_size=(2, 2, 1), strides = (3,3,1), name = 'MaxPooiling_MUR_Conv1')(MUR_Conv1) 
        MUR_Conv2 = Conv3D(filters = 32, kernel_size=(3, 3, 1), padding ='same', name = 'MUR_Conv2')(MUR_Conv1)
        MUR_Conv2 = BatchNormalization()(MUR_Conv2)
        MUR_Conv2 = PReLU()(MUR_Conv2) 
        MUR_Conv2 = MaxPooling3D(pool_size=(2, 2, 1), name = 'MaxPooiling_MUR_Conv2')(MUR_Conv2) 

        MUR_Conv3 = Conv3D(filters = 1, kernel_size=(3, 3, 1), padding ='same', name = 'MUR_Conv3')(MUR_Conv2)
        MUR_Conv3 = BatchNormalization()(MUR_Conv3) 
        MUR_Conv3 = PReLU()(MUR_Conv3) 
        MUR_Conv3 = MaxPooling3D(pool_size=(2, 2, 1), name= 'MaxPooiling_MUR_Conv3')(MUR_Conv3)   
        #======================================================================================
        # ===================================Group1 ===========================================
        NAM_G1_Depth      = self.input_nam_G1_shape._keras_shape[3]

        NAM_Spectral_A_G1 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G1_Depth))(self.input_nam_G1_shape)
        NAM_Spectral_A_G1 = BatchNormalization()(NAM_Spectral_A_G1)
        NAM_Spectral_A_G1 = PReLU()(NAM_Spectral_A_G1)
        NAM_Spectral_A_G1 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G1, self.filters) 


        NAM_Spatial_G1    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G1)
        NAM_Spatial_A_G1  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G1) 
        # ===================================Group2 ===========================================
        NAM_G2_Depth      = self.input_nam_G2_shape._keras_shape[3]

        NAM_Spectral_A_G2 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G2_Depth))(self.input_nam_G2_shape)
        NAM_Spectral_A_G2 = BatchNormalization()(NAM_Spectral_A_G2)
        NAM_Spectral_A_G2 = PReLU()(NAM_Spectral_A_G2)
        NAM_Spectral_A_G2 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G2, self.filters)

     
        NAM_Spatial_G2    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G2)
        NAM_Spatial_A_G2  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G2) 

        # ===================================Group3 ===========================================
        NAM_G3_Depth      = self.input_nam_G3_shape._keras_shape[3]

        NAM_Spectral_A_G3 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G3_Depth))(self.input_nam_G3_shape)
        NAM_Spectral_A_G3 = BatchNormalization()(NAM_Spectral_A_G3)
        NAM_Spectral_A_G3 = PReLU()(NAM_Spectral_A_G3)
        NAM_Spectral_A_G3 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G3, self.filters) 

        NAM_Spatial_G3    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G3)
        NAM_Spatial_A_G3  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G3) 
        # ===================================Group4 ===========================================
        NAM_G4_Depth      = self.input_nam_G4_shape._keras_shape[3]

        NAM_Spectral_A_G4 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G4_Depth))(self.input_nam_G4_shape)
        NAM_Spectral_A_G4 = BatchNormalization()(NAM_Spectral_A_G4)
        NAM_Spectral_A_G4 = PReLU()(NAM_Spectral_A_G4)
        NAM_Spectral_A_G4 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G4, self.filters) 

        NAM_Spatial_G4    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G4)
        NAM_Spatial_A_G4  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G4) 
        # ===================================Group5 ===========================================
        MIXED             = concatenate([self.input_nam_mixed_shape, MUR_Conv3], axis = 3)  

        NAM_MIXED_Depth      = MIXED._keras_shape[3]

        NAM_Spectral_A_MIXED = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_MIXED_Depth))(MIXED) 
        NAM_Spectral_A_MIXED = BatchNormalization()(NAM_Spectral_A_MIXED)
        NAM_Spectral_A_MIXED = PReLU()(NAM_Spectral_A_MIXED)
        NAM_Spectral_A_MIXED = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_MIXED, self.filters)

        NAM_Spatial_MIXED    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_MIXED)
        NAM_Spatial_A_MIXED  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_MIXED) 

        # ===================================Group1 ===========================================
 
        NAM_Spatial = concatenate([NAM_Spatial_A_G1, NAM_Spatial_A_G2, NAM_Spatial_A_G3, NAM_Spatial_A_G4, NAM_Spatial_A_MIXED], axis = 3) 
        NAM_Spatial_Multi = FogNetConfig.NAMDilationBlock(NAM_Spatial) 
        GLOBAL_NAM_Spatial_Multi = GlobalAveragePooling3D()(NAM_Spatial_Multi)
        GLOBAL_NAM_Spatial_Multi = Dropout(self.dropout)(GLOBAL_NAM_Spatial_Multi) # defult = 0.4 
     
        #print("The size of input flatten for classification: ", FinalConcat._keras_shape) 
        # prediction
        prediction = Dense(units = self.num_classes, activation='softmax')(GLOBAL_NAM_Spatial_Multi) 
        model = Model(inputs = [self.input_nam_G1_shape, self.input_nam_G2_shape, self.input_nam_G3_shape, self.input_nam_G4_shape, self.input_nam_mixed_shape, self.input_mur_shape], 
            outputs=prediction)
        model.summary()

        return model 





class spect():
    def __init__(self, input_nam_G1_shape,input_nam_G2_shape,input_nam_G3_shape,input_nam_G4_shape,input_nam_mixed_shape, input_mur_shape, filters, dropout, num_classes): 

        self.num_classes = num_classes
        self.filters = filters
        self.dropout = dropout
        self.input_nam_G1_shape    = input_nam_G1_shape       # 32*32*81
        self.input_nam_G2_shape    = input_nam_G2_shape       # 32*32*72
        self.input_nam_G3_shape    = input_nam_G3_shape       # 32*32*81
        self.input_nam_G4_shape    = input_nam_G4_shape       # 32*32*45
        self.input_nam_mixed_shape = input_nam_mixed_shape    # 32*32*9 
        self.input_mur_shape       = input_mur_shape


    def BuildModel(self):
        # Mur Data processing: 
        MUR_Conv1 = FogNetConfig.SSTDilationBlock(self.input_mur_shape)

        MUR_Conv1 = MaxPooling3D(pool_size=(2, 2, 1), strides = (3,3,1), name = 'MaxPooiling_MUR_Conv1')(MUR_Conv1) 
        MUR_Conv2 = Conv3D(filters = 32, kernel_size=(3, 3, 1), padding ='same', name = 'MUR_Conv2')(MUR_Conv1)
        MUR_Conv2 = BatchNormalization()(MUR_Conv2)
        MUR_Conv2 = PReLU()(MUR_Conv2) 
        MUR_Conv2 = MaxPooling3D(pool_size=(2, 2, 1), name = 'MaxPooiling_MUR_Conv2')(MUR_Conv2) 

        MUR_Conv3 = Conv3D(filters = 1, kernel_size=(3, 3, 1), padding ='same', name = 'MUR_Conv3')(MUR_Conv2)
        MUR_Conv3 = BatchNormalization()(MUR_Conv3) 
        MUR_Conv3 = PReLU()(MUR_Conv3) 
        MUR_Conv3 = MaxPooling3D(pool_size=(2, 2, 1), name= 'MaxPooiling_MUR_Conv3')(MUR_Conv3)   
        #======================================================================================
        # ===================================Group1 ===========================================
        NAM_G1_Depth      = self.input_nam_G1_shape._keras_shape[3]

        NAM_Spectral_A_G1 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G1_Depth))(self.input_nam_G1_shape)
        NAM_Spectral_A_G1 = BatchNormalization()(NAM_Spectral_A_G1)
        NAM_Spectral_A_G1 = PReLU()(NAM_Spectral_A_G1)
        NAM_Spectral_A_G1 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G1, self.filters) 


   
        # ===================================Group2 ===========================================
        NAM_G2_Depth      = self.input_nam_G2_shape._keras_shape[3]

        NAM_Spectral_A_G2 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G2_Depth))(self.input_nam_G2_shape)
        NAM_Spectral_A_G2 = BatchNormalization()(NAM_Spectral_A_G2)
        NAM_Spectral_A_G2 = PReLU()(NAM_Spectral_A_G2)
        NAM_Spectral_A_G2 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G2, self.filters)


        # ===================================Group3 ===========================================
        NAM_G3_Depth      = self.input_nam_G3_shape._keras_shape[3]

        NAM_Spectral_A_G3 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G3_Depth))(self.input_nam_G3_shape)
        NAM_Spectral_A_G3 = BatchNormalization()(NAM_Spectral_A_G3)
        NAM_Spectral_A_G3 = PReLU()(NAM_Spectral_A_G3)
        NAM_Spectral_A_G3 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G3, self.filters) 

      
        # ===================================Group4 ===========================================
        NAM_G4_Depth      = self.input_nam_G4_shape._keras_shape[3]

        NAM_Spectral_A_G4 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G4_Depth))(self.input_nam_G4_shape)
        NAM_Spectral_A_G4 = BatchNormalization()(NAM_Spectral_A_G4)
        NAM_Spectral_A_G4 = PReLU()(NAM_Spectral_A_G4)
        NAM_Spectral_A_G4 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G4, self.filters) 

 
        # ===================================Group5 ===========================================
        MIXED             = concatenate([self.input_nam_mixed_shape, MUR_Conv3], axis = 3)  
        NAM_MIXED_Depth      = MIXED._keras_shape[3]
        NAM_Spectral_A_MIXED = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_MIXED_Depth))(MIXED) 
        NAM_Spectral_A_MIXED = BatchNormalization()(NAM_Spectral_A_MIXED)
        NAM_Spectral_A_MIXED = PReLU()(NAM_Spectral_A_MIXED)
        NAM_Spectral_A_MIXED = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_MIXED, self.filters)


        # ===================================Group1 ===========================================
        NAM_Spectral = concatenate([NAM_Spectral_A_G1, NAM_Spectral_A_G2, NAM_Spectral_A_G3, NAM_Spectral_A_G4, NAM_Spectral_A_MIXED], axis = 3) 
        NAM_Spectral_Multi = FogNetConfig.NAMDilationBlock(NAM_Spectral) 
        GLOBAL_NAM_Spectral_Multi = GlobalAveragePooling3D()(NAM_Spectral_Multi) 
        GLOBAL_NAM_Spectral_Multi = Dropout(self.dropout)(GLOBAL_NAM_Spectral_Multi)                       # defult = 0.4 

        # prediction
        prediction = Dense(units = self.num_classes, activation='softmax')(GLOBAL_NAM_Spectral_Multi)
        model = Model(inputs = [self.input_nam_G1_shape, self.input_nam_G2_shape, self.input_nam_G3_shape, self.input_nam_G4_shape, self.input_nam_mixed_shape, self.input_mur_shape], 
            outputs=prediction)
        model.summary()

        return model 





class parallel():
    def __init__(self, input_nam_G1_shape,input_nam_G2_shape,input_nam_G3_shape,input_nam_G4_shape,input_nam_mixed_shape, input_mur_shape, filters, dropout, num_classes): 
        self.num_classes = num_classes
        self.filters = filters
        self.dropout = dropout
        self.input_nam_G1_shape    = input_nam_G1_shape       # 32*32*81
        self.input_nam_G2_shape    = input_nam_G2_shape       # 32*32*72
        self.input_nam_G3_shape    = input_nam_G3_shape       # 32*32*81
        self.input_nam_G4_shape    = input_nam_G4_shape       # 32*32*45
        self.input_nam_mixed_shape = input_nam_mixed_shape    # 32*32*9 
        self.input_mur_shape       = input_mur_shape


    def BuildModel(self):
        # Mur Data processing: 
        MUR_Conv1 = FogNetConfig.SSTDilationBlock(self.input_mur_shape)

        MUR_Conv1 = MaxPooling3D(pool_size=(2, 2, 1), strides = (3,3,1), name = 'MaxPooiling_MUR_Conv1')(MUR_Conv1) 
        MUR_Conv2 = Conv3D(filters = 32, kernel_size=(3, 3, 1), padding ='same', name = 'MUR_Conv2')(MUR_Conv1)
        MUR_Conv2 = BatchNormalization()(MUR_Conv2)
        MUR_Conv2 = PReLU()(MUR_Conv2) 
        MUR_Conv2 = MaxPooling3D(pool_size=(2, 2, 1), name = 'MaxPooiling_MUR_Conv2')(MUR_Conv2) 

        MUR_Conv3 = Conv3D(filters = 1, kernel_size=(3, 3, 1), padding ='same', name = 'MUR_Conv3')(MUR_Conv2)
        MUR_Conv3 = BatchNormalization()(MUR_Conv3) 
        MUR_Conv3 = PReLU()(MUR_Conv3) 
        MUR_Conv3 = MaxPooling3D(pool_size=(2, 2, 1), name= 'MaxPooiling_MUR_Conv3')(MUR_Conv3)   
        #======================================================================================
        # ===================================Group1 ===========================================
        NAM_G1_Depth      = self.input_nam_G1_shape._keras_shape[3]

        NAM_G1_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G1_shape)
        NAM_Spectral_A_G1 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G1_Depth))(NAM_G1_Dense)
        NAM_Spectral_A_G1 = BatchNormalization()(NAM_Spectral_A_G1)
        NAM_Spectral_A_G1 = PReLU()(NAM_Spectral_A_G1)
        NAM_Spectral_A_G1 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G1, self.filters) 

        NAM_Spatial_G1    = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G1_Depth))(self.input_nam_G1_shape)  # defult = 24
        NAM_Spatial_G1    = FogNetConfig.SpatialDenseBlock(NAM_Spatial_G1)
        NAM_Spatial_A_G1  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G1) 
        # ===================================Group2 ===========================================
        NAM_G2_Depth      = self.input_nam_G2_shape._keras_shape[3]

        NAM_G2_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G2_shape)
        NAM_Spectral_A_G2 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G2_Depth))(NAM_G2_Dense)
        NAM_Spectral_A_G2 = BatchNormalization()(NAM_Spectral_A_G2)
        NAM_Spectral_A_G2 = PReLU()(NAM_Spectral_A_G2)
        NAM_Spectral_A_G2 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G2, self.filters)

        NAM_Spatial_G2    = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G2_Depth))(self.input_nam_G2_shape)  # defult = 24
        NAM_Spatial_G2    = FogNetConfig.SpatialDenseBlock(NAM_Spatial_G2)
        NAM_Spatial_A_G2  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G2) 

        # ===================================Group3 ===========================================
        NAM_G3_Depth      = self.input_nam_G3_shape._keras_shape[3]

        NAM_G3_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G3_shape)
        NAM_Spectral_A_G3 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G3_Depth))(NAM_G3_Dense)
        NAM_Spectral_A_G3 = BatchNormalization()(NAM_Spectral_A_G3)
        NAM_Spectral_A_G3 = PReLU()(NAM_Spectral_A_G3)
        NAM_Spectral_A_G3 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G3, self.filters) 

        NAM_Spatial_G3    = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G3_Depth))(self.input_nam_G3_shape)  # defult = 24
        NAM_Spatial_G3    = FogNetConfig.SpatialDenseBlock(NAM_Spatial_G3)
        NAM_Spatial_A_G3  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G3) 
        # ===================================Group4 ===========================================
        NAM_G4_Depth      = self.input_nam_G4_shape._keras_shape[3]

        NAM_G4_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G4_shape)
        NAM_Spectral_A_G4 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G4_Depth))(NAM_G4_Dense)
        NAM_Spectral_A_G4 = BatchNormalization()(NAM_Spectral_A_G4)
        NAM_Spectral_A_G4 = PReLU()(NAM_Spectral_A_G4)
        NAM_Spectral_A_G4 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G4, self.filters) 

        NAM_Spatial_G4    = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G4_Depth))(self.input_nam_G4_shape)  # defult = 24
        NAM_Spatial_G4    = FogNetConfig.SpatialDenseBlock(NAM_Spatial_G4)
        NAM_Spatial_A_G4  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G4) 
        # ===================================Group5 ===========================================
        MIXED             = concatenate([self.input_nam_mixed_shape, MUR_Conv3], axis = 3)  

        NAM_MIXED_Depth      = MIXED._keras_shape[3]

        NAM_MIXED_Dense      = FogNetConfig.SpectralDenseBlock(MIXED)
        NAM_Spectral_A_MIXED = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_MIXED_Depth))(NAM_MIXED_Dense) 
        NAM_Spectral_A_MIXED = BatchNormalization()(NAM_Spectral_A_MIXED)
        NAM_Spectral_A_MIXED = PReLU()(NAM_Spectral_A_MIXED)
        NAM_Spectral_A_MIXED = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_MIXED, self.filters)

        NAM_Spatial_MIXED    = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_MIXED_Depth))(MIXED)  # defult = 24
        NAM_Spatial_MIXED    = FogNetConfig.SpatialDenseBlock(NAM_Spatial_MIXED)
        NAM_Spatial_A_MIXED  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_MIXED) 

        # ===================================Group1 ===========================================
        NAM_Spectral = concatenate([NAM_Spectral_A_G1, NAM_Spectral_A_G2, NAM_Spectral_A_G3, NAM_Spectral_A_G4, NAM_Spectral_A_MIXED], axis = 3) 
        NAM_Spectral_Multi = FogNetConfig.NAMDilationBlock(NAM_Spectral) 
        GLOBAL_NAM_Spectral_Multi = GlobalAveragePooling3D()(NAM_Spectral_Multi) 
        GLOBAL_NAM_Spectral_Multi = Dropout(self.dropout)(GLOBAL_NAM_Spectral_Multi)                       # defult = 0.4 


        NAM_Spatial = concatenate([NAM_Spatial_A_G1, NAM_Spatial_A_G2, NAM_Spatial_A_G3, NAM_Spatial_A_G4, NAM_Spatial_A_MIXED], axis = 3) 
        NAM_Spatial_Multi = FogNetConfig.NAMDilationBlock(NAM_Spatial) 
        GLOBAL_NAM_Satial_Multi = GlobalAveragePooling3D()(NAM_Spatial_Multi)
        GLOBAL_NAM_Satial_Multi = Dropout(self.dropout)(GLOBAL_NAM_Satial_Multi) # defult = 0.4 
        # Concatenate Spectral ana Spatial features
        FinalConcat = concatenate([GLOBAL_NAM_Spectral_Multi, GLOBAL_NAM_Satial_Multi], axis = 1)
        #print("The size of input flatten for classification: ", FinalConcat._keras_shape) 
        # prediction
        prediction = Dense(units = self.num_classes, activation='softmax')(FinalConcat)
        model = Model(inputs = [self.input_nam_G1_shape, self.input_nam_G2_shape, self.input_nam_G3_shape, self.input_nam_G4_shape, self.input_nam_mixed_shape, self.input_mur_shape], 
            outputs=prediction)
        model.summary()

        return model 




class spat():
    def __init__(self, input_nam_G1_shape,input_nam_G2_shape,input_nam_G3_shape,input_nam_G4_shape,input_nam_mixed_shape, input_mur_shape, filters, dropout, num_classes): 
        self.num_classes = num_classes
        self.filters = filters
        self.dropout = dropout
        self.input_nam_G1_shape    = input_nam_G1_shape       # 32*32*81
        self.input_nam_G2_shape    = input_nam_G2_shape       # 32*32*72
        self.input_nam_G3_shape    = input_nam_G3_shape       # 32*32*81
        self.input_nam_G4_shape    = input_nam_G4_shape       # 32*32*45
        self.input_nam_mixed_shape = input_nam_mixed_shape    # 32*32*9 
        self.input_mur_shape       = input_mur_shape


    def BuildModel(self):
        # Mur Data processing: 
        MUR_Conv1 = FogNetConfig.SSTDilationBlock(self.input_mur_shape)

        MUR_Conv1 = MaxPooling3D(pool_size=(2, 2, 1), strides = (3,3,1), name = 'MaxPooiling_MUR_Conv1')(MUR_Conv1) 
        MUR_Conv2 = Conv3D(filters = 32, kernel_size=(3, 3, 1), padding ='same', name = 'MUR_Conv2')(MUR_Conv1)
        MUR_Conv2 = BatchNormalization()(MUR_Conv2)
        MUR_Conv2 = PReLU()(MUR_Conv2) 
        MUR_Conv2 = MaxPooling3D(pool_size=(2, 2, 1), name = 'MaxPooiling_MUR_Conv2')(MUR_Conv2) 

        MUR_Conv3 = Conv3D(filters = 1, kernel_size=(3, 3, 1), padding ='same', name = 'MUR_Conv3')(MUR_Conv2)
        MUR_Conv3 = BatchNormalization()(MUR_Conv3) 
        MUR_Conv3 = PReLU()(MUR_Conv3) 
        MUR_Conv3 = MaxPooling3D(pool_size=(2, 2, 1), name= 'MaxPooiling_MUR_Conv3')(MUR_Conv3)   
        #======================================================================================
        # ===================================Group1 ===========================================
        NAM_G1_Depth      = self.input_nam_G1_shape._keras_shape[3]

        NAM_Spatial_G1    = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G1_Depth))(self.input_nam_G1_shape)  # defult = 24
        NAM_Spatial_G1    = FogNetConfig.SpatialDenseBlock(NAM_Spatial_G1)
        NAM_Spatial_A_G1  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G1) 
        # ===================================Group2 ===========================================
        NAM_G2_Depth      = self.input_nam_G2_shape._keras_shape[3]

        NAM_Spatial_G2    = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G2_Depth))(self.input_nam_G2_shape)  # defult = 24
        NAM_Spatial_G2    = FogNetConfig.SpatialDenseBlock(NAM_Spatial_G2)
        NAM_Spatial_A_G2  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G2) 

        # ===================================Group3 ===========================================
        NAM_G3_Depth      = self.input_nam_G3_shape._keras_shape[3]


        NAM_Spatial_G3    = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G3_Depth))(self.input_nam_G3_shape)  # defult = 24
        NAM_Spatial_G3    = FogNetConfig.SpatialDenseBlock(NAM_Spatial_G3)
        NAM_Spatial_A_G3  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G3) 
        # ===================================Group4 ===========================================
        NAM_G4_Depth      = self.input_nam_G4_shape._keras_shape[3]


        NAM_Spatial_G4    = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G4_Depth))(self.input_nam_G4_shape)  # defult = 24
        NAM_Spatial_G4    = FogNetConfig.SpatialDenseBlock(NAM_Spatial_G4)
        NAM_Spatial_A_G4  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G4) 
        # ===================================Group5 ===========================================
        MIXED             = concatenate([self.input_nam_mixed_shape, MUR_Conv3], axis = 3)  

        NAM_MIXED_Depth      = MIXED._keras_shape[3]


        NAM_Spatial_MIXED    = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_MIXED_Depth))(MIXED)  # defult = 24
        NAM_Spatial_MIXED    = FogNetConfig.SpatialDenseBlock(NAM_Spatial_MIXED)
        NAM_Spatial_A_MIXED  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_MIXED) 

        # =================================== Group1 ===========================================                   
        NAM_Spatial = concatenate([NAM_Spatial_A_G1, NAM_Spatial_A_G2, NAM_Spatial_A_G3, NAM_Spatial_A_G4, NAM_Spatial_A_MIXED], axis = 3) 
        NAM_Spatial_Multi = FogNetConfig.NAMDilationBlock(NAM_Spatial) 
        GLOBAL_NAM_Satial_Multi = GlobalAveragePooling3D()(NAM_Spatial_Multi)
        GLOBAL_NAM_Satial_Multi = Dropout(self.dropout)(GLOBAL_NAM_Satial_Multi) # defult = 0.4 

        # prediction
        prediction = Dense(units = self.num_classes, activation='softmax')(GLOBAL_NAM_Satial_Multi) 
        model = Model(inputs = [self.input_nam_G1_shape, self.input_nam_G2_shape, self.input_nam_G3_shape, self.input_nam_G4_shape, self.input_nam_mixed_shape, self.input_mur_shape], 
            outputs=prediction)
        model.summary()

        return model 

class wms():
    def __init__(self, input_nam_G1_shape,input_nam_G2_shape,input_nam_G3_shape,input_nam_G4_shape,input_nam_mixed_shape, input_mur_shape, filters, dropout, num_classes): 

        self.num_classes = num_classes
        self.filters = filters
        self.dropout = dropout
        self.input_nam_G1_shape    = input_nam_G1_shape       # 32*32*81
        self.input_nam_G2_shape    = input_nam_G2_shape       # 32*32*72
        self.input_nam_G3_shape    = input_nam_G3_shape       # 32*32*81
        self.input_nam_G4_shape    = input_nam_G4_shape       # 32*32*45
        self.input_nam_mixed_shape = input_nam_mixed_shape    # 32*32*9 
        self.input_mur_shape       = input_mur_shape


    def BuildModel(self):
        # Mur Data processing: 
        MUR_Conv1 = FogNetConfig.SSTDilationBlock(self.input_mur_shape)

        MUR_Conv1 = MaxPooling3D(pool_size=(2, 2, 1), strides = (3,3,1), name = 'MaxPooiling_MUR_Conv1')(MUR_Conv1) 
        MUR_Conv2 = Conv3D(filters = 32, kernel_size=(3, 3, 1), padding ='same', name = 'MUR_Conv2')(MUR_Conv1)
        MUR_Conv2 = BatchNormalization()(MUR_Conv2)
        MUR_Conv2 = PReLU()(MUR_Conv2) 
        MUR_Conv2 = MaxPooling3D(pool_size=(2, 2, 1), name = 'MaxPooiling_MUR_Conv2')(MUR_Conv2) 

        MUR_Conv3 = Conv3D(filters = 1, kernel_size=(3, 3, 1), padding ='same', name = 'MUR_Conv3')(MUR_Conv2)
        MUR_Conv3 = BatchNormalization()(MUR_Conv3) 
        MUR_Conv3 = PReLU()(MUR_Conv3) 
        MUR_Conv3 = MaxPooling3D(pool_size=(2, 2, 1), name= 'MaxPooiling_MUR_Conv3')(MUR_Conv3)   
        #======================================================================================
        # ===================================Group1 ===========================================
        NAM_G1_Depth      = self.input_nam_G1_shape._keras_shape[3]

        NAM_G1_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G1_shape)
        NAM_Spectral_A_G1 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G1_Depth))(NAM_G1_Dense) 
        NAM_Spectral_A_G1 = BatchNormalization()(NAM_Spectral_A_G1)
        NAM_Spectral_A_G1 = PReLU()(NAM_Spectral_A_G1)
        NAM_Spectral_A_G1 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G1, self.filters) 


        NAM_Spatial_G1    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G1)
        NAM_Spatial_A_G1  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G1) 
        # ===================================Group2 ===========================================
        NAM_G2_Depth      = self.input_nam_G2_shape._keras_shape[3]

        NAM_G2_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G2_shape)
        NAM_Spectral_A_G2 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G2_Depth))(NAM_G2_Dense) 
        NAM_Spectral_A_G2 = BatchNormalization()(NAM_Spectral_A_G2)
        NAM_Spectral_A_G2 = PReLU()(NAM_Spectral_A_G2)
        NAM_Spectral_A_G2 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G2, self.filters)

     
        NAM_Spatial_G2    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G2)
        NAM_Spatial_A_G2  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G2) 

        # ===================================Group3 ===========================================
        NAM_G3_Depth      = self.input_nam_G3_shape._keras_shape[3]

        NAM_G3_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G3_shape)
        NAM_Spectral_A_G3 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G3_Depth))(NAM_G3_Dense) 
        NAM_Spectral_A_G3 = BatchNormalization()(NAM_Spectral_A_G3)
        NAM_Spectral_A_G3 = PReLU()(NAM_Spectral_A_G3)
        NAM_Spectral_A_G3 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G3, self.filters) 

        NAM_Spatial_G3    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G3)
        NAM_Spatial_A_G3  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G3) 
        # ===================================Group4 ===========================================
        NAM_G4_Depth      = self.input_nam_G4_shape._keras_shape[3]

        NAM_G4_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G4_shape)
        NAM_Spectral_A_G4 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G4_Depth))(NAM_G4_Dense) 
        NAM_Spectral_A_G4 = BatchNormalization()(NAM_Spectral_A_G4)
        NAM_Spectral_A_G4 = PReLU()(NAM_Spectral_A_G4)
        NAM_Spectral_A_G4 = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_G4, self.filters) 

        NAM_Spatial_G4    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G4)
        NAM_Spatial_A_G4  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_G4) 
        # ===================================Group5 ===========================================
        MIXED             = concatenate([self.input_nam_mixed_shape, MUR_Conv3], axis = 3)  

        NAM_MIXED_Depth      = MIXED._keras_shape[3]
        NAM_MIXED_Dense      = FogNetConfig.SpectralDenseBlock(MIXED)
        NAM_Spectral_A_MIXED = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_MIXED_Depth))(NAM_MIXED_Dense) 
        NAM_Spectral_A_MIXED = BatchNormalization()(NAM_Spectral_A_MIXED)
        NAM_Spectral_A_MIXED = PReLU()(NAM_Spectral_A_MIXED)
        NAM_Spectral_A_MIXED = FogNetConfig.SpectralAttentionBlock(NAM_Spectral_A_MIXED, self.filters)

        NAM_Spatial_MIXED    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_MIXED)
        NAM_Spatial_A_MIXED  = FogNetConfig.SpatialAttentionBlock(NAM_Spatial_MIXED) 

        # ===================================Group1 ===========================================
        NAM_Spectral = concatenate([NAM_Spectral_A_G1, NAM_Spectral_A_G2, NAM_Spectral_A_G3, NAM_Spectral_A_G4, NAM_Spectral_A_MIXED], axis = 3) 
        GLOBAL_NAM_Spectral_Multi = GlobalAveragePooling3D()(NAM_Spectral) 
        GLOBAL_NAM_Spectral_Multi = Dropout(self.dropout)(GLOBAL_NAM_Spectral_Multi)                       # defult = 0.4 


        NAM_Spatial = concatenate([NAM_Spatial_A_G1, NAM_Spatial_A_G2, NAM_Spatial_A_G3, NAM_Spatial_A_G4, NAM_Spatial_A_MIXED], axis = 3) 
        GLOBAL_NAM_Satial_Multi = GlobalAveragePooling3D()(NAM_Spatial)
        GLOBAL_NAM_Satial_Multi = Dropout(self.dropout)(GLOBAL_NAM_Satial_Multi) # defult = 0.4 
        # Concatenate Spectral ana Spatial features
        FinalConcat = concatenate([GLOBAL_NAM_Spectral_Multi, GLOBAL_NAM_Satial_Multi], axis = 1)
        #print("The size of input flatten for classification: ", FinalConcat._keras_shape) 
        # prediction
        prediction = Dense(units = self.num_classes, activation='softmax')(FinalConcat)
        model = Model(inputs = [self.input_nam_G1_shape, self.input_nam_G2_shape, self.input_nam_G3_shape, self.input_nam_G4_shape, self.input_nam_mixed_shape, self.input_mur_shape], 
            outputs=prediction)
        model.summary()

        return model 


class wam():
    def __init__(self, input_nam_G1_shape,input_nam_G2_shape,input_nam_G3_shape,input_nam_G4_shape,input_nam_mixed_shape, input_mur_shape, filters, dropout, num_classes): 

        self.num_classes = num_classes
        self.filters = filters
        self.dropout = dropout
        self.input_nam_G1_shape    = input_nam_G1_shape       # 32*32*81
        self.input_nam_G2_shape    = input_nam_G2_shape       # 32*32*72
        self.input_nam_G3_shape    = input_nam_G3_shape       # 32*32*81
        self.input_nam_G4_shape    = input_nam_G4_shape       # 32*32*45
        self.input_nam_mixed_shape = input_nam_mixed_shape    # 32*32*9 
        self.input_mur_shape       = input_mur_shape


    def BuildModel(self):
        # Mur Data processing: 
        MUR_Conv1 = FogNetConfig.SSTDilationBlock(self.input_mur_shape)

        MUR_Conv1 = MaxPooling3D(pool_size=(2, 2, 1), strides = (3,3,1), name = 'MaxPooiling_MUR_Conv1')(MUR_Conv1) 
        MUR_Conv2 = Conv3D(filters = 32, kernel_size=(3, 3, 1), padding ='same', name = 'MUR_Conv2')(MUR_Conv1)
        MUR_Conv2 = BatchNormalization()(MUR_Conv2)
        MUR_Conv2 = PReLU()(MUR_Conv2) 
        MUR_Conv2 = MaxPooling3D(pool_size=(2, 2, 1), name = 'MaxPooiling_MUR_Conv2')(MUR_Conv2) 

        MUR_Conv3 = Conv3D(filters = 1, kernel_size=(3, 3, 1), padding ='same', name = 'MUR_Conv3')(MUR_Conv2)
        MUR_Conv3 = BatchNormalization()(MUR_Conv3) 
        MUR_Conv3 = PReLU()(MUR_Conv3) 
        MUR_Conv3 = MaxPooling3D(pool_size=(2, 2, 1), name= 'MaxPooiling_MUR_Conv3')(MUR_Conv3)   
        #======================================================================================
        # ===================================Group1 ===========================================
        NAM_G1_Depth      = self.input_nam_G1_shape._keras_shape[3]

        NAM_G1_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G1_shape)
        NAM_Spectral_A_G1 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G1_Depth))(NAM_G1_Dense) 
        NAM_Spectral_A_G1 = BatchNormalization()(NAM_Spectral_A_G1)
        NAM_Spectral_A_G1 = PReLU()(NAM_Spectral_A_G1)


        NAM_Spatial_G1    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G1)
        # ===================================Group2 ===========================================
        NAM_G2_Depth      = self.input_nam_G2_shape._keras_shape[3]

        NAM_G2_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G2_shape)
        NAM_Spectral_A_G2 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G2_Depth))(NAM_G2_Dense) 
        NAM_Spectral_A_G2 = BatchNormalization()(NAM_Spectral_A_G2)
        NAM_Spectral_A_G2 = PReLU()(NAM_Spectral_A_G2)

     
        NAM_Spatial_G2    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G2)

        # ===================================Group3 ===========================================
        NAM_G3_Depth      = self.input_nam_G3_shape._keras_shape[3]

        NAM_G3_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G3_shape)
        NAM_Spectral_A_G3 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G3_Depth))(NAM_G3_Dense) 
        NAM_Spectral_A_G3 = BatchNormalization()(NAM_Spectral_A_G3)
        NAM_Spectral_A_G3 = PReLU()(NAM_Spectral_A_G3)

        NAM_Spatial_G3    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G3)
        # ===================================Group4 ===========================================
        NAM_G4_Depth      = self.input_nam_G4_shape._keras_shape[3]

        NAM_G4_Dense      = FogNetConfig.SpectralDenseBlock(self.input_nam_G4_shape)
        NAM_Spectral_A_G4 = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_G4_Depth))(NAM_G4_Dense) 
        NAM_Spectral_A_G4 = BatchNormalization()(NAM_Spectral_A_G4)
        NAM_Spectral_A_G4 = PReLU()(NAM_Spectral_A_G4)

        NAM_Spatial_G4    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_G4)
        # ===================================Group5 ===========================================
        MIXED             = concatenate([self.input_nam_mixed_shape, MUR_Conv3], axis = 3)  

        NAM_MIXED_Depth      = MIXED._keras_shape[3]
        NAM_MIXED_Dense      = FogNetConfig.SpectralDenseBlock(MIXED)
        NAM_Spectral_A_MIXED = Conv3D(filters=self.filters, kernel_size=(1, 1, NAM_MIXED_Depth))(NAM_MIXED_Dense) 
        NAM_Spectral_A_MIXED = BatchNormalization()(NAM_Spectral_A_MIXED)
        NAM_Spectral_A_MIXED = PReLU()(NAM_Spectral_A_MIXED)

        NAM_Spatial_MIXED    = FogNetConfig.SpatialDenseBlock(NAM_Spectral_A_MIXED)

        # ===================================Group1 ===========================================
        NAM_Spectral = concatenate([NAM_Spectral_A_G1, NAM_Spectral_A_G2, NAM_Spectral_A_G3, NAM_Spectral_A_G4, NAM_Spectral_A_MIXED], axis = 3) 
        NAM_Spectral_Multi = FogNetConfig.NAMDilationBlock(NAM_Spectral) 
        GLOBAL_NAM_Spectral_Multi = GlobalAveragePooling3D()(NAM_Spectral_Multi) 
        GLOBAL_NAM_Spectral_Multi = Dropout(self.dropout)(GLOBAL_NAM_Spectral_Multi)                       # defult = 0.4 


        NAM_Spatial = concatenate([NAM_Spatial_G1, NAM_Spatial_G2, NAM_Spatial_G3, NAM_Spatial_G4, NAM_Spatial_MIXED], axis = 3) 
        NAM_Spatial_Multi = FogNetConfig.NAMDilationBlock(NAM_Spatial) 
        GLOBAL_NAM_Satial_Multi = GlobalAveragePooling3D()(NAM_Spatial_Multi)
        GLOBAL_NAM_Satial_Multi = Dropout(self.dropout)(GLOBAL_NAM_Satial_Multi) # defult = 0.4 
        # Concatenate Spectral ana Spatial features
        FinalConcat = concatenate([GLOBAL_NAM_Spectral_Multi, GLOBAL_NAM_Satial_Multi], axis = 1)
        #print("The size of input flatten for classification: ", FinalConcat._keras_shape) 
        # prediction
        prediction = Dense(units = self.num_classes, activation='softmax')(FinalConcat)
        model = Model(inputs = [self.input_nam_G1_shape, self.input_nam_G2_shape, self.input_nam_G3_shape, self.input_nam_G4_shape, self.input_nam_mixed_shape, self.input_mur_shape], 
            outputs=prediction)
        model.summary()

        return model 
