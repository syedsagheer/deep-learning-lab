from __future__ import division
import os
import time
import math
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils

import tensorflow.contrib as tc 

from layers_slim import *



def FCN_Seg(self, is_training=True):

    #Set training hyper-parameters
    self.is_training = is_training
    self.normalizer = tc.layers.batch_norm
    self.bn_params = {'is_training': self.is_training}

    print("i'm in FCN")  
    print("input", self.tgt_image)

    with tf.variable_scope('First_conv'):
        conv1 = tc.layers.conv2d(self.tgt_image, 32, 3, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params)

        print("Conv1 shape")
        print(conv1.get_shape())
    x = inverted_bottleneck(conv1, 1, 16, 0,self.normalizer, self.bn_params, 1)
    #print("Conv 1")
    print(x.get_shape())
    x = inverted_bottleneck(x, 6, 24, 1,self.normalizer, self.bn_params, 2) 
    x = inverted_bottleneck(x, 6, 24, 0,self.normalizer, self.bn_params, 3)
   
    print("Block One dim ")
    print(x)

    DB2_skip_connection = x    
    x = inverted_bottleneck(x, 6, 32, 1,self.normalizer, self.bn_params, 4)
    print(x)
    x = inverted_bottleneck(x, 6, 32, 0,self.normalizer, self.bn_params, 5)
    
    print("Block Two dim ")
    print(x)

    DB3_skip_connection = x
    x = inverted_bottleneck(x, 6, 64, 1,self.normalizer, self.bn_params, 6)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 7)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 8)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 9)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 10)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 11)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 12)
    
    print("Block Three dim ")
    print(x)
    DB4_skip_connection = x
    #23x23x160
    x = inverted_bottleneck(x, 6, 160, 1,self.normalizer, self.bn_params, 13)
    x = inverted_bottleneck(x, 6, 160, 0,self.normalizer, self.bn_params, 14)
    x = inverted_bottleneck(x, 6, 160, 0,self.normalizer, self.bn_params, 15)
    print("Block Four dim ")
    print(x)
    x = inverted_bottleneck(x, 6, 320, 0,self.normalizer, self.bn_params, 16)
    
    print("Block Four dim ")
    print(x)
    

    # Configuration 1 - single upsampling layer
    if self.configuration == 1:

        #input is features named 'x'

        # TODO(1.1) - incorporate a upsample function which takes the features of x 
        # and produces 120 output feature maps, which are 16x bigger in resolution than 
        # x. Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up5
        
        
        current_up5=TransitionUp_elu(x,120,16,"Net_prediction")
        print(current_up5)
        if (current_up5.shape[1] != self.tgt_image.shape[1]):
            current_up5=crop(current_up5,self.tgt_image)
            print("implementing a upsampling feature from layers_slim")
            End_maps_decoder1 = slim.conv2d(current_up5, self.N_classes, [1, 1], scope='Final') #(batchsize, width, height, N_classes)
            print(End_maps_decoder1)
            Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))
            print("End map size Decoder: ")
            print(Reshaped_map)
    
        else:
            current_up5   
            print("implementing a upsampling feature from layers_slim")
            End_maps_decoder1 = slim.conv2d(current_up5, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)
            print(End_maps_decoder1)
        
            Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

            print("End map size Decoder: ")
            print(Reshaped_map)

    # Configuration 2 - single upsampling layer plus skip connection
    if self.configuration == 2:

        #input is features named 'x'

        # TODO (2.1) - implement the refinement block which upsample the data 2x like in configuration 1 
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps

    
        print("current_up3")
        current_up3=TransitionUp_elu(x,120,2,"transition2")
       
        print(DB4_skip_connection)
        
        
        if (current_up3.shape[1]!= DB4_skip_connection.shape[1]):
            
            current_up3=crop(DB4_skip_connection,current_up3)
            current_up3=Concat_layers(current_up3,DB4_skip_connection, nm='test2')#38*38*192
            current_up3=Convolution(current_up3, 256, 3, "2_one_conv")
            print("current_up3")
            print(current_up3)
            
            current_up3=TransitionUp_elu(x,120,8,"2_TU")
            print(current_up3)
            if (current_up3.shape[1] != self.tgt_image.shape[1]):
                current_up3=crop(current_up3,self.tgt_image)
                print("implementing a upsampling feature from layers_slim")

                print("current_up3")
                print(current_up3)
            else:
                
                current_up3    
            
        else:
            
            current_up3=Concat_layers(current_up3,DB4_skip_connection, nm='test2')
            current_up3=Convolution(current_up3, 256, 3, "test2")
            current_up3=TransitionUp_elu(current_up3,120,8,"final2")
            print("current_up3")
            print(current_up3)
            
            
            print(current_up3)
            if (current_up3.shape[1] != self.tgt_image.shape[1]):
                print(current_up3)
                print(self.tgt_image)
                
                current_up3=crop(current_up3,self.tgt_image)
                print("implementing a upsampling feature from layers_slim")

                print("current_up3")
                print(current_up3)
                
            else:
                current_up3
            
        End_maps_decoder1 = slim.conv2d(current_up3, self.N_classes, [1, 1], scope='Final_decoder')
        
        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)

        # TODO (2.2) - incorporate a upsample function which takes the features from TODO (2.1) 
        # and produces 120 output feature maps, which are 8x bigger in resolution than 
        # TODO (2.1). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up3

        
        
    # Configuration 3 - Two upsampling layer plus skip connection
    if self.configuration == 3:

        #input is features named 'x'

        # TODO (3.1) - implement the refinement block which upsample the data 2x like in configuration 1 
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps

        
        print("current_up4_a")
        current_up4_a=TransitionUp_elu(x,120,2,"3_2_times")
       
        print(DB4_skip_connection)
        
        if (current_up4_a.shape[1]!= DB4_skip_connection.shape[1]):
            
            current_up4_a=crop(DB4_skip_connection,current_up4_a)
            current_up4_a=Concat_layers(current_up4_a,DB4_skip_connection, nm='test3_a_concat')
            current_up4_a=Convolution(current_up4_a, 256, 3, "3_a_concat")
            
        else:
            
            current_up4_a=Concat_layers(current_up4_a,DB4_skip_connection, nm='test3_a_concat')
            current_up4_a=Convolution(current_up4_a, 256, 3, "3_a_concat")
                  


        # TODO (3.2) - Repeat TODO(3.1) now producing 160 output feature maps and fusing the upsampled features 
        # with the corresponding skip connection (DB3_skip_connection) through concatenation.

       # current_up4_b=TransitionUp_elu(current_up4_a,120,2,"3_2times")
        print("DB3 connection")   
        print(DB3_skip_connection)
        print("current_up4_b")
        current_up4_b=TransitionUp_elu(current_up4_a,160,2,"3_4times")
        
        
        if (current_up4_b.shape[1]!= DB3_skip_connection.shape[1]):
            
            current_up4_b=crop(current_up4_b,DB3_skip_connection)
            print("concatenated shape is")
            current_up4_b=Concat_layers(current_up4_b,DB3_skip_connection, nm='test3_a_concat')       
            
            
        else:
            
            current_up4_b=Concat_layers(current_up4_a,DB4_skip_connection, nm='test3_a_concat')
            current_up4_b=Convolution(current_up4_a, 256, 3, "3_a_concat")


        # TODO (3.3) - incorporate a upsample function which takes the features from TODO (3.2)  
        # and produces 120 output feature maps which are 4x bigger in resolution than 
        # TODO (3.2). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up4  
        
        
        
        current_up4=TransitionUp_elu(current_up4_b,120,4,"3_final4times")
        print("current_up4")
        print(current_up4)


        if (current_up4.shape[1] != self.tgt_image.shape[1]):
            current_up4=crop(current_up3,self.tgt_image)
            print("implementing a upsampling feature from layers_slim")

            print("current_up4")
            print(current_up4)
        else:
                
            current_up4              
                    

        End_maps_decoder1 = slim.conv2d(current_up4, self.N_classes, [1, 1], scope='Final_decoder_3') #(batchsize, width, height, N_classes)
        
        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    #Full configuration 
    if self.configuration == 4:

        ######################################################################################
        ######################################### DECODER Full #############################################

       
        
        # TODO (4.1) - implement the refinement block which upsample the data 2x like in configuration 1 
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps
        
        print("current_up5_a")
        current_up5_a=TransitionUp_elu(x,120,2,"4_2times")
       
        print(DB4_skip_connection)
        
        if (current_up5_a.shape[1]!= DB4_skip_connection.shape[1]):
            
            current_up5_a=crop(DB5_skip_connection,current_up5_a)
            current_up5_a=Concat_layers(current_up5_a,DB4_skip_connection, nm='test3_a_concat')
            current_up5_a=Convolution(current_up5_a, 256, 3, "3_a_concat")
            
        else:
            
            current_up5_a=Concat_layers(current_up5_a,DB4_skip_connection, nm='test3_a_concat')
            current_up5_a=Convolution(current_up5_a, 256, 3, "3_a_concat")
       
        
       
        # TODO (4.2) - Repeat TODO(4.1) now producing 160 output feature maps and fusing the upsampled features 
        # with the corresponding skip connection (DB3_skip_connection) through concatenation.

        print("DB3 connection")   
        print(DB3_skip_connection)
        print("current_up4_b")
        current_up5_b=TransitionUp_elu(current_up5_a,160,2,"4_2_1times")
        
        
        if (current_up5_b.shape[1]!= DB3_skip_connection.shape[1]):
            
            current_up5_b=crop(current_up5_b,DB3_skip_connection)
            print("concatenated shape is")
            current_up5_b=Concat_layers(current_up5_b,DB3_skip_connection, nm='test4_a_concat')
            print(current_up5_b)            
            
            
        else:
            print("im in the else2 loop")
            current_up5_b=Concat_layers(current_up5_b,DB3_skip_connection, nm='test4_a_concat')
            current_up5_b=Convolution(current_up5_b, 256, 3, "test4_a_concat")


        # TODO (4.3) - Repeat TODO(4.2) now producing 96 output feature maps and fusing the upsampled features 
        # with the corresponding skip connection (DB2_skip_connection) through concatenation.
        print("DB2 connection")   
        print(DB2_skip_connection)
        print("current_up4_b")
        current_up5_c=TransitionUp_elu(current_up5_b,96,2,"4_2_2times")
        
        
        if (current_up5_c.shape[1]!= DB2_skip_connection.shape[1]):
            
            current_up5_c=crop(current_up5_c,DB2_skip_connection)
            print("concatenated shape is")
            current_up5_c=Concat_layers(current_up5_c,DB2_skip_connection, nm='test4_c_concat')
                      
            
        else:
            print("im in the else2 loop")
            current_up5_c=Concat_layers(current_up5_c,DB2_skip_connection, nm='test4_c_concat')
                  

        # TODO (4.4) - incorporate a upsample function which takes the features from TODO(4.3) 
        # and produce 120 output feature maps which are 2x bigger in resolution than 
        # TODO(4.3). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up4


        current_up5=TransitionUp_elu(current_up5_c,120,2,"3_final4times")
        print("current_up5")
        print(current_up5)



        if (current_up5.shape[1] != self.tgt_image.shape[1]):
            current_up5=crop(current_up5,self.tgt_image)
            print("implementing a upsampling feature from layers_slim")

            print("current_up4")
            print(current_up5)
        else:
                
            current_up5       
        
        
        End_maps_decoder1 = slim.conv2d(current_up5, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)
        
        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)

    
    return Reshaped_map

