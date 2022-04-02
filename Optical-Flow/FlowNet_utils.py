# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 13:37:30 2022

@author: YNI
"""

import tensorflow as tf
import torch

import numpy as np


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(
                padding=(kernel_size-1)//2),
            tf.keras.layers.Conv2D(
                filters=out_planes, 
                kernel_size=kernel_size, 
                strides=stride, 
                padding='valid',
                #data_format=None, 
                #dilation_rate=(1, 1), 
                #groups=1, 
                #activation=None,
                use_bias=False),
            tf.keras.layers.BatchNormalization(
                axis=-1, 
                momentum=0.1, 
                epsilon=1e-05, 
                center=True, 
                scale=True),
            tf.keras.layers.LeakyReLU(alpha=0.1)])
    else:
        return tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(
                padding=(kernel_size-1)//2),
            tf.keras.layers.Conv2D(
                filters=out_planes,
                kernel_size=kernel_size,
                strides=stride,
                padding='valid',
                #data_format=None,
                #dilation_rate=(1, 1),
                #groups=1,
                #activation=None,
                use_bias=True),
            tf.keras.layers.LeakyReLU(alpha=0.1)])
    
    
def predict_flow(in_planes):
    return tf.keras.Sequential([
        tf.keras.layers.ZeroPadding2D(padding=1),
        tf.keras.layers.Conv2D(
            filters=2,
            kernel_size=3,
            strides=1,
            padding='valid',
            #data_format=None,
            #dilation_rate=(1, 1),
            #groups=1,
            #activation=None,
            use_bias=False)])

def deconv(in_planes, out_planes):
    return tf.keras.Sequential([
        tf.keras.layers.ZeroPadding2D(padding=0),
        tf.keras.layers.Conv2DTranspose(
            filters=out_planes, 
            kernel_size=4, 
            strides=2, 
            padding='same',
            #output_padding=None, 
            #data_format=None, 
            #dilation_rate=(1, 1), 
            #activation=None,
            use_bias=False),
        tf.keras.layers.LeakyReLU(alpha=0.1)])


def crop_like(input, target):
    if input.get_shape().as_list()[1:3] == target.get_shape().as_list()[1:3]:
        return input
    else:
        target_h, target_w = target.get_shape().as_list()[1:3]
        return input[:, :target_h, :target_w, :]
    

class FlowNetS(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self,batchNorm=True):
        super(FlowNetS,self).__init__()

        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        #self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        #self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        #self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        #self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        
        self.upsampled_flow6_to_5 = tf.keras.layers.Conv2DTranspose(
                filters=2, kernel_size=4, strides=2,
                padding='same',
                use_bias=False)
        self.upsampled_flow5_to_4 = tf.keras.layers.Conv2DTranspose(
                filters=2, kernel_size=4, strides=2,
                padding='same',
                use_bias=False)
        self.upsampled_flow4_to_3 = tf.keras.layers.Conv2DTranspose(
                filters=2, kernel_size=4, strides=2,
                padding='same',
                use_bias=False)
        self.upsampled_flow3_to_2 = tf.keras.layers.Conv2DTranspose(
                filters=2, kernel_size=4, strides=2,
                padding='same',
                use_bias=False)  

        # NY: commented for now
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #        kaiming_normal_(m.weight, 0.1)
        #        if m.bias is not None:
        #            constant_(m.bias, 0)
        #    elif isinstance(m, nn.BatchNorm2d):
        #        constant_(m.weight, 1)
        #        constant_(m.bias, 0)

    def call(self, x):
        test = self.conv1(x)
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        #concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        concat5 = tf.concat([out_conv5,out_deconv5,flow6_up],-1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        #concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        concat4 = tf.concat([out_conv4,out_deconv4,flow5_up],-1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        #concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        concat3 = tf.concat([out_conv3,out_deconv3,flow4_up],-1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        #concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        concat2 = tf.concat([out_conv2,out_deconv2,flow3_up],-1)
        flow2 = self.predict_flow2(concat2)

        #if self.training:
        #    return flow2,flow3,flow4,flow5,flow6
        #else:
        #    return flow2

        return flow2
    
def load_checkpoint(model, checkpoint_PATH):
    checkpoint = torch.load(
        checkpoint_PATH, map_location=torch.device('cpu'))['state_dict']
    checkpoint = {k:v.detach().numpy() for k, v in checkpoint.items()}
    
    # conv layers
    conv1_w = np.transpose(checkpoint['conv1.0.weight'], (2,3,1,0))
    conv1_b = checkpoint['conv1.0.bias']
    conv2_w = np.transpose(checkpoint['conv2.0.weight'], (2,3,1,0))
    conv2_b = checkpoint['conv2.0.bias']
    conv3_w = np.transpose(checkpoint['conv3.0.weight'], (2,3,1,0))
    conv3_b = checkpoint['conv3.0.bias']
    conv3_1_w = np.transpose(checkpoint['conv3_1.0.weight'], (2,3,1,0))
    conv3_1_b = checkpoint['conv3_1.0.bias']
    conv4_w = np.transpose(checkpoint['conv4.0.weight'], (2,3,1,0))
    conv4_b = checkpoint['conv4.0.bias']
    conv4_1_w = np.transpose(checkpoint['conv4_1.0.weight'], (2,3,1,0))
    conv4_1_b = checkpoint['conv4_1.0.bias']
    conv5_w = np.transpose(checkpoint['conv5.0.weight'], (2,3,1,0))
    conv5_b = checkpoint['conv5.0.bias']
    conv5_1_w = np.transpose(checkpoint['conv5_1.0.weight'], (2,3,1,0))
    conv5_1_b = checkpoint['conv5_1.0.bias']
    conv6_w = np.transpose(checkpoint['conv6.0.weight'], (2,3,1,0))
    conv6_b = checkpoint['conv6.0.bias']
    conv6_1_w = np.transpose(checkpoint['conv6_1.0.weight'], (2,3,1,0))
    conv6_1_b = checkpoint['conv6_1.0.bias']

    layer_conv1 = model.layers[1].conv1.layers[1]
    layer_conv2 = model.layers[1].conv2.layers[1]
    layer_conv3 = model.layers[1].conv3.layers[1]
    layer_conv3_1 = model.layers[1].conv3_1.layers[1]
    layer_conv4 = model.layers[1].conv4.layers[1]
    layer_conv4_1 = model.layers[1].conv4_1.layers[1]
    layer_conv5 = model.layers[1].conv5.layers[1]
    layer_conv5_1 = model.layers[1].conv5_1.layers[1]
    layer_conv6 = model.layers[1].conv6.layers[1]
    layer_conv6_1 = model.layers[1].conv6_1.layers[1]
    
    layer_conv1.set_weights([conv1_w, conv1_b])
    layer_conv2.set_weights([conv2_w, conv2_b])
    layer_conv3.set_weights([conv3_w, conv3_b])
    layer_conv3_1.set_weights([conv3_1_w, conv3_1_b])
    layer_conv4.set_weights([conv4_w, conv4_b])
    layer_conv4_1.set_weights([conv4_1_w, conv4_1_b])
    layer_conv5.set_weights([conv5_w, conv5_b])
    layer_conv5_1.set_weights([conv5_1_w, conv5_1_b])
    layer_conv6.set_weights([conv6_w, conv6_b])
    layer_conv6_1.set_weights([conv6_1_w, conv6_1_b])
    
    # deconv layers
    deconv5_w = np.transpose(checkpoint['deconv5.0.weight'], (2,3,1,0))
    deconv4_w = np.transpose(checkpoint['deconv4.0.weight'], (2,3,1,0))
    deconv3_w = np.transpose(checkpoint['deconv3.0.weight'], (2,3,1,0))
    deconv2_w = np.transpose(checkpoint['deconv2.0.weight'], (2,3,1,0))
    
    layer_deconv5 = model.layers[1].deconv5.layers[1]
    layer_deconv4 = model.layers[1].deconv4.layers[1]
    layer_deconv3 = model.layers[1].deconv3.layers[1]
    layer_deconv2 = model.layers[1].deconv2.layers[1]
    
    layer_deconv5.set_weights([deconv5_w])
    layer_deconv4.set_weights([deconv4_w])
    layer_deconv3.set_weights([deconv3_w])
    layer_deconv2.set_weights([deconv2_w])
    
    # predict_flow layers
    predict_flow6_w = np.transpose(checkpoint['predict_flow6.weight'], (2,3,1,0))
    predict_flow5_w = np.transpose(checkpoint['predict_flow5.weight'], (2,3,1,0))
    predict_flow4_w = np.transpose(checkpoint['predict_flow4.weight'], (2,3,1,0))
    predict_flow3_w = np.transpose(checkpoint['predict_flow3.weight'], (2,3,1,0))
    predict_flow2_w = np.transpose(checkpoint['predict_flow2.weight'], (2,3,1,0))
    
    layer_predict_flow6 = model.layers[1].predict_flow6.layers[1]
    layer_predict_flow5 = model.layers[1].predict_flow5.layers[1]
    layer_predict_flow4 = model.layers[1].predict_flow4.layers[1]
    layer_predict_flow3 = model.layers[1].predict_flow3.layers[1]
    layer_predict_flow2 = model.layers[1].predict_flow2.layers[1]
    
    layer_predict_flow6.set_weights([predict_flow6_w])
    layer_predict_flow5.set_weights([predict_flow5_w])
    layer_predict_flow4.set_weights([predict_flow4_w])
    layer_predict_flow3.set_weights([predict_flow3_w])
    layer_predict_flow2.set_weights([predict_flow2_w])
    
    # upsampled_flow layers
    upsampled_flow6_to_5_w = np.transpose(checkpoint['upsampled_flow6_to_5.weight'], (2,3,1,0))
    upsampled_flow5_to_4_w = np.transpose(checkpoint['upsampled_flow5_to_4.weight'], (2,3,1,0))
    upsampled_flow4_to_3_w = np.transpose(checkpoint['upsampled_flow4_to_3.weight'], (2,3,1,0))
    upsampled_flow3_to_2_w = np.transpose(checkpoint['upsampled_flow3_to_2.weight'], (2,3,1,0))
    
    layer_upsampled_flow6_to_5 = model.layers[1].upsampled_flow6_to_5
    layer_upsampled_flow5_to_4 = model.layers[1].upsampled_flow5_to_4
    layer_upsampled_flow4_to_3 = model.layers[1].upsampled_flow4_to_3
    layer_upsampled_flow3_to_2 = model.layers[1].upsampled_flow3_to_2

    layer_upsampled_flow6_to_5.set_weights([upsampled_flow6_to_5_w])
    layer_upsampled_flow5_to_4.set_weights([upsampled_flow5_to_4_w])
    layer_upsampled_flow4_to_3.set_weights([upsampled_flow4_to_3_w])
    layer_upsampled_flow3_to_2.set_weights([upsampled_flow3_to_2_w])
    
    return model
    

def flow2rgb(flow_map_np, max_value=None):
    _, h, w = flow_map_np.shape

    if False:
        flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
  
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else: 
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    
    return rgb_map.clip(0,1)  
        
    
