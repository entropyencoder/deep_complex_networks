#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Olexa Bilaniuk, Chiheb Trabelsi, Sandeep Subramanian

# Imports
import complexnn
from   complexnn                             import ComplexBN,\
                                                    ComplexConv1D,\
                                                    ComplexConv2D,\
                                                    ComplexConv3D,\
                                                    ComplexConvConcat2D,\
                                                    ComplexDense
#                                                    FFT,IFFT,FFT2,IFFT2,\
#                                                    SpectralPooling1D,SpectralPooling2D
from complexnn import GetImag, GetReal
import h5py                                  as     H
import keras
from   keras.callbacks                       import Callback, ModelCheckpoint, LearningRateScheduler
from   keras.datasets                        import cifar10, cifar100, mnist, fashion_mnist
from   keras.initializers                    import Orthogonal
from   keras.layers                          import Layer, AveragePooling2D, AveragePooling3D, add, Add, concatenate, Concatenate, Input, Flatten, Dense, Convolution2D, BatchNormalization, Activation, Reshape, ConvLSTM2D, Conv2D
from   keras.models                          import Model, load_model, save_model
from   keras.optimizers                      import SGD, Adam, RMSprop
from   keras.preprocessing.image             import ImageDataGenerator
from   keras.regularizers                    import l2
from   keras.utils.np_utils                  import to_categorical
import keras.backend                         as     K
import keras.models                          as     KM
from   kerosene.datasets                     import svhn2
import logging                               as     L
import numpy                                 as     np
import os, pdb, socket, sys, time
#import theano                                as     T
import tensorflow                                as     T
from keras.layers import SeparableConv2D
from keras.layers import Lambda

# Disallow eager use of GPU memory
if 'tensorflow' == K.backend():
	import tensorflow as tf
	from keras.backend.tensorflow_backend import set_session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.visible_device_list = "0"
	set_session(tf.Session(config=config))


def ComplexJointReLU(I):
	I_real = I[:, :(I.shape[1]//2), :, :]
	I_imag = I[:, (I.shape[1]//2):, :, :]

	I_cond_real = tf.logical_or(I_real>=0, I_imag>=0)
	I_cond_imag = tf.logical_or(I_real>=0, I_imag>=0)

	I_real = tf.where(I_cond_real, I_real, tf.zeros_like(I_real))
	I_imag = tf.where(I_cond_imag, I_imag, tf.zeros_like(I_imag))

	O = Concatenate(axis=1)([I_real, I_imag])

	return O

#
# Residual Net Utilities
#

def learnConcatRealImagBlock(I, filter_size, featmaps, stage, block, convArgs, bnArgs, d):
	"""Learn initial imaginary component for input."""

	conv_name_base = 'res'+str(stage)+block+'_branch'
	bn_name_base   = 'bn' +str(stage)+block+'_branch'

	O = BatchNormalization(name=bn_name_base+'2a', **bnArgs)(I)
	O = Activation(d.act)(O)
	O = Convolution2D(featmaps[0], filter_size,
	                  name               = conv_name_base+'2a',
	                  padding            = 'same',
	                  kernel_initializer = 'he_normal',
	                  use_bias           = False,
	                  kernel_regularizer = l2(0.0001))(O)

	O = BatchNormalization(name=bn_name_base+'2b', **bnArgs)(O)
	O = Activation(d.act)(O)
	O = Convolution2D(featmaps[1], filter_size,
	                  name               = conv_name_base+'2b',
	                  padding            = 'same',
	                  kernel_initializer = 'he_normal',
	                  use_bias           = False,
	                  kernel_regularizer = l2(0.0001))(O)

	return O

def getResidualBlock(I, filter_size, featmaps, stage, block, shortcut, convArgs, convArgs_real, bnArgs, d):
	"""Get residual block."""

	activation           = d.act
	drop_prob            = d.dropout
	nb_fmaps1, nb_fmaps2 = featmaps
	conv_name_base       = 'res'+str(stage)+block+'_branch'
	bn_name_base         = 'bn' +str(stage)+block+'_branch'
	if K.image_data_format() == 'channels_first' and K.ndim(I) != 3:
		channel_axis = 1
	else:
		channel_axis = -1


	# if   d.model == "real":
	if "real" in d.model:
		O = BatchNormalization(name=bn_name_base+'_2a', **bnArgs)(I)
	# elif d.model == "complex":
	elif "complex" in d.model:
		O = ComplexBN(name=bn_name_base+'_2a', **bnArgs)(I)

	if d.aact=="complex_joint_relu":
		O = Lambda(ComplexJointReLU)(O)
	else:
		O = Activation(activation)(O)

	if shortcut == 'regular' or d.spectral_pool_scheme == "nodownsample":
		if   d.model == "real":
			O = Conv2D(nb_fmaps1, filter_size, name=conv_name_base+'2a', **convArgs)(O)
		elif d.model == "real_dws":
			O = SeparableConv2D(nb_fmaps1, filter_size, name=conv_name_base + '2a', **convArgs)(O)
		elif (d.model == "real_group") or (d.model=="real_group_pwc_full") or (d.model=="real_group_pwc_group"):
			O_g0 = Lambda(lambda O: O[:,:(O.shape[1]//2),:,:])(O)
			O_g1 = Lambda(lambda O: O[:,(O.shape[1]//2):,:,:])(O)
			O_g0 = Conv2D(nb_fmaps1 // 2, filter_size, name=conv_name_base + '2a_g0', **convArgs)(O_g0)
			O_g1 = Conv2D(nb_fmaps1 // 2, filter_size, name=conv_name_base + '2a_g1', **convArgs)(O_g1)
			O_g00 = Lambda(lambda O_g0: O_g0[:,:(O_g0.shape[1]//2),:,:])(O_g0)
			O_g01 = Lambda(lambda O_g0: O_g0[:,(O_g0.shape[1]//2):,:,:])(O_g0)
			O_g10 = Lambda(lambda O_g1: O_g1[:,:(O_g1.shape[1]//2),:,:])(O_g1)
			O_g11 = Lambda(lambda O_g1: O_g1[:,(O_g1.shape[1]//2):,:,:])(O_g1)
			O = Concatenate(axis=1)([O_g00, O_g11, O_g01, O_g10])	# This ordering allows permutation of odd-numbered outputs (O_g0, O_g1).
			if d.model == "real_group_pwc_full":
				O = Conv2D(nb_fmaps1, (1, 1), name=conv_name_base + '2a_pwc', **convArgs)(O)
			elif d.model == "real_group_pwc_group":
				O_g0 = Lambda(lambda O: O[:, :(O.shape[1] // 2), :, :])(O)
				O_g1 = Lambda(lambda O: O[:, (O.shape[1] // 2):, :, :])(O)
				O_g0 = Conv2D(int(O.shape[1] // 2), (1, 1), name=conv_name_base + '2a_pwc_g0', **convArgs_real)(O_g0)
				O_g1 = Conv2D(int(O.shape[1] // 2), (1, 1), name=conv_name_base + '2a_pwc_g1', **convArgs_real)(O_g1)
				O = Concatenate(axis=1)([O_g0, O_g1])
		elif d.model == "complex":
			O = ComplexConv2D(nb_fmaps1, filter_size, name=conv_name_base+'2a', **convArgs)(O)
		elif (d.model == "complex_concat") or (d.model=="complex_concat_pwc_group"):
			O = ComplexConvConcat2D(nb_fmaps1 // 2, filter_size, name=conv_name_base + '2a', **convArgs)(O)
			if d.model == "complex_concat_pwc_group":
				O_g0 = Lambda(lambda O: O[:, :(O.shape[1] // 2), :, :])(O)
				O_g1 = Lambda(lambda O: O[:, (O.shape[1] // 2):, :, :])(O)
				O_g0 = Conv2D(int(O.shape[1] // 2), (1, 1), name=conv_name_base + '2a_pwc_g0', **convArgs_real)(O_g0)
				O_g1 = Conv2D(int(O.shape[1] // 2), (1, 1), name=conv_name_base + '2a_pwc_g1', **convArgs_real)(O_g1)
				O = Concatenate(axis=1)([O_g0, O_g1])
		else:
			print("Error: unknown model type")
			exit(-1)
	elif shortcut == 'projection':
		if d.spectral_pool_scheme == "proj":
			O = applySpectralPooling(O, d)

		if   d.model == "real":
			O = Conv2D(nb_fmaps1, filter_size, name=conv_name_base+'2a', strides=(2, 2), **convArgs)(O)
		elif d.model == "real_dws":
			O = SeparableConv2D(nb_fmaps1, filter_size, name=conv_name_base + '2a', strides=(2, 2), **convArgs)(O)
		elif (d.model == "real_group") or (d.model=="real_group_pwc_full") or (d.model=="real_group_pwc_group"):
			O_g0 = Lambda(lambda O: O[:,:(O.shape[1]//2),:,:])(O)
			O_g1 = Lambda(lambda O: O[:,(O.shape[1]//2):,:,:])(O)
			O_g0 = Conv2D(nb_fmaps1 // 2, filter_size, name=conv_name_base + '2a_g0', strides=(2, 2), **convArgs)(O_g0)
			O_g1 = Conv2D(nb_fmaps1 // 2, filter_size, name=conv_name_base + '2a_g1', strides=(2, 2), **convArgs)(O_g1)
			O_g00 = Lambda(lambda O_g0: O_g0[:,:(O_g0.shape[1]//2),:,:])(O_g0)
			O_g01 = Lambda(lambda O_g0: O_g0[:,(O_g0.shape[1]//2):,:,:])(O_g0)
			O_g10 = Lambda(lambda O_g1: O_g1[:,:(O_g1.shape[1]//2),:,:])(O_g1)
			O_g11 = Lambda(lambda O_g1: O_g1[:,(O_g1.shape[1]//2):,:,:])(O_g1)
			O = Concatenate(axis=1)([O_g00, O_g11, O_g01, O_g10])
			if d.model == "real_group_pwc_full":
				O = Conv2D(nb_fmaps1, (1, 1), name=conv_name_base + '2a_pwc', **convArgs)(O)
			elif d.model == "real_group_pwc_group":
				O_g0 = Lambda(lambda O: O[:, :(O.shape[1] // 2), :, :])(O)
				O_g1 = Lambda(lambda O: O[:, (O.shape[1] // 2):, :, :])(O)
				O_g0 = Conv2D(int(O.shape[1] // 2), (1, 1), name=conv_name_base + '2a_pwc_g0', **convArgs_real)(O_g0)
				O_g1 = Conv2D(int(O.shape[1] // 2), (1, 1), name=conv_name_base + '2a_pwc_g1', **convArgs_real)(O_g1)
				O = Concatenate(axis=1)([O_g0, O_g1])
		elif d.model == "complex":
			O = ComplexConv2D(nb_fmaps1, filter_size, name=conv_name_base+'2a', strides=(2, 2), **convArgs)(O)
		elif (d.model == "complex_concat") or (d.model == "complex_concat_pwc_group"):
			O = ComplexConvConcat2D(nb_fmaps1 // 2, filter_size, name=conv_name_base+'2a', strides=(2, 2), **convArgs)(O)
			if d.model == "complex_concat_pwc_group":
				O_g0 = Lambda(lambda O: O[:, :(O.shape[1] // 2), :, :])(O)
				O_g1 = Lambda(lambda O: O[:, (O.shape[1] // 2):, :, :])(O)
				O_g0 = Conv2D(int(O.shape[1] // 2), (1, 1), name=conv_name_base + '2a_pwc_g0', **convArgs_real)(O_g0)
				O_g1 = Conv2D(int(O.shape[1] // 2), (1, 1), name=conv_name_base + '2a_pwc_g1', **convArgs_real)(O_g1)
				O = Concatenate(axis=1)([O_g0, O_g1])
		else:
			print("Error: unknown model type")
			exit(-1)


	if   d.model == "real":
		O = BatchNormalization(name=bn_name_base+'_2b', **bnArgs)(O)
		O = Activation(activation)(O)
		O = Conv2D(nb_fmaps2, filter_size, name=conv_name_base+'2b', **convArgs)(O)
	elif d.model == "real_dws":
		O = BatchNormalization(name=bn_name_base + '_2b', **bnArgs)(O)
		O = Activation(activation)(O)
		O = SeparableConv2D(nb_fmaps2, filter_size, name=conv_name_base + '2b', **convArgs)(O)
	elif (d.model == "real_group") or (d.model == "real_group_pwc_full") or (d.model == "real_group_pwc_group"):
		O = BatchNormalization(name=bn_name_base + '_2b', **bnArgs)(O)
		O = Activation(activation)(O)
		O_g0 = Lambda(lambda O: O[:, :(O.shape[1] // 2), :, :])(O)
		O_g1 = Lambda(lambda O: O[:, (O.shape[1] // 2):, :, :])(O)
		O_g0 = Conv2D(nb_fmaps2 // 2, filter_size, name=conv_name_base + '2b_g0', **convArgs)(O_g0)
		O_g1 = Conv2D(nb_fmaps2 // 2, filter_size, name=conv_name_base + '2b_g1', **convArgs)(O_g1)
		O_g00 = Lambda(lambda O_g0: O_g0[:, :(O_g0.shape[1] // 2), :, :])(O_g0)
		O_g01 = Lambda(lambda O_g0: O_g0[:, (O_g0.shape[1] // 2):, :, :])(O_g0)
		O_g10 = Lambda(lambda O_g1: O_g1[:, :(O_g1.shape[1] // 2), :, :])(O_g1)
		O_g11 = Lambda(lambda O_g1: O_g1[:, (O_g1.shape[1] // 2):, :, :])(O_g1)
		O = Concatenate(axis=1)([O_g00, O_g11, O_g01, O_g10])
		if d.model == "real_group_pwc_full":
			O = Conv2D(nb_fmaps2, (1, 1), name=conv_name_base + '2b_pwc', **convArgs)(O)
		elif d.model == "real_group_pwc_group":
			O_g0 = Lambda(lambda O: O[:, :(O.shape[1] // 2), :, :])(O)
			O_g1 = Lambda(lambda O: O[:, (O.shape[1] // 2):, :, :])(O)
			O_g0 = Conv2D(int(O.shape[1] // 2), (1, 1), name=conv_name_base + '2b_pwc_g0', **convArgs_real)(O_g0)
			O_g1 = Conv2D(int(O.shape[1] // 2), (1, 1), name=conv_name_base + '2b_pwc_g1', **convArgs_real)(O_g1)
			O = Concatenate(axis=1)([O_g0, O_g1])
	elif d.model == "complex":
		O = ComplexBN(name=bn_name_base+'_2b', **bnArgs)(O)
		if d.aact == "complex_joint_relu":
			O = Lambda(ComplexJointReLU)(O)
		else:
			O = Activation(activation)(O)
		O = ComplexConv2D(nb_fmaps2, filter_size, name=conv_name_base+'2b', **convArgs)(O)
	elif (d.model == "complex_concat") or (d.model=="complex_concat_pwc_group"):
		O = ComplexBN(name=bn_name_base+'_2b', **bnArgs)(O)
		if d.aact == "complex_joint_relu":
			O = Lambda(ComplexJointReLU)(O)
		else:
			O = Activation(activation)(O)
		O = ComplexConvConcat2D(nb_fmaps2 // 2, filter_size, name=conv_name_base+'2b', **convArgs)(O)
		if d.model == "complex_concat_pwc_group":
			O_g0 = Lambda(lambda O: O[:, :(O.shape[1] // 2), :, :])(O)
			O_g1 = Lambda(lambda O: O[:, (O.shape[1] // 2):, :, :])(O)
			O_g0 = Conv2D(int(O.shape[1] // 2), (1, 1), name=conv_name_base + '2b_pwc_g0', **convArgs_real)(O_g0)
			O_g1 = Conv2D(int(O.shape[1] // 2), (1, 1), name=conv_name_base + '2b_pwc_g1', **convArgs_real)(O_g1)
			O = Concatenate(axis=1)([O_g0, O_g1])

	else:
		print("Error: unknown model type")
		exit(-1)

	if   shortcut == 'regular':
		O = Add()([O, I])
	elif shortcut == 'projection':
		if d.spectral_pool_scheme == "proj":
			I = applySpectralPooling(I, d)
		if   d.model == "real":
			X = Conv2D(nb_fmaps2, (1, 1),
			           name    = conv_name_base+'1',
			           strides = (2, 2) if d.spectral_pool_scheme != "nodownsample" else
			                     (1, 1),
			           **convArgs)(I)
			O      = Concatenate(channel_axis)([X, O])
		elif d.model == "real_dws":
			X = SeparableConv2D(nb_fmaps2, (1, 1),
					   name=conv_name_base + '1',
					   strides=(2, 2) if d.spectral_pool_scheme != "nodownsample" else
					   (1, 1),
					   **convArgs)(I)
			O = Concatenate(channel_axis)([X, O])
		elif (d.model == "real_group") or (d.model=="real_group_pwc_full") or (d.model=="real_group_pwc_group"):
			I_g0 = Lambda(lambda I: I[:, :(I.shape[1] // 2), :, :])(I)
			I_g1 = Lambda(lambda I: I[:, (I.shape[1] // 2):, :, :])(I)
			X_g0 = Conv2D(nb_fmaps2 // 2, (1, 1),
					   name=conv_name_base + '1_g0',
					   strides=(2, 2) if d.spectral_pool_scheme != "nodownsample" else
					   (1, 1),
					   **convArgs)(I_g0)
			X_g1 = Conv2D(nb_fmaps2 // 2, (1, 1),
					   name=conv_name_base + '1_g1',
					   strides=(2, 2) if d.spectral_pool_scheme != "nodownsample" else
					   (1, 1),
					   **convArgs)(I_g1)
			X_g00 = Lambda(lambda X_g0: X_g0[:, :(X_g0.shape[1] // 2), :, :])(X_g0)
			X_g01 = Lambda(lambda X_g0: X_g0[:, (X_g0.shape[1] // 2):, :, :])(X_g0)
			X_g10 = Lambda(lambda X_g1: X_g1[:, :(X_g1.shape[1] // 2), :, :])(X_g1)
			X_g11 = Lambda(lambda X_g1: X_g1[:, (X_g1.shape[1] // 2):, :, :])(X_g1)
			X = Concatenate(axis=1)([X_g00, X_g11, X_g01, X_g10])
			if d.model == "real_group_pwc_full":
				X = Conv2D(nb_fmaps2, (1, 1), name=conv_name_base + '1_pwc', **convArgs)(X)
			elif d.model == "real_group_pwc_group":
				X_g0 = Lambda(lambda X: X[:, :(X.shape[1] // 2), :, :])(X)
				X_g1 = Lambda(lambda X: X[:, (X.shape[1] // 2):, :, :])(X)
				X_g0 = Conv2D(int(X.shape[1] // 2), (1, 1), name=conv_name_base + '1_pwc_g0', **convArgs_real)(X_g0)
				X_g1 = Conv2D(int(X.shape[1] // 2), (1, 1), name=conv_name_base + '1_pwc_g1', **convArgs_real)(X_g1)
				X = Concatenate(axis=1)([X_g0, X_g1])
			O = Concatenate(channel_axis)([X, O])
		elif d.model == "complex":
			X = ComplexConv2D(nb_fmaps2, (1, 1),
			                  name    = conv_name_base+'1',
			                  strides = (2, 2) if d.spectral_pool_scheme != "nodownsample" else
			                            (1, 1),
			                  **convArgs)(I)
			O_real = Concatenate(channel_axis)([GetReal()(X), GetReal()(O)])
			O_imag = Concatenate(channel_axis)([GetImag()(X), GetImag()(O)])
			O = Concatenate(1)([O_real, O_imag])
		elif (d.model == "complex_concat") or (d.model == "complex_concat_pwc_group"):
			X = ComplexConvConcat2D(nb_fmaps2 // 2, (1, 1),
							  name=conv_name_base + '1',
							  strides=(2, 2) if d.spectral_pool_scheme != "nodownsample" else
							  (1, 1),
							  **convArgs)(I)
			if d.model == "complex_concat_pwc_group":
				X_g0 = Lambda(lambda X: X[:, :(X.shape[1] // 2), :, :])(X)
				X_g1 = Lambda(lambda X: X[:, (X.shape[1] // 2):, :, :])(X)
				X_g0 = Conv2D(int(X.shape[1] // 2), (1, 1), name=conv_name_base + '1_pwc_g0', **convArgs_real)(X_g0)
				X_g1 = Conv2D(int(X.shape[1] // 2), (1, 1), name=conv_name_base + '1_pwc_g1', **convArgs_real)(X_g1)
				X = Concatenate(axis=1)([X_g0, X_g1])
			O_real = Concatenate(channel_axis)([GetReal()(X), GetReal()(O)])
			O_imag = Concatenate(channel_axis)([GetImag()(X), GetImag()(O)])
			O      = Concatenate(      1     )([O_real,     O_imag])
		else:
			print("Error: unknown model type")
			exit(-1)

	return O

def applySpectralPooling(x, d):
	"""Perform spectral pooling on input."""

	if d.spectral_pool_gamma > 0 and d.spectral_pool_scheme != "none":
		x = FFT2 ()(x)
		x = SpectralPooling2D(gamma=(d.spectral_pool_gamma,
		                             d.spectral_pool_gamma))(x)
		x = IFFT2()(x)
	return x

#
# Get ResNet Model
#

def getResnetModel(d):
	n             = d.num_blocks
	sf            = d.start_filter
	dataset       = d.dataset
	activation    = d.act
	advanced_act  = d.aact
	drop_prob     = d.dropout
	if "mnist" in dataset:
		inputShape    = (1, 28, 28) if K.image_dim_ordering() == "th" else (28, 28, 1)
	else:
		inputShape    = (3, 32, 32) if K.image_dim_ordering() == "th" else (32, 32, 3)
	channelAxis   = 1 if K.image_data_format() == 'channels_first' else -1
	filsize       = (3, 3)
	convArgs      = {
		"padding":                  "same",
		"use_bias":                 False,
		"kernel_regularizer":       l2(0.0001),
	}
	bnArgs        = {
		"axis":                     channelAxis,
		"momentum":                 0.9,
		"epsilon":                  1e-04
	}

	import copy

	convArgs_real=copy.deepcopy(convArgs)

	# if   d.model == "real":
	if "real" in d.model:
		sf *= 2
		convArgs.update({"kernel_initializer": Orthogonal(float(np.sqrt(2)))})
		convArgs_real = convArgs
	# elif d.model == "complex":
	elif "complex" in d.model:
		convArgs.update({"spectral_parametrization": d.spectral_param,
						 "kernel_initializer": d.comp_init})


	#
	# Input Layer
	#

	I = Input(shape=inputShape)

	#
	# Stage 1
	#

	O = learnConcatRealImagBlock(I, (1, 1), (3, 3), 0, '0', convArgs, bnArgs, d)
	O = Concatenate(channelAxis)([I, O])
	if d.model == "real":
		O = Conv2D(sf, filsize, name='conv1', **convArgs)(O)
		O = BatchNormalization(name="bn_conv1_2a", **bnArgs)(O)
	elif d.model == "real_dws":
		O = SeparableConv2D(sf, filsize, name='conv1', **convArgs)(O)
		O = BatchNormalization(name="bn_conv1_2a", **bnArgs)(O)
	elif (d.model == "real_group") or (d.model == "real_group_pwc_full") or (d.model == "real_group_pwc_group"):
		O_g0 = Lambda(lambda O: O[:, :(O.shape[1] // 2), :, :])(O)
		O_g1 = Lambda(lambda O: O[:, (O.shape[1] // 2):, :, :])(O)
		O_g0 = Conv2D(sf // 2, filsize, name='conv1_g0', **convArgs)(O_g0)
		O_g1 = Conv2D(sf // 2, filsize, name='conv1_g1', **convArgs)(O_g1)
		O_g00 = Lambda(lambda O_g0: O_g0[:, :(O_g0.shape[1] // 2), :, :])(O_g0)
		O_g01 = Lambda(lambda O_g0: O_g0[:, (O_g0.shape[1] // 2):, :, :])(O_g0)
		O_g10 = Lambda(lambda O_g1: O_g1[:, :(O_g1.shape[1] // 2), :, :])(O_g1)
		O_g11 = Lambda(lambda O_g1: O_g1[:, (O_g1.shape[1] // 2):, :, :])(O_g1)
		O = Concatenate(axis=1)([O_g00, O_g11, O_g01, O_g10])
		if d.model == "real_group_pwc_full":
			O = Conv2D(sf, (1, 1), name='conv1_pwc', **convArgs)(O)
		elif d.model == "real_group_pwc_group":
			O_g0 = Lambda(lambda O: O[:, :(O.shape[1] // 2), :, :])(O)
			O_g1 = Lambda(lambda O: O[:, (O.shape[1] // 2):, :, :])(O)
			O_g0 = Conv2D(int(O.shape[1] // 2), (1, 1), name='conv1_pwc_g0', **convArgs_real)(O_g0)
			O_g1 = Conv2D(int(O.shape[1] // 2), (1, 1), name='conv1_pwc_g1', **convArgs_real)(O_g1)
			O = Concatenate(axis=1)([O_g0, O_g1])
		O = BatchNormalization(name="bn_conv1_2a", **bnArgs)(O)
	elif d.model == "complex":
		O = ComplexConv2D(sf, filsize, name='conv1', **convArgs)(O)
		O = ComplexBN(name="bn_conv1_2a", **bnArgs)(O)
	elif (d.model == "complex_concat") or (d.model=="complex_concat_pwc_group"):
		O = ComplexConvConcat2D(sf // 2, filsize, name='conv1', **convArgs)(O)
		O = ComplexBN(name="bn_conv1_2a", **bnArgs)(O)
		if d.model == "complex_concat_pwc_group":
			O_g0 = Lambda(lambda O: O[:, :(O.shape[1] // 2), :, :])(O)
			O_g1 = Lambda(lambda O: O[:, (O.shape[1] // 2):, :, :])(O)
			O_g0 = Conv2D(int(O.shape[1] // 2), (1, 1), name='conv1_pwc_g0', **convArgs_real)(O_g0)
			O_g1 = Conv2D(int(O.shape[1] // 2), (1, 1), name='conv1_pwc_g1', **convArgs_real)(O_g1)
			O = Concatenate(axis=1)([O_g0, O_g1])
	else:
		print("Error: unknown model type")
		exit(-1)

	if d.aact=="complex_joint_relu":
		O = Lambda(ComplexJointReLU)(O)
	else:
		O = Activation(activation)(O)

	#
	# Stage 2
	#

	for i in xrange(n):
		O = getResidualBlock(O, filsize, [sf, sf], 2, str(i), 'regular', convArgs, convArgs_real, bnArgs, d)
		if i == n//2 and d.spectral_pool_scheme == "stagemiddle":
			O = applySpectralPooling(O, d)

	#
	# Stage 3
	#

	O = getResidualBlock(O, filsize, [sf, sf], 3, '0', 'projection', convArgs, convArgs_real, bnArgs, d)
	if d.spectral_pool_scheme == "nodownsample":
		O = applySpectralPooling(O, d)

	for i in xrange(n-1):
		O = getResidualBlock(O, filsize, [sf*2, sf*2], 3, str(i+1), 'regular', convArgs, convArgs_real, bnArgs, d)
		if i == n//2 and d.spectral_pool_scheme == "stagemiddle":
			O = applySpectralPooling(O, d)

	#
	# Stage 4
	#

	O = getResidualBlock(O, filsize, [sf*2, sf*2], 4, '0', 'projection', convArgs, convArgs_real, bnArgs, d)
	if d.spectral_pool_scheme == "nodownsample":
		O = applySpectralPooling(O, d)

	for i in xrange(n-1):
		O = getResidualBlock(O, filsize, [sf*4, sf*4], 4, str(i+1), 'regular', convArgs, convArgs_real, bnArgs, d)
		if i == n//2 and d.spectral_pool_scheme == "stagemiddle":
			O = applySpectralPooling(O, d)

	#
	# Pooling
	#

	if d.spectral_pool_scheme == "nodownsample":
		O = applySpectralPooling(O, d)
		if "mnist" in dataset:
			O = AveragePooling2D(pool_size=(28, 28))(O)
		else:
			O = AveragePooling2D(pool_size=(32, 32))(O)
	else:
		if "mnist" in dataset:
			O = AveragePooling2D(pool_size=(7, 7))(O)
		else:
			O = AveragePooling2D(pool_size=(8, 8))(O)

	#
	# Flatten
	#

	O = Flatten()(O)

	#
	# Dense
	#

	if   dataset == 'cifar10':
		O = Dense(10,  activation='softmax', kernel_regularizer=l2(0.0001))(O)
	elif dataset == 'cifar100':
		O = Dense(100, activation='softmax', kernel_regularizer=l2(0.0001))(O)
	elif dataset == 'svhn':
		O = Dense(10,  activation='softmax', kernel_regularizer=l2(0.0001))(O)
	elif dataset == 'mnist':
		O = Dense(10,  activation='softmax', kernel_regularizer=l2(0.0001))(O)
	elif dataset == 'fashion_mnist':
		O = Dense(10,  activation='softmax', kernel_regularizer=l2(0.0001))(O)
	else:
		raise ValueError("Unknown dataset "+d.dataset)

	# Return the model
	return Model(I, O)

def getSimpleConvnetModel(d):
	n = d.num_blocks
	sf = d.start_filter
	dataset = d.dataset
	activation = d.act
	advanced_act = d.aact
	drop_prob = d.dropout
	if "mnist" in dataset:
		inputShape = (1, 28, 28) if K.image_dim_ordering() == "th" else (28, 28, 1)
	else:
		inputShape = (3, 32, 32) if K.image_dim_ordering() == "th" else (32, 32, 3)
	channelAxis = 1 if K.image_data_format() == 'channels_first' else -1
	filsize = (3, 3)
	convArgs = {
		"padding": "same",
		"use_bias": False,
		"kernel_regularizer": l2(0.0001),
	}
	bnArgs = {
		"axis": channelAxis,
		"momentum": 0.9,
		"epsilon": 1e-04
	}

	if d.model == "real":
		sf *= 2
		convArgs.update({"kernel_initializer": Orthogonal(float(np.sqrt(2)))})
	elif d.model == "complex":
		convArgs.update({"spectral_parametrization": d.spectral_param,
						 "kernel_initializer": d.comp_init})

	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Flatten
	from keras.layers import Conv2D, MaxPooling2D
	from keras.layers.normalization import BatchNormalization

	batch_size = 256
	num_classes = 10
	epochs = 50

	# input image dimensions
	img_rows, img_cols = 28, 28

	I = Input(shape=inputShape)
	O = Conv2D(32, kernel_size=(3, 3),
			   activation='relu',
			   kernel_initializer='he_normal',
			   input_shape=inputShape)(I)
	O = MaxPooling2D(pool_size=(2, 2))(O)
	O = Dropout(0.25)(O)
	O = Conv2D(64, (3, 3), activation='relu')(O)
	O = MaxPooling2D(pool_size=(2, 2))(O)
	O = Dropout(0.25)(O)
	O = Conv2D(128, (3, 3), activation='relu')(O)
	O = Dropout(0.4)(O)
	O = Flatten()(O)
	O = Dense(128, activation='relu')(O)
	O = Dropout(0.3)(O)
	O = Dense(num_classes, activation='softmax')(O)

	# Return the model
	return Model(I, O)

#
# Callbacks:
#

#
# Print a newline after each epoch, because Keras doesn't. Grumble.
#

class PrintNewlineAfterEpochCallback(Callback):
	def on_epoch_end(self, epoch, logs={}):
		sys.stdout.write("\n")

#
# LrDivisor. To use:
#
# lrDivisorCb     = LrDivisor(patience          = float(50000),
#                             division_cst      = 10.0,
#                             verbose           = 1,
#                             epoch_checkpoints = {75})
#

class LrDivisor(Callback):
	def __init__(self, patience=float(50000), division_cst=10.0, epsilon=1e-03, verbose=1, epoch_checkpoints={41, 61}):
		super(Callback, self).__init__()
		self.patience = patience
		self.checkpoints = epoch_checkpoints
		self.wait = 0
		self.previous_score = 0.
		self.division_cst = division_cst
		self.epsilon = epsilon
		self.verbose = verbose
		self.iterations = 0

	def on_batch_begin(self, batch, logs={}):
		self.iterations += 1

	def on_epoch_end(self, epoch, logs={}):
		current_score = logs.get('val_acc')
		divide = False
		if (epoch + 1) in self.checkpoints:
			divide = True
		elif (current_score >= self.previous_score - self.epsilon and current_score <= self.previous_score + self.epsilon):
			self.wait +=1
			if self.wait == self.patience:
				divide = True
		else:
			self.wait = 0
		if divide == True:
			K.set_value(self.model.optimizer.lr, self.model.optimizer.lr.get_value() / self.division_cst)
			self.wait = 0
			if self.verbose > 0:
				L.getLogger("train").info("Current learning rate is divided by"+str(self.division_cst) + ' and his values is equal to: ' + str(self.model.optimizer.lr.get_value()))
		self.previous_score = current_score

#
# Also evaluate performance on test set at each epoch end.
#

class TestErrorCallback(Callback):
	def __init__(self, test_data):
		self.test_data    = test_data
		self.loss_history = []
		self.acc_history  = []

	def on_epoch_end(self, epoch, logs={}):
		x, y = self.test_data

		L.getLogger("train").info("Epoch {:5d} Evaluating on test set...".format(epoch+1))
		test_loss, test_acc = self.model.evaluate(x, y, verbose=0)
		L.getLogger("train").info("                                      complete.")

		self.loss_history.append(test_loss)
		self.acc_history.append(test_acc)

		L.getLogger("train").info("Epoch {:5d} train_loss: {}, train_acc: {}, val_loss: {}, val_acc: {}, test_loss: {}, test_acc: {}".format(
		                          epoch+1,
		                          logs["loss"],     logs["acc"],
		                          logs["val_loss"], logs["val_acc"],
		                          test_loss,        test_acc))

#
# Keep a history of the validation performance.
#

class TrainValHistory(Callback):
	def __init__(self):
		self.train_loss = []
		self.train_acc  = []
		self.val_loss   = []
		self.val_acc    = []

	def on_epoch_end(self, epoch, logs={}):
		self.train_loss.append(logs.get('loss'))
		self.train_acc .append(logs.get('acc'))
		self.val_loss  .append(logs.get('val_loss'))
		self.val_acc   .append(logs.get('val_acc'))

#
# Save checkpoints.
#

class SaveLastModel(Callback):
	def __init__(self, workdir, period=10):
		self.workdir          = workdir
		self.chkptsdir        = os.path.join(self.workdir, "chkpts")
		if not os.path.isdir(self.chkptsdir):
			os.mkdir(self.chkptsdir)
		self.period_of_epochs = period
		self.linkFilename     = os.path.join(self.chkptsdir, "ModelChkpt.hdf5")

	def on_epoch_end(self, epoch, logs={}):
		if (epoch + 1) % self.period_of_epochs == 0:
			# Filenames
			baseHDF5Filename = "ModelChkpt{:06d}.hdf5".format(epoch+1)
			baseYAMLFilename = "ModelChkpt{:06d}.yaml".format(epoch+1)
			hdf5Filename     = os.path.join(self.chkptsdir, baseHDF5Filename)
			yamlFilename     = os.path.join(self.chkptsdir, baseYAMLFilename)

			# YAML
			yamlModel = self.model.to_yaml()
			with open(yamlFilename, "w") as yamlFile:
				yamlFile.write(yamlModel)

			# HDF5
			KM.save_model(self.model, hdf5Filename)
			with H.File(hdf5Filename, "r+") as f:
				f.require_dataset("initialEpoch", (), "uint64", True)[...] = int(epoch+1)
				f.flush()

			# Symlink to new HDF5 file, then atomically rename and replace.
			os.symlink(baseHDF5Filename, self.linkFilename+".rename")
			os.rename (self.linkFilename+".rename",
			           self.linkFilename)

			# Print
			L.getLogger("train").info("Saved checkpoint to {:s} at epoch {:5d}".format(hdf5Filename, epoch+1))

#
# Save record-best models.
#

class SaveBestModel(Callback):
	def __init__(self, workdir):
		self.workdir   = workdir
		self.bestdir   = os.path.join(self.workdir, "best")
		if not os.path.isdir(self.bestdir):
			os.mkdir(self.bestdir)
		self.best_acc  = 0
		self.best_loss = +np.inf

	def on_epoch_end(self, epoch, logs={}):
		val_loss = logs['loss']
		val_acc  = logs['acc']
		if val_acc > self.best_acc:
			self.best_acc  = val_acc
			self.best_loss = val_loss

			# Filenames
			hdf5Filename = os.path.join(self.bestdir, "Bestmodel_{:06d}_{:.4f}_{:.4f}.hdf5".format(epoch+1, val_acc, val_loss))
			yamlFilename = os.path.join(self.bestdir, "Bestmodel_{:06d}_{:.4f}_{:.4f}.yaml".format(epoch+1, val_acc, val_loss))

			# YAML
			yamlModel = self.model.to_yaml()
			with open(yamlFilename, "w") as yamlFile:
				yamlFile.write(yamlModel)

			# HDF5
			KM.save_model(self.model, hdf5Filename)
			with H.File(hdf5Filename, "r+") as f:
				f.require_dataset("initialEpoch", (), "uint64", True)[...] = int(epoch+1)
				f.flush()

			# Print
			L.getLogger("train").info("Saved best model to {:s} at epoch {:5d}".format(hdf5Filename, epoch+1))

#
# ResNet Learning-rate Schedules.
#

def schedule(epoch):
	if   epoch >=   0 and epoch <  10:
		lrate = 0.01
		if epoch == 0:
			L.getLogger("train").info("Current learning rate value is "+str(lrate))
	elif epoch >=  10 and epoch < 100:
		lrate = 0.1
		if epoch == 10:
			L.getLogger("train").info("Current learning rate value is "+str(lrate))
	elif epoch >= 100 and epoch < 120:
		lrate = 0.01
		if epoch == 100:
			L.getLogger("train").info("Current learning rate value is "+str(lrate))
	elif epoch >= 120 and epoch < 150:
		lrate = 0.001
		if epoch == 120:
			L.getLogger("train").info("Current learning rate value is "+str(lrate))
	elif epoch >= 150:
		lrate = 0.0001
		if epoch == 150:
			L.getLogger("train").info("Current learning rate value is "+str(lrate))
	return lrate

#
# Summarize environment variable.
#

def summarizeEnvvar(var):
	if var in os.environ: return var+"="+os.environ.get(var)
	else:                 return var+" unset"

#
# TRAINING PROCESS
#

def train(d):
	#
	# Log important data about how we were invoked.
	#

	L.getLogger("entry").info("INVOCATION:     "+" ".join(sys.argv))
	L.getLogger("entry").info("HOSTNAME:       "+socket.gethostname())
	L.getLogger("entry").info("PWD:            "+os.getcwd())

	summary  = "\n"
	#summary += "Environment:\n"
	#summary += summarizeEnvvar("THEANO_FLAGS")+"\n"
	#summary += "\n"
	summary += "Software Versions:\n"
	#summary += "Theano:                  "+T.__version__+"\n"
	summary += "TensorFlow:              "+T.__version__+"\n"
	summary += "Keras:                   "+keras.__version__+"\n"
	summary += "\n"
	summary += "Arguments:\n"
	summary += "Path to Datasets:        "+str(d.datadir)+"\n"
	summary += "Path to Workspace:       "+str(d.workdir)+"\n"
	summary += "Model:                   "+str(d.model)+"\n"
	summary += "Dataset:                 "+str(d.dataset)+"\n"
	summary += "Number of Epochs:        "+str(d.num_epochs)+"\n"
	summary += "Batch Size:              "+str(d.batch_size)+"\n"
	summary += "Number of Start Filters: "+str(d.start_filter)+"\n"
	summary += "Number of Blocks/Stage:  "+str(d.num_blocks)+"\n"
	summary += "Optimizer:               "+str(d.optimizer)+"\n"
	summary += "Learning Rate:           "+str(d.lr)+"\n"
	summary += "Learning Rate Decay:     "+str(d.decay)+"\n"
	summary += "Learning Rate Schedule:  "+str(d.schedule)+"\n"
	summary += "Clipping Norm:           "+str(d.clipnorm)+"\n"
	summary += "Clipping Value:          "+str(d.clipval)+"\n"
	summary += "Dropout Probability:     "+str(d.dropout)+"\n"
	if d.optimizer in ["adam"]:
		summary += "Beta 1:                  "+str(d.beta1)+"\n"
		summary += "Beta 2:                  "+str(d.beta2)+"\n"
	else:
		summary += "Momentum:                "+str(d.momentum)+"\n"
	L.getLogger("entry").info(summary[:-1])

	#
	# Load dataset
	#

	L.getLogger("entry").info("Loading dataset {:s} ...".format(d.dataset))
	np.random.seed(d.seed % 2**32)
	if   d.dataset == 'cifar10':
		(X_train, y_train), (X_test, y_test) = cifar10.load_data()
		nb_classes                           = 10
		n_train                              = 45000
	elif d.dataset == 'cifar100':
		(X_train, y_train), (X_test, y_test) = cifar100.load_data()
		nb_classes                           = 100
		n_train                              = 45000
	elif d.dataset == 'svhn':
		(X_train, y_train), (X_test, y_test) = svhn2.load_data()
		nb_classes                           = 10
		# Make classes 0 - 9 instead of 1 - 10
		y_train                              = y_train - 1
		y_test                               = y_test  - 1
		n_train                              = 65000
	elif d.dataset == 'mnist':
		(X_train, y_train), (X_test, y_test) = mnist.load_data()
		nb_classes                           = 10
		n_train                              = 55000
		X_train = np.expand_dims(X_train, axis=1)
		X_test  = np.expand_dims(X_test,  axis=1)
	elif d.dataset == 'fashion_mnist':
		(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
		nb_classes                           = 10
		n_train                              = 55000
		X_train = np.expand_dims(X_train, axis=1)
		X_test  = np.expand_dims(X_test,  axis=1)

	#
	# Compute and Shuffle Training/Validation/Test Split
	#

	shuf_inds  = np.arange(len(y_train))
	np.random.seed(0xDEADBEEF)
	np.random.shuffle(shuf_inds)
	train_inds = shuf_inds[:n_train]
	val_inds   = shuf_inds[n_train:]

	X_train    = X_train.astype('float32')/255.0
	X_test     = X_test .astype('float32')/255.0

	X_train_split = X_train[train_inds]
	X_val_split   = X_train[val_inds  ]
	y_train_split = y_train[train_inds]
	y_val_split   = y_train[val_inds  ]

	pixel_mean = np.mean(X_train_split, axis=0)

	X_train    = X_train_split.astype(np.float32) - pixel_mean
	X_val      = X_val_split  .astype(np.float32) - pixel_mean
	X_test     = X_test       .astype(np.float32) - pixel_mean

	Y_train    = to_categorical(y_train_split, nb_classes)
	Y_val      = to_categorical(y_val_split,   nb_classes)
	Y_test     = to_categorical(y_test,        nb_classes)

	if d.no_validation:
	    X_train = np.concatenate([X_train, X_val], axis=0)
	    Y_train = np.concatenate([Y_train, Y_val], axis=0)

	L.getLogger("entry").info("Training   set shape: "+str(X_train.shape))
	L.getLogger("entry").info("Validation set shape: "+str(X_val.shape))
	L.getLogger("entry").info("Test       set shape: "+str(X_test.shape))
	L.getLogger("entry").info("Loaded  dataset {:s}.".format(d.dataset))



	#
	# Initial Entry or Resume?
	#

	initialEpoch  = 0
	chkptFilename = os.path.join(d.workdir, "chkpts", "ModelChkpt.hdf5")
	isResuming    = os.path.isfile(chkptFilename)
	if isResuming:
		# Reload Model and Optimizer
		L.getLogger("entry").info("Reloading a model from "+chkptFilename+" ...")
		np.random.seed(d.seed % 2**32)
		model = KM.load_model(chkptFilename, custom_objects={
			"ComplexConv2D":             ComplexConv2D,
			"ComplexConvConcat2D":       ComplexConvConcat2D,
			"ComplexBatchNormalization": ComplexBN,
			"GetReal":                   GetReal,
			"GetImag":                   GetImag
		})
		L.getLogger("entry").info("... reloading complete.")

		with H.File(chkptFilename, "r") as f:
			initialEpoch = int(f["initialEpoch"][...])
		L.getLogger("entry").info("Training will restart at epoch {:5d}.".format(initialEpoch+1))
		L.getLogger("entry").info("Compilation Started.")
	else:
		# Model
		L.getLogger("entry").info("Creating new model from scratch.")
		np.random.seed(d.seed % 2**32)
		model = getResnetModel(d)
		# model = getSimpleConvnetModel(d)

		from keras.utils import plot_model
		plot_model(model, to_file=d.workdir+'/model_'+d.dataset+'_'+d.model+'.png')

		# Optimizer
		if   d.optimizer in ["sgd", "nag"]:
			opt = SGD    (lr       = d.lr,
			              momentum = d.momentum,
			              decay    = d.decay,
			              nesterov = (d.optimizer=="nag"),
			              clipnorm = d.clipnorm)
		elif d.optimizer == "rmsprop":
			opt = RMSProp(lr       = d.lr,
			              decay    = d.decay,
			              clipnorm = d.clipnorm)
		elif d.optimizer == "adam":
			opt = Adam   (lr       = d.lr,
			              beta_1   = d.beta1,
			              beta_2   = d.beta2,
			              decay    = d.decay,
			              clipnorm = d.clipnorm)
		else:
			raise ValueError("Unknown optimizer "+d.optimizer)

		# Compile the model with that optimizer.
		L.getLogger("entry").info("Compilation Started.")
		model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])
		model.summary()

	#
	# Precompile several backend functions
	#

	if d.summary:
		model.summary()
	L.getLogger("entry").info("# of Parameters:              {:10d}".format(model.count_params()))
	L.getLogger("entry").info("Compiling Train   Function...")
	t =- time.time()
	model._make_train_function()
	t += time.time()
	L.getLogger("entry").info("                              {:10.3f}s".format(t))
	L.getLogger("entry").info("Compiling Predict Function...")
	t =- time.time()
	model._make_predict_function()
	t += time.time()
	L.getLogger("entry").info("                              {:10.3f}s".format(t))
	L.getLogger("entry").info("Compiling Test    Function...")
	t =- time.time()
	model._make_test_function()
	t += time.time()
	L.getLogger("entry").info("                              {:10.3f}s".format(t))
	L.getLogger("entry").info("Compilation Ended.")

	#
	# Create Callbacks
	#

	newLineCb      = PrintNewlineAfterEpochCallback()
	lrSchedCb      = LearningRateScheduler(schedule)
	testErrCb      = TestErrorCallback((X_test, Y_test))
	saveLastCb     = SaveLastModel(d.workdir, period=10)
	saveBestCb     = SaveBestModel(d.workdir)
	trainValHistCb = TrainValHistory()


	callbacks  = []
	callbacks += [newLineCb]
	if d.schedule == "default":
		callbacks += [lrSchedCb]
	callbacks += [testErrCb]
	callbacks += [saveLastCb]
	callbacks += [saveBestCb]
	callbacks += [trainValHistCb]

	if d.tb == "on":
		if 'tensorflow' == K.backend():
			tensorBoardCb  = keras.callbacks.TensorBoard(
				log_dir=d.workdir+'/'+d.tbdir, histogram_freq=1, batch_size=d.batch_size,
				write_graph=True, write_grads=False, write_images=True,
				embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
			callbacks += [tensorBoardCb]

	#
	# Create training data generator
	#

	datagen         = ImageDataGenerator(height_shift_range = 0.125,
	                                     width_shift_range  = 0.125,
	                                     horizontal_flip    = True)

	#
	# Enter training loop.
	#

	L               .getLogger("entry").info("**********************************************")
	if isResuming: L.getLogger("entry").info("*** Reentering Training Loop @ Epoch {:5d} ***".format(initialEpoch+1))
	else:          L.getLogger("entry").info("***  Entering Training Loop  @ First Epoch ***")
	L               .getLogger("entry").info("**********************************************")

	model.fit_generator(generator       = datagen.flow(X_train, Y_train, batch_size=d.batch_size),
	                    steps_per_epoch = (len(X_train)+d.batch_size-1) // d.batch_size,
	                    epochs          = d.num_epochs,
	                    verbose         = 1,
	                    callbacks       = callbacks,
	                    validation_data = (X_val, Y_val),
	                    initial_epoch   = initialEpoch)

	#
	# Dump histories.
	#

	np.savetxt(os.path.join(d.workdir, 'test_loss.txt'),  np.asarray(testErrCb.loss_history))
	np.savetxt(os.path.join(d.workdir, 'test_acc.txt'),   np.asarray(testErrCb.acc_history))
	np.savetxt(os.path.join(d.workdir, 'train_loss.txt'), np.asarray(trainValHistCb.train_loss))
	np.savetxt(os.path.join(d.workdir, 'train_acc.txt'),  np.asarray(trainValHistCb.train_acc))
	np.savetxt(os.path.join(d.workdir, 'val_loss.txt'),   np.asarray(trainValHistCb.val_loss))
	np.savetxt(os.path.join(d.workdir, 'val_acc.txt'),    np.asarray(trainValHistCb.val_acc))

	# CIFAR-10:
	# - Baseline
	# - Baseline but with complex parametrization
	# - Baseline but with spectral pooling
