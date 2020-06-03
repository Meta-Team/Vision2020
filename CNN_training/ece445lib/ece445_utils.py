#########################################################################################################
#                   ece445_utils version 0.2.3                                                          #
#                   General utility functions.                                                          #
#                   Project Member: Jinghua Wang and Jintao Sun.                                        #
#                   Last modified on 2020-05-19.                                                        #
#########################################################################################################


#########################################################################################################
######################################### import libraries ##############################################
# ----------------------------------------------------------------------------------------------------- #
import pickle
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd, init, ndarray as nd
from mxnet.gluon import loss as gloss, nn
import gluoncv
from matplotlib import pyplot as plt
# ----------------------------------------------------------------------------------------------------- #
######################################### import libraries ##############################################
#########################################################################################################


#########################################################################################################
######################################### accuracy metrics ##############################################
# ----------------------------------------------------------------------------------------------------- #
def ROCOAID_acc(y_pred, y_true):
    """
    Calculate the values of ROCOAID custom accuracy in a batch. 
    Used for human visual aid shooting mode.
    -------------------------
    Parameters:
    y_pred : ndarray of float with each term from 0.0 to 1.0, shape is (batch_size, 2, 3)
        a batch of predicted results in shape:
        (batch_size, color{'red', 'blue'}, info{appear_bool, relative_x, relative_y})
    y_true : ndarray of float with each term from 0.0 to 1.0, shape is (batch_size, 2, 3)
        a batch of true results in shape:
        (batch_size, color{'red', 'blue'}, info{appear_bool, relative_x, relative_y})
    -------------------------
    Returns: ndarray of float, shape is (batchsize,)
        array of float, with result scores from 0.0 to 1.0 for each image in a batch.
    """
    appear_comp = nd.equal(y_pred[:, :, 0], y_true[:, :, 0])
    rela_pos_comp = acc_aid(y_pred[:, :, 1:], y_true[:, :, 1:])
    return nd.mean(appear_comp * rela_pos_comp, axis=1)

def acc_aid(y_p, y_t):
    """
    Helper function for calculating relative position accuracy component for ROCOAID_acc.
    -------------------------
    Parameters:
    y_p : ndarray of float with each term from 0.0 to 1.0, shape is (batch_size, 2, 2)
        a batch of predicted results for relative positions in shape:
        (batch_size, color{'red', 'blue'}, info{relative_x, relative_y}))
    y_t : ndarray of float with each term from 0.0 to 1.0, shape is (batch_size, 2, 2)
        a batch of true results for relative positions in shape:
        (batch_size, color{'red', 'blue'}, info{relative_x, relative_y}))
    -------------------------
    Return : ndarray of float, shape is (batchsize, 2)
        relative position scores from 0.0 to 1.0 for each image in a batch.
        the last dim is color{'red', 'blue'}
    """
    return nd.mean(1 - nd.abs(y_p-y_t)**(1/2) * (1 - 4*nd.relu((y_p-0.5)*(y_t-0.5)))**2, axis=1)

def ROCOAUTO_acc(y_pred, y_true):
    """
    Calculate the values of ROCOAUTO custom accuracy in a batch.
    Used for fully automated shooting mode.
    -------------------------
    Parameters:
    y_pred: ndarray of float, shape is (batch_size, 2, 3)
        a batch of predicted results in shape:
        (batch_size, color{'red', 'blue'}, info{appear_bool, relative_x, relative_y})
    y_true: ndarray of float, shape is (batch_size, 2, 3)
        a batch of true results in shape:
        (batch_size, color{'red', 'blue'}, info{appear_bool, relative_x, relative_y})
    -------------------------
    Returns: ndarray of float, shape is (batchsize,)
        array of float, with result scores from 0.0 to 1.0 for each image in a batch.
    """
    appear_comp = nd.equal(y_pred[:, :, 0], y_true[:, :, 0])
    rela_pos_comp = acc_auto(y_pred[:, :, 1:], y_true[:, :, 1:])
    return nd.mean(appear_comp * rela_pos_comp, axis=1)

def acc_auto(y_p, y_t):
    """
    Helper function for calculating relative position accuracy component for ROCOAUTO_acc.
    -------------------------
    Parameters:
    y_p : ndarray of float with each term from 0.0 to 1.0, shape is (batch_size, 2, 2)
        a batch of predicted results for relative positions in shape:
        (batch_size, color{'red', 'blue'}, info{relative_x, relative_y}))
    y_t : ndarray of float with each term from 0.0 to 1.0, shape is (batch_size, 2, 2)
        a batch of true results for relative positions in shape:
        (batch_size, color{'red', 'blue'}, info{relative_x, relative_y}))
    -------------------------
    Returns: ndarray of float, shape is (batchsize, 2)
        Relative position scores from 0.0 to 1.0 for each image in a batch.
        the last dim is color{'red', 'blue'}
    """
    return nd.mean( (1 - nd.abs(y_p-y_t)*(1 - 4*nd.relu((y_p-0.5)*(y_t-0.5)))) * (1-nd.relu(nd.tanh(-100*(y_p-0.5)*(y_t-0.5)))), axis=1)
# ----------------------------------------------------------------------------------------------------- #
######################################### accuracy metrics ##############################################
#########################################################################################################


#########################################################################################################
###################################### conversion to ROCOACC ############################################
# ----------------------------------------------------------------------------------------------------- #
def bbox_2_class_to_ROCOACC(label, label_mode='relative', absolute_size=(640, 480)):
    """
    Convert 2-class bbox model inference format to ROCOACC format, can take batches (in tuples) or single
    input (in a numpy.ndarray or mxnet.ndarray.NDArray).
    -----------------------
    label : tuple of numpy.ndarrays, numpy.ndarray, or mxnet.ndarray.NDArray. In either case, the single
        array always has shape (N-bboxes-armor, 5)
        if label is tuple with length N, we process the all the batch with size N.
            tuple of numpy.ndarray, tuple with length N, array with shape (N-bboxes-armor, 5)
            The 5-term entry is of format:
                (xmin, ymin, xmax, ymax, class_idx) Where class_idx represents 0: red armor, 1: blue armor.
        if label is numpy.ndarray or mxnet.ndarray.NDArray, we process this label of one image.
    label_mode : String, {'relative', 'absolute'}, default is 'relative'.
        if label_mode is 'relative', then xmin, ymin, xmax, ymax in range from 0.0 to 1.0.
        if label_mode is 'absolute', then xmin, ymin, xmax, ymax are floats in absolute image size.
    absolute_size : tuple of 2 ints, the absolute size of image of input label. default is (640, 480).
        only neede when label_mode is 'absolute'.
    -----------------------
    Returns : A tuple of numpy.ndarrays, an numpy.ndarray or a mxnet.ndarray.NDArray, with dtype numpy.float32.
        for batch input, return a tuple with N ndarrays of shape (2, 3).
        for label of a single image as input, return an numpy.ndarray or mxnet.ndarray.NDArray with shape (2,3).
        for label of a single iamge as input, the return data type is the same as the input data type.
    """
    if label_mode not in ['relative', 'absolute']:
        raise ValueError("label_mode should be 'relative' or 'absolute', but is: "+str(type(label_mode))+".")
    if type(label) == tuple:
        retlist = []
        for batch_item in label:
            batch_item_red, batch_item_blue = batch_item[batch_item[:, -1]==0], batch_item[batch_item[:, -1]==1]
            ret_item = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]], dtype=np.float32)
            if len(batch_item_red) > 0:
                if label_mode == 'absolute':
                    batch_item_red = batch_item_red / np.array([absolute_size[0], absolute_size[1], absolute_size[0], absolute_size[1], 1])
                batch_item_red_center = np.vstack([(batch_item_red[:, 0] + batch_item_red[:, 2])/2, (batch_item_red[:, 1] + batch_item_red[:, 3])/2]).T
                ret_item[0,:] = np.hstack([1.0, batch_item_red_center[np.argmin(np.sum(np.abs(batch_item_red_center - np.array([0.5, 0.5])), axis=1))]])
            if len(batch_item_blue) > 0:
                if label_mode == 'absolute':
                    batch_item_blue = batch_item_blue / np.array([absolute_size[0], absolute_size[1], absolute_size[0], absolute_size[1], 1])
                batch_item_blue_center = np.vstack([(batch_item_blue[:, 0] + batch_item_blue[:, 2])/2, (batch_item_blue[:, 1] + batch_item_blue[:, 3])/2]).T
                ret_item[1,:] = np.hstack([1.0, batch_item_blue_center[np.argmin(np.sum(np.abs(batch_item_blue_center - np.array([0.5, 0.5])), axis=1))]])
            retlist.append(ret_item)
        return tuple(retlist)
    elif type(label) == np.ndarray:
        ret_label = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]], dtype=np.float32)
        label_red, label_blue = label[label[:, -1]==0], label[label[:, -1]==1]
        if len(label_red) > 0:
            if label_mode == 'absolute':
                label_red = label_red / np.array([absolute_size[0], absolute_size[1], absolute_size[0], absolute_size[1], 1])
            label_red_center = np.vstack([(label_red[:, 0] + label_red[:, 2])/2, (label_red[:, 1] + label_red[:, 3])/2]).T
            ret_label[0,:] = np.hstack([1.0, label_red_center[np.argmin(np.sum(np.abs(label_red_center - np.array([0.5, 0.5])), axis=1))]])
        if len(label_blue) > 0:
            if label_mode == 'absolute':
                label_blue = label_blue / np.array([absolute_size[0], absolute_size[1], absolute_size[0], absolute_size[1], 1])
            label_blue_center = np.vstack([(label_blue[:, 0] + label_blue[:, 2])/2, (label_blue[:, 1] + label_blue[:, 3])/2]).T
            ret_label[1,:] = np.hstack([1.0, label_blue_center[np.argmin(np.sum(np.abs(label_blue_center - np.array([0.5, 0.5])), axis=1))]])
        return ret_label
    elif type(label) == nd.NDArray:
        ret_label = nd.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]], dtype=np.float32, ctx=label.context)
        label_red = nd.where(label[:,-1]==0, label, -nd.ones(label.shape, ctx=label.context), axis=0)
        label_blue = nd.where(label[:,-1]==1, label, -nd.ones(label.shape, ctx=label.context), axis=0)
        # pylint: disable=no-member
        if nd.sum(label[:, -1] == 0) > 0:
            if label_mode == 'absolute':
                label_red = label_red / nd.array([absolute_size[0], absolute_size[1], absolute_size[0], absolute_size[1], 1], ctx=label.context)
            label_red_center = nd.stack((label_red[:, 0] + label_red[:, 2])/2, (label_red[:, 1] + label_red[:, 3])/2).T
            ret_label[0, 0] = 1.0
            ret_label[0,1:] = label_red_center[nd.argmin(nd.sum(nd.abs(label_red_center - 0.5*nd.ones((label.shape[0],2), ctx=label.context)), axis=1), axis=0)][0]
        if nd.sum(label[:, -1] == 1) > 0:
            if label_mode == 'absolute':
                label_blue = label_blue / nd.array([absolute_size[0], absolute_size[1], absolute_size[0], absolute_size[1], 1], ctx=label.context)
            label_blue_center = nd.stack((label_blue[:, 0] + label_blue[:, 2])/2, (label_blue[:, 1] + label_blue[:, 3])/2).T
            ret_label[1, 0] = 1.0
            ret_label[1,1:] = label_blue_center[nd.argmin(nd.sum(nd.abs(label_blue_center - 0.5*nd.ones((label.shape[0],2), ctx=label.context)), axis=1), axis=0)][0]
        return ret_label
    else:
        raise TypeError("label should be tuple, numpy.ndarray, but is: "+str(type(label))+".")

def bbox_2_class_to_ROCOACC_batch(input_batch, label_mode='relative', absolute_size=(640, 480)):
    """
    Convert a batch of data from bbox 2 class format to numpy.ndarray in ROCOACC format.
    This function iteratively calls the function bbox_2_class_to_ROCOACC to convert label format.
    ------------------------
    input_batch : tuple of numpy.ndarrays, mxnet.ndarray.NDArray, or numpy.ndarray, the whole input data batch.
    label_mode :  String, {'relative', 'absolute'}, default is 'relative'.
        parameter passed to function bbox_2_class_to_ROCOACC.
    absolute_size : tuple of 2 ints, the absolute size of image of input label. default is (640, 480).
        parameter passed to function bbox_2_class_to_ROCOACC.
    ------------------------
    Returns : An numpy.ndarray of batch labels in ROCOACC format.
    """
    if type(input_batch) == tuple:
        return bbox_2_class_to_ROCOACC(input_batch, label_mode=label_mode, absolute_size=absolute_size)
    elif type(input_batch) == np.ndarray:
        output_batch = np.zeros((len(input_batch), 2, 3))
        for i in range(len(input_batch)):
            output_batch[i] = bbox_2_class_to_ROCOACC(input_batch[i], label_mode=label_mode, absolute_size=absolute_size)
    elif type(input_batch) == nd.NDArray:
        output_batch = nd.zeros((len(input_batch), 2, 3), ctx=input_batch.context)
        for i in range(len(input_batch)):
            output_batch[i] = bbox_2_class_to_ROCOACC(input_batch[i], label_mode=label_mode, absolute_size=absolute_size)
    return output_batch
# ----------------------------------------------------------------------------------------------------- #
###################################### conversion to ROCOACC ############################################
#########################################################################################################


#########################################################################################################
###################### helper function for saving and reloading train records ###########################
# ----------------------------------------------------------------------------------------------------- #
def save_train_record(record, record_name):
    with open(record_name+'.record', 'wb') as f:
        pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)

def load_train_record(record_name):
    with open(record_name+".record", 'rb') as f:
        return pickle.load(f)
# ----------------------------------------------------------------------------------------------------- #
###################### helper function for saving and reloading train records ###########################
#########################################################################################################


#########################################################################################################
####################################### plotting and debugging ##########################################
# ----------------------------------------------------------------------------------------------------- #
def draw_symbol_net(filename_json, outfilename, in_data_shape, save_format='pdf', view=False):
    symnet = mx.symbol.load(filename_json)
    a = mx.viz.plot_network(symnet, shape={"data":in_data_shape}, save_format=save_format, node_attrs={"shape":'rect',"fixedsize":'false'})
    a.render(outfilename)
    if view:
        a.view()

def draw_net(net, in_data_shape, net_name="DNN_structure", save_format='pdf', view=False):
    X = nd.random.uniform(shape=in_data_shape)
    net.hybridize()
    net(X)
    net.export(net_name)
    draw_symbol_net(net_name+"-symbol.json",net_name+"structure", in_data_shape, save_format=save_format, view=view)

def plot_image_with_bbox(image, bbox, figsize=(8,6), dpi=150, colors=None,
                        class_names=('car', 'watcher', 'base', 'ignore', 'armor')):
    _, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if len(bbox) > 0:
        plt.show(gluoncv.utils.viz.plot_bbox(image.asnumpy(), bbox[:,:4], scores=None, ax=ax,
                        labels=bbox[:, 4:5], colors=colors, class_names=class_names))
    else:
        plt.show(gluoncv.utils.viz.plot_bbox(image.asnumpy(), np.array([]), scores=None, ax=ax,
                        colors=colors, class_names=class_names))

def plot_image_with_ROCOACC_label(image, ROCOACC_label, figsize=(8,6), dpi=150):
    #TODO
    return
# ----------------------------------------------------------------------------------------------------- #
####################################### plotting and debugging ##########################################
#########################################################################################################