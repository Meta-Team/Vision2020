#########################################################################################################
#                   ece445lib version 0.2.4                                                             #
#                   Training and Utils functions and Classes for YoLov3 models.                         #
#                   Project Member: Jinghua Wang and Jintao Sun.                                        #
#                   Last modified on 2020-05-20.                                                        #
#########################################################################################################


#########################################################################################################
######################################### import libraries ##############################################
# ----------------------------------------------------------------------------------------------------- #
import time
import copy
import numpy as np
import mxnet as mx
import gluoncv.data.transforms.bbox as tbbox
import gluoncv.data.transforms.image as timage
from gluoncv.data.transforms import experimental
from mxnet import gluon, autograd, init, ndarray as nd
from mxnet.gluon import loss as gloss, nn
import gluoncv
from ..ece445_utils import save_train_record
# ----------------------------------------------------------------------------------------------------- #
######################################### import libraries ##############################################
#########################################################################################################


#########################################################################################################
####################################### YOLOv3 data transform ###########################################
# ----------------------------------------------------------------------------------------------------- #
class YOLOv3TrainTransform_DJIROCO(object):
    """YOLO training transform for DJIROCO dataset which includes tons of image augmentations.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    net : mxnet.gluon.HybridBlock, optional
        The yolo network.

        .. hint::

            If net is ``None``, the transformation will not generate training targets.
            Otherwise it will generate training targets to accelerate the training phase
            since we push some workload to CPU workers instead of GPUs.

    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    box_norm : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.
    x_y_normalize : Boolean, default True.
        Whether we normalize the (xmin, ymin, xmax, ymax) in labels to [0,1].
    label_filter : function, default None.
        label filter, for example you can use label_filter_armor_2_class.
    random_resize_interp : boolean, default False
        Whether we use random resize interpolation for input images, should be false when 
        loading test dataset. 
    --------------
    Return : stacked images, center_targets, scale_targets, gradient weights, objectness_targets, class_targets
        additionally, return padded ground truth bboxes, so there are 7 components returned by dataloader
    """
    def __init__(self, width, height, net=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), mixup=False, x_y_normalize=True, label_filter=None,
                 random_resize_interp=False, **kwargs):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std
        self._mixup = mixup
        self._target_generator = None
        self._x_y_normalize = x_y_normalize
        self._label_filter = label_filter
        self._random_resize_interp = random_resize_interp
        if net is None:
            return
        # in case network has reset_ctx to gpu
        self._num_classes = len(net.classes)
        self._fake_x = mx.nd.zeros((1, 3, height, width))
        net = copy.deepcopy(net)
        net.collect_params().reset_ctx(None)
        with autograd.train_mode():
            _, self._anchors, self._offsets, self._feat_maps, _, _, _, _ = net(self._fake_x)
        from gluoncv.model_zoo.yolo.yolo_target import YOLOV3PrefetchTargetGenerator
        self._target_generator = YOLOV3PrefetchTargetGenerator(
            num_class=len(net.classes), **kwargs)

    def __call__(self, img, bbox):
        """Apply transform to training image/label."""
        # filter the label(bbox) if needed
        if self._label_filter != None:
            bbox = self._label_filter(bbox)
        # # random color jittering, currently disabled
        # img = experimental.image.random_color_distort(src)
        # # random expansion with prob 0.5, currently disabled
        # if np.random.uniform(0, 1) > 0.5:
        #     img, expand = timage.random_expand(img, fill=[m * 255 for m in self._mean])
        #     bbox = tbbox.translate(label, x_offset=expand[0], y_offset=expand[1])
        # else:
        #     img, bbox = img, label
        # # random cropping, currently disabled
        # h, w, _ = img.shape
        # bbox, crop = experimental.bbox.random_crop_with_constraints(bbox, (w, h))
        # x0, y0, w, h = crop
        # img = mx.image.fixed_crop(img, x0, y0, w, h)
        # resize with random interpolation
        h, w, _ = img.shape
        if self._random_resize_interp:
            interp = np.random.randint(0, 5)
            img = timage.imresize(img, self._width, self._height, interp=interp)
        else:
            img = timage.imresize(img, self._width, self._height)
        if len(bbox) > 0:
            bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))
        # # random horizontal flip, currently disabled
        # h, w, _ = img.shape
        # img, flips = timage.random_flip(img, px=0.5)
        # bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])
        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        if self._target_generator is None:
            return img, bbox.astype(img.dtype)
        # generate training target so cpu workers can help reduce the workload on gpu
        if len(bbox) > 0:
            if self._x_y_normalize:
                if type(bbox) == nd.NDArray:
                    bbox[:,:4] = bbox[:,:4] * nd.array([1/self._width, 1/self._height, 1/self._width, 1/self._height])
                elif type(bbox) == np.ndarray: 
                    bbox[:,:4] = bbox[:,:4] * np.array([1/self._width, 1/self._height, 1/self._width, 1/self._height])
                else:
                    raise ValueError("bbox should be nd.array or numpy.ndarray, but is: "+str(type(bbox)))
            gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
            gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])
            if self._mixup:
                gt_mixratio = mx.nd.array(bbox[np.newaxis, :, -1:])
            else:
                gt_mixratio = None
            objectness, center_targets, scale_targets, weights, class_targets = self._target_generator(
                self._fake_x, self._feat_maps, self._anchors, self._offsets,
                gt_bboxes, gt_ids, gt_mixratio)
            return(img, objectness[0], center_targets[0], scale_targets[0], weights[0],
                        class_targets[0], gt_bboxes[0])
        else:
            total_num_anchors = 0
            for i in range(len(self._feat_maps)):
                # self._anchors[i].shape[-1] is always 2 (x and y), self._anchors[i].shape[-2] is number of anchors per level
                total_num_anchors += np.prod(self._feat_maps[i].shape) * self._anchors[i].shape[-2] 
            objectness = nd.zeros((total_num_anchors, 1))
            center_targets = nd.zeros((total_num_anchors, 2))
            scale_targets = nd.zeros((total_num_anchors, 2))
            weights = nd.zeros((total_num_anchors, 2))
            class_targets = -nd.ones((total_num_anchors,self._num_classes))
            # for the last returned variable, the ground truth bbox, we pad with -1 in default
            # -mx.nd.ones((1,4)) is an init for padding later
            return(img, objectness, center_targets, scale_targets, weights, class_targets, -mx.nd.ones((1,4)))

class YOLOv3ValTransform_DJIROCO(object):
    """YOLO validation transform for DJIROCO dataset.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    label_filter : function, default None.
        label filter, for example you can use label_filter_armor_2_class.
    """
    def __init__(self, width, height, label_filter=None,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std
        self._label_filter = label_filter

    def __call__(self, src, label):
        """Apply transform to validation image/label."""
        # resize
        h, w, _ = src.shape
        img = timage.imresize(src, self._width, self._height, interp=0) # gluoncv uses interp=9 in some cases.
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        if self._label_filter != None:
            label = self._label_filter(label)
        if len(label) > 0:
            bbox = tbbox.resize(label, in_size=(w, h), out_size=(self._width, self._height))
        else:
            bbox = -nd.ones((1,5)) # we do this for later padding with -1
        return img, bbox.astype(img.dtype)
# ----------------------------------------------------------------------------------------------------- #
####################################### YOLOv3 data transform ###########################################
#########################################################################################################


#########################################################################################################
################################### 2-class armor detection YOLOv3 ######################################
# ----------------------------------------------------------------------------------------------------- #
def get_2_class_YOLOv3(symbolic=False):
    '''
    symbolic : Boolean, default False.
        Whether we use symbolic computation with nn.Hybrid.
    ---------------------
    return : a custom designed YOLOv3 model.
    '''
    #TODO
    return
# ----------------------------------------------------------------------------------------------------- #
################################### 2-class armor detection YOLOv3 ######################################
#########################################################################################################


#########################################################################################################
####################################### YOLOv3 train function ###########################################
# ----------------------------------------------------------------------------------------------------- #
def train_YOLOv3(net, ctx, train_dataloader, num_epochs, num_batches, lr, batch_size, model_filename, warmup_epochs=None, lr_decay=0.1,
             lr_decay_epoch=[50,100], weight_decay=5e-4, momentum=0.9, use_AMP=False, print_info=True):
    from mxnet.contrib import amp # MxNet Automatic Mixed Precision (FP16 / FP32)
    from gluoncv.utils import LRScheduler, LRSequential

    if warmup_epochs == None:
        warmup_epochs = num_epochs - 1
    #num_batches = args.num_samples // args.batch_size
    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=lr,
                    nepochs=warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler('step', base_lr=lr, # lr_mode can be 'step', 'poly' or 'cosine'.
                    nepochs=num_epochs - warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_decay_epoch,
                    step_factor=lr_decay, power=2),
    ])
    trainer = gluon.Trainer(
            net.collect_params(), 'sgd',
            {'wd': weight_decay, 'momentum': momentum, 'lr_scheduler': lr_scheduler},
            kvstore='local', update_on_kvstore=(False if use_AMP else None))
    if use_AMP:
        amp.init()
        amp.init_trainer(trainer)
    train_record = []
    for epoch in range(num_epochs):
        epoch_record = {'sum_losses':[], 'obj_losses':[], 'center_losses':[], "scale_losses":[], "cls_losses":[], "time":[]}
        for i, batch in enumerate(train_dataloader):
            btic = time.time()
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            # objectness, center_targets, scale_targets, weights, class_targets
            fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0) for it in range(1, 6)]
            gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0)
            sum_losses = []
            obj_losses = []
            center_losses = []
            scale_losses = []
            cls_losses = []
            with autograd.record():
                for ix, x in enumerate(data):
                    # note that len(data) is 1, we do enumerate just for ix, the index issue
                    obj_loss, center_loss, scale_loss, cls_loss = net(x, gt_boxes[ix], *[ft[ix] for ft in fixed_targets])
                    sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                    obj_losses.append(obj_loss)
                    center_losses.append(center_loss)
                    scale_losses.append(scale_loss)
                    cls_losses.append(cls_loss)
                if use_AMP:
                    with amp.scale_loss(sum_losses, trainer) as scaled_loss:
                        autograd.backward(scaled_loss)
                else:
                    autograd.backward(sum_losses)
            trainer.step(batch_size)
            # update and print loss and other training info
            btime = time.time() - btic
            cur_sum_loss = nd.mean(sum_losses[-1])[0].asnumpy()[0]
            cur_obj_loss = nd.mean(obj_losses[-1])[0].asnumpy()[0]
            cur_center_loss = nd.mean(center_losses[-1]).asnumpy()[0]
            cur_scale_loss = nd.mean(scale_losses[-1]).asnumpy()[0]
            cur_cls_loss = nd.mean(cls_losses[-1]).asnumpy()[0]
            if print_info:
                print("Epoch "+str(epoch+1)+", batch "+str(i+1)
                        +", sum loss: {:.4f}".format(cur_sum_loss)
                        +", obj loss: {:.4f}".format(cur_obj_loss)
                        +", center loss: {:.4f}".format(cur_center_loss)
                        +", scale loss: {:.4f}".format(cur_cls_loss)
                        +", cls loss: {:.4f}".format(cur_cls_loss)
                        +", batch time: {:.4f}".format(btime)+" s          "
                        , end='\r')
            epoch_record['sum_losses'].append(cur_sum_loss)
            epoch_record['obj_losses'].append(cur_obj_loss)
            epoch_record['center_losses'].append(cur_center_loss)
            epoch_record['scale_losses'].append(cur_scale_loss)
            epoch_record['cls_losses'].append(cur_cls_loss)
            epoch_record['time'].append(btime)
        train_record.append(epoch_record)
    # save trained models and records if needed
    if model_filename != None:
        net.save_parameters(model_filename+".params")
        save_train_record(train_record, model_filename)
        if print_info:
            print("Model saved to file: "+model_filename+".params")
            print("Train record saved to file: "+model_filename+".record")
    return train_record
# ----------------------------------------------------------------------------------------------------- #
####################################### YOLOv3 train function ###########################################
#########################################################################################################
