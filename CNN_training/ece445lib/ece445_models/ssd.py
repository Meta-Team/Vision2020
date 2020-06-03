#########################################################################################################
#                   ece445lib version 0.2.3                                                             #
#                   Training and Utils functions and Classes for SSD models.                            #
#                   Project Member: Jinghua Wang and Jintao Sun.                                        #
#                   Last modified on 2020-05-15.                                                        #
#########################################################################################################


#########################################################################################################
######################################### import libraries ##############################################
# ----------------------------------------------------------------------------------------------------- #
import time
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd, init, ndarray as nd
from mxnet.gluon import loss as gloss, nn
import gluoncv
from ..ece445_utils import save_train_record
# ----------------------------------------------------------------------------------------------------- #
######################################### import libraries ##############################################
#########################################################################################################


#########################################################################################################
######################################## SSD train transform ############################################
# ----------------------------------------------------------------------------------------------------- #
class SSDTrainTransform_DJIROCO(object):
    """SSD training transform for DJIROCO dataset which includes tons of image augmentations.
    ----------
    width : int
        Image width.
    height : int
        Image height.
    anchors : mxnet.nd.NDArray, optional
        Anchors generated from SSD networks, the shape must be ``(1, N, 4)``.
        Since anchors are shared in the entire batch so it is ``1`` for the first dimension.
        ``N`` is the number of anchors for each image.
        If anchors is ``None``, the transformation will not generate training targets.
        Otherwise it will generate training targets to accelerate the training phase
        since we push some workload to CPU workers instead of GPUs.
    label_filter : function, default None.
        label filter, for example you can use label_filter_armor_2_class.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    x_y_normalize : Boolean, default True.
        Whether we normalize the (xmin, ymin, xmax, ymax) in labels to [0,1].
    box_norm : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.
    random_resize_interp : boolean, default False
        Whether we use random resize interpolation for input images, should be false when 
        loading test dataset. 
    """

    def __init__(self, width, height, anchors=None, label_filter=None,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), 
                 iou_thresh=0.5, x_y_normalize=True, box_norm=(0.1, 0.1, 0.2, 0.2),
                 random_resize_interp=False,
                 **kwargs):
        self._width = width
        self._height = height
        self._anchors = anchors
        self._mean = mean
        self._std = std
        self._x_y_normalize = x_y_normalize
        self._label_filter = label_filter
        self._random_resize_interp = random_resize_interp
        if anchors is None:
            return
        self._anchors_shape = anchors.shape
        # since we do not have predictions yet, so we ignore sampling here
        from gluoncv.model_zoo.ssd.target import SSDTargetGenerator
        self._target_generator = SSDTargetGenerator(
            iou_thresh=iou_thresh, stds=box_norm, negative_mining_ratio=-1, **kwargs)

    def __call__(self, img, bbox):
        """Apply transform to training image/label."""
        import gluoncv.data.transforms.bbox as tbbox
        import gluoncv.data.transforms.image as timage
        from gluoncv.data.transforms import experimental
        # filter the label(bbox) if needed
        if self._label_filter != None:
            bbox = self._label_filter(bbox)
        img_orig_size = img.shape
        # # random color jittering, currently disabled
        # img = experimental.image.random_color_distort(img)
        # # random expansion with prob 0.5, currently disabled
        # if np.random.uniform(0, 1) > 0.5:
        #     img, expand = timage.random_expand(img, fill=[m * 255 for m in self._mean])
        #     bbox = tbbox.translate(bbox, x_offset=expand[0], y_offset=expand[1])
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
        # if len(bbox) > 0:
        #     bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])
        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        if self._anchors is None:
            if len(bbox) > 0 and self._x_y_normalize:
                if type(bbox) == nd.NDArray:
                    bbox[:,:4] = bbox[:,:4] * nd.array([1/img_orig_size[1], 1/img_orig_size[0], 1/img_orig_size[1], 1/img_orig_size[0]])
                elif type(bbox) == np.ndarray:   
                    bbox[:,:4] = bbox[:,:4] * np.array([1/img_orig_size[1], 1/img_orig_size[0], 1/img_orig_size[1], 1/img_orig_size[0]])
                else:
                    raise ValueError("bbox should be nd.array or numpy.array, but is: "+str(type(bbox)))
            return img, bbox.astype(img.dtype)
        # generate training target so cpu workers can help reduce the workload on gpu
        if len(bbox) > 0:
            if self._x_y_normalize:
                if type(bbox) == nd.NDArray:
                    bbox[:,:4] = bbox[:,:4] * nd.array([1/img_orig_size[1], 1/img_orig_size[0], 1/img_orig_size[1], 1/img_orig_size[0]])
                elif type(bbox) == np.ndarray: 
                    bbox[:,:4] = bbox[:,:4] * np.array([1/img_orig_size[1], 1/img_orig_size[0], 1/img_orig_size[1], 1/img_orig_size[0]])
                else:
                    raise ValueError("bbox should be nd.array or numpy.ndarray, but is: "+str(type(bbox)))
            gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
            gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])
            cls_targets, box_targets, _ = self._target_generator(self._anchors, None, gt_bboxes, gt_ids)
        else:
            cls_targets = nd.zeros(self._anchors.shape[:-1])
            box_targets = nd.zeros(self._anchors.shape)
        return img, cls_targets[0], box_targets[0]
# ----------------------------------------------------------------------------------------------------- #
######################################## SSD train transform ############################################
#########################################################################################################        

#########################################################################################################
#################################### 2-class armor detection SSD ########################################
# ----------------------------------------------------------------------------------------------------- #
def get_SSD_sizes_and_ratios(config='original'):
    '''
    Set the SSD bbox configurations of sizes and ratios.
    ---------------------
    config : {'original', 'from_data"}, more configurations could be added later.
    ---------------------
    Return : (sizes, ratios)
    '''
    if config == 'original':
        sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
        ratios = [[1, 2, 0.5]] * 5
    elif config == 'from_data':
        # The optimal bbox aspect ratios x/y are: 0.33, 0.75, 1, 1.5, 2. 
        sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
        ratios = [[0.33, 1, 1.5]] * 5
    else:
        raise RuntimeError("config can only be 'original' 'from_data', but is: "+str(config))
    return sizes, ratios

class SSD(nn.Block):
    '''
    SSD model, non-hybrid version.
    ---------------------
    num_classes : number of classes to detect, int.
    sizes : size of anchor boxes, list.
    ratios : aspect ratios of anchor boxes, list.
    forward_mode : {'train', 'val', 'return_anchors'}
        'train': forward of SSD returns (cls_pred, bbox_pred)
            cls_pred shape = (batch_size, num_bboxes, num_classes)
            bbox_pred shape = (batch_size, num_bboxes, 4)
        'val' : forward of SSD returns (ids, scores, bboxes)
            ids: float type with int values. shape = (batch_size, post_nms)
                Predicted bbox class indices.
            scores : float, shape = (batch_size, post_nms)
                Prediction confidence scores.
            bboxes : float, shape = (batch_size, batch_size, post_nums, 4)
                Predicted bboxes in corner format, (x_min, y_min, x_max, y_max), normalized to [0,1].
                Note : bboxes out of the image range is also returned.
        'return_anchors' : forward of SSD returns (anchors, cls_pred, bbox_pred)
    stds : default (0.1, 0.1, 0.2, 0.2)
        bbox_norms, should be the same as the bbox_norms used in train trainsform.
    nums_thresh : default 0.45.
    nms_topk : default 400.
    post_nms : default 20.
    ---------------------
    return : an SSD model of class SSD inherited from nn.block.
    '''
    def __init__(self, num_classes, sizes, ratios, forward_mode='val', 
                stds=(0.1,0.1,0.2,0.2), nms_thresh=0.45, nms_topk=400, post_nms=20, **kwargs):
        super(SSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.num_anchors = len(sizes[0]) + len(ratios[0]) - 1
        self.sizes = sizes
        self.ratios = ratios
        if forward_mode not in ['train', 'val', 'return_anchors']:
            raise ValueError("forward_mode should be 'train', 'val', or 'return_anchors', but is: "+str(forward_mode))
        self.forward_mode = forward_mode
        self.stds, self.nms_thresh, self.nms_topk, self.post_nms = stds, nms_thresh, nms_topk, post_nms
        for i in range(5):
            # The assignment statement is self.blk_i = get_blk(i)
            setattr(self, 'blk_%d' % i, self.get_blk(i))
            setattr(self, 'cls_%d' % i, self.cls_predictor())
            setattr(self, 'bbox_%d' % i, self.bbox_predictor())
    
    def cls_predictor(self):
        return nn.Conv2D(self.num_anchors * (self.num_classes + 1), kernel_size=3, padding=1)

    def bbox_predictor(self):
        return nn.Conv2D(self.num_anchors * 4, kernel_size=3, padding=1)

    def blk_forward(self, X, blk, size, ratio, cls_predictor, bbox_predictor):
        Y = blk(X)
        if self.forward_mode != 'train':
            anchors = nd.contrib.MultiBoxPrior(Y, sizes=size, ratios=ratio)
            return (Y, anchors, cls_predictor(Y), bbox_predictor(Y))
        else:
            return (Y, cls_predictor(Y), bbox_predictor(Y))

    def flatten_pred(self, pred):
        return pred.transpose((0, 2, 3, 1)).flatten()

    def concat_preds(self, preds):
        return nd.concat(*[self.flatten_pred(p) for p in preds], dim=1)

    def down_sample_blk(self, num_channels):
        '''
        Down sample the input by a factor of 2 in width and height.
        Conv2D with 3x3 kernel, batchNorm, and MaxPooling.
        ---------------------
        num_channels : number of output channels, int.
        '''
        blk = nn.Sequential()
        for _ in range(2):
            blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                    nn.BatchNorm(in_channels=num_channels),
                    nn.Activation('relu'))
        blk.add(nn.MaxPool2D(2))
        return blk
    
    def get_blk(self, i):
        if i == 0:
            # in = 3x640x480, out = 32x40x30
            blk = nn.Sequential()
            for num_filters in [4, 8, 16, 32]:
                blk.add(self.down_sample_blk(num_filters))
        elif i == 1:
            # in = 32x40x30, out = 32x30x20
            blk = nn.Sequential()
            blk.add(
                nn.Conv2D(32, kernel_size=6),
                nn.BatchNorm(in_channels=32),
                nn.Activation('relu'),
                nn.Conv2D(32, kernel_size=6),
                nn.BatchNorm(in_channels=32),
                nn.Activation('relu')
            )
        elif i == 2:
            # in = 32x30x20, out = 32x14x9
            blk = nn.Sequential()
            blk.add(
                nn.Conv2D(32, kernel_size=3),
                nn.BatchNorm(in_channels=32),
                nn.Activation('relu'),
                nn.MaxPool2D(2)
            )
        elif i == 3:
            # in = 32x14x9, out = 32x10x5
            blk = nn.Sequential()
            blk.add(
                nn.Conv2D(32, kernel_size=5),
                nn.BatchNorm(in_channels=32),
                nn.Activation('relu'),
            )
        elif i == 4:
            # in = 32x10x5, out = 32x2x1
            blk = nn.Sequential()
            blk.add(
                nn.Conv2D(32, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=32),
                nn.Activation('relu'),
                nn.MaxPool2D(5)
            )
        else:
            raise RuntimeError("get_blk parameter index i should be 0,1,2,3,4, but is: "+str(i))
        return blk
    
    def SSD_pred_decode(self, cls_preds, bbox_preds, anchors, num_classes,
        stds=(0.1,0.1,0.2,0.2), nms_thresh=0.45, nms_topk=400, post_nms=20):
        '''
        Get SSD model inference predictions from model outputs of class predictions and bbox preditions,
        decode the SSD model inference immediate results to plot format.
        Parameters:
        -------------------------
        cls_preds : array-like, shape = (batch_size, num_bboxes, num_classes)
            one-hot encoded-like class predictions.
        bbox_preds : array-like, shape = (batch_size, num_bboxes, 4)
            prediction results of bboxes.
        anchors : array-like, anchors of center type. shape = (1, num_bboxes, 4)
        num_classes : int, number of positive classes.
        stds : tuple of float, default is (0.1, 0.1, 0.2, 0.2)
            Std values to be divided/multiplied to box encoded values.
        nms_thresh : float, default is 0.45.
            Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
        nms_topk : int, default is 400
            Apply NMS to top k detection results, use -1 to disable so that every Detection
            result is used in NMS.
        post_nms : int, default is 10
            Only return top `post_nms` detection results. You can use -1 to return all detections.
        -------------------------
        Return : (ids, scores, bboxes)
            ids: float type with int values. shape = (batch_size, post_nms)
                Predicted bbox class indices.
            scores : float, shape = (batch_size, post_nms)
                Prediction confidence scores.
            bboxes : float, shape = (batch_size, batch_size, post_nums, 4)
                Predicted bboxes in corner format, (x_min, y_min, x_max, y_max), normalized to [0,1].
                Note : bboxes out of image range is also returned.
        '''
        from gluoncv.nn.coder import MultiPerClassDecoder, NormalizedBoxCenterDecoder
        bbox_decoder = NormalizedBoxCenterDecoder(stds)
        cls_decoder = MultiPerClassDecoder(num_classes + 1, thresh=0.01)
        bboxes = bbox_decoder(bbox_preds, anchors)
        cls_ids, scores = cls_decoder(nd.softmax(cls_preds, axis=-1))
        results = []
        for i in range(num_classes):
            cls_id = cls_ids.slice_axis(axis=-1, begin=i, end=i+1)
            score = scores.slice_axis(axis=-1, begin=i, end=i+1)
            # per class results
            per_result = nd.concat(*[cls_id, score, bboxes], dim=-1)
            results.append(per_result)
        result = nd.concat(*results, dim=1)
        if nms_thresh > 0 and nms_thresh < 1:
            result = nd.contrib.box_nms(
                result, overlap_thresh=nms_thresh, topk=nms_topk, valid_thresh=0.01,
                id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=post_nms)
        # pylint: disable=no-member
        ids = nd.slice_axis(result, axis=2, begin=0, end=1).reshape(0,-1)
        scores = nd.slice_axis(result, axis=2, begin=1, end=2).reshape(0,-1)
        bboxes = nd.slice_axis(result, axis=2, begin=2, end=6)
        return ids, scores, bboxes
    
    def forward(self, X):
        # pylint: disable=no-member
        if self.forward_mode == 'return_anchors':
            anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
            for i in range(5):
                # getattr(self, 'blk_%d' % i) accesses self.blk_i
                X, anchors[i], cls_preds[i], bbox_preds[i] = self.blk_forward(
                        X, getattr(self, 'blk_%d' % i), self.sizes[i], self.ratios[i],
                           getattr(self, 'cls_%d' % i), getattr(self, 'bbox_%d' % i))
            # In the reshape function, 0 indicates that the batch size remains unchanged
            return (nd.concat(*anchors, dim=1),
                    self.concat_preds(cls_preds).reshape((0, -1, self.num_classes + 1)),
                    self.concat_preds(bbox_preds).reshape((0, -1, 4)))
        elif self.forward_mode == 'train':
            cls_preds, bbox_preds = [None] * 5, [None] * 5
            for i in range(5):
                # getattr(self, 'blk_%d' % i) accesses self.blk_i
                X, cls_preds[i], bbox_preds[i] = self.blk_forward(
                        X, getattr(self, 'blk_%d' % i), self.sizes[i], self.ratios[i],
                           getattr(self, 'cls_%d' % i), getattr(self, 'bbox_%d' % i))
            # In the reshape function, 0 indicates that the batch size remains unchanged
            return (self.concat_preds(cls_preds).reshape((0, -1, self.num_classes + 1)),
                    self.concat_preds(bbox_preds).reshape((0, -1, 4)))
        elif self.forward_mode == 'val':
            anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
            for i in range(5):
                # getattr(self, 'blk_%d' % i) accesses self.blk_i
                X, anchors[i], cls_preds[i], bbox_preds[i] = self.blk_forward(
                        X, getattr(self, 'blk_%d' % i), self.sizes[i], self.ratios[i],
                           getattr(self, 'cls_%d' % i), getattr(self, 'bbox_%d' % i))
            # In the reshape function, 0 indicates that the batch size remains unchanged
            return self.SSD_pred_decode(cls_preds=self.concat_preds(cls_preds).reshape((0, -1, self.num_classes + 1)), 
                                   bbox_preds=self.concat_preds(bbox_preds).reshape((0, -1, 4)),
                                   anchors=nd.concat(*anchors, dim=1), num_classes=self.num_classes,
                                   stds=self.stds, nms_thresh=self.nms_thresh, nms_topk=self.nms_topk, post_nms=self.post_nms)
        else:
            raise ValueError("self.forward_mode should be 'return_anchors', 'train' or 'val', but is: "+str(self.forward_mode))
# ----------------------------------------------------------------------------------------------------- #
#################################### 2-class armor detection SSD ########################################
#########################################################################################################


#########################################################################################################
######################################## SSD train function #############################################
# ----------------------------------------------------------------------------------------------------- #
def train_SSD(net, ctx, trainer, batch_size, dataloader, num_epochs, 
        validate_net=False, test_dataloader=None, model_filename=None, cls_bbox_ctx=mx.cpu(0), print_info=True):
    '''
    The train function for SSD model.
    ---------------------
    net : neural network model that is already initialized.
    ctx : the device context of main net forward operations.
    trainer : gluon.Trainer instance.
    batch_size : int.
    dataloader : gluon.data.DataLoader instance for loading batches of data.
    num_epochs : total number of epochs during the training process.
    validate_net : Boolean, default False
    test_dataloader : gluon.data.DataLoader, only used when validate_net is True.
    model_filename : string of filename to which we save the model parameters.
        model_filename is None by default, which means model won't be saved.
    cls_bbox_ctx : the device context of loss evaluation, default mx.cpu(0).
    print_info: whether to print the loss info during training process, default True.
    ---------------------
    Return : train_record, list of dict of list.

    '''
    train_record = []
    for epoch in range(num_epochs):
        mbox_loss = gluoncv.loss.SSDMultiBoxLoss()
        ce_metric = mx.metric.Loss('CrossEntropy')
        smoothl1_metric = mx.metric.Loss('SmoothL1')
        epoch_record = {'crossEntropy':[], 'smoothL1':[], 'time':[]}
        for i, data_iter in enumerate(dataloader):
            btic = time.time()
            image_iter, cls_iter, bbox_iter = data_iter
            X = image_iter.as_in_context(ctx)
            with autograd.record():
                cls_preds, box_preds = net(X)
                sum_loss, cls_loss, box_loss = mbox_loss(cls_preds.as_in_context(cls_bbox_ctx), 
                                                        box_preds.as_in_context(cls_bbox_ctx), 
                                                        cls_iter.as_in_context(cls_bbox_ctx),
                                                        bbox_iter.as_in_context(cls_bbox_ctx))
                autograd.backward(sum_loss[0].as_in_context(ctx))
            trainer.step(batch_size)
            # evaluate and update numeric losses
            ce_metric.update(0, [l * batch_size for l in cls_loss])
            smoothl1_metric.update(0, [l * batch_size for l in box_loss])
            _, loss1 = ce_metric.get()
            _, loss2 = smoothl1_metric.get()
            batch_time = time.time()-btic
            if print_info:
                print("Epoch "+str(epoch+1)+", batch "+str(i+1)
                        +", crossEntropy Loss: {:.4f}".format(loss1)
                        +", smoothL1 Loss: {:.4f}".format(loss2)
                        +", batch time: {:.4f}".format(batch_time)+"s"
                        , end='\r')
            epoch_record['crossEntropy'].append(loss1)
            epoch_record['smoothL1'].append(loss2)
            epoch_record['time'].append(batch_time)
        if validate_net:
            net.forward_mode = 'val'
            epoch_ious = get_ious_SSD(net, test_dataloader, ctx, one_batch=True)
            net.forward_mode = 'train'
            epoch_record['IOU'] = epoch_ious
        train_record.append(epoch_record)
        if print_info:
            if validate_net:
                print("Epoch "+str(epoch+1)
                    +", time: {:.4f}".format(np.sum(epoch_record['time']))+"s"
                    +", Avg crossEntropy Loss: {:.4f}".format(np.mean(epoch_record['crossEntropy']))
                    +", Avg smoothL1 Loss: {:.4f}".format(np.mean(epoch_record['smoothL1']))
                    +", IOU: [avg={:.4f}".format(np.mean(epoch_ious))
                    +", max={:.4f}".format(np.max(epoch_ious))+"]")
            else:
                print("Epoch "+str(epoch+1)
                    +", time: {:.4f}".format(np.sum(epoch_record['time']))+"s"
                    +", Avg crossEntropy Loss: {:.4f}".format(np.mean(epoch_record['crossEntropy']))
                    +", Avg smoothL1 Loss: {:.4f}".format(np.mean(epoch_record['smoothL1'])))
    # save trained models and records if needed
    if model_filename != None:
        net.save_parameters(model_filename+".params")
        save_train_record(train_record, model_filename)
        if print_info:
            print("Model saved to file: "+model_filename+".params")
            print("Train record saved to file: "+model_filename+".record")
    return train_record

# ----------------------------------------------------------------------------------------------------- #
######################################## SSD train function #############################################
#########################################################################################################


#########################################################################################################
########################################### SSD utilities ###############################################
# ----------------------------------------------------------------------------------------------------- #
def get_ious_SSD(net, test_dataloader, ctx, one_batch=True):
    '''
    Get custom designed metric for SSD : best match IOU per image.
    Parameters:
    ------------------
    net : nn.block or nn.hybridblock, should have been initialized.
    test_dataloader : gluon.data.DataLoader instance for loading batches of test data.
        Note: test_dataloader should use normalized bbox labels.
    ctx : device context of net.
    one_batch : Boolean, default True.
        Whether we use only one batch for testing, usually True in training phase to save time.
        Use one_batch = False to obtain predictions on all test data.
    ------------------
    Returns : list of iou values.
    '''
    iou_block = gluoncv.nn.bbox.BBoxBatchIOU()
    iou_block.initialize(ctx)
    ious = []
    for batch in test_dataloader:
        image_iter, label_iter = batch
        image_iter, label_iter = image_iter.as_in_context(ctx), label_iter.as_in_context(ctx)
        pred_cls, pred_score, pred_bbox = net(image_iter)
        for i in range(len(label_iter)):
            # pylint: disable=no-member
            bbox_iou_mat = iou_block(pred_bbox[i], label_iter[i][:,:4])
            ious.append(np.max(bbox_iou_mat * nd.broadcast_equal(nd.stack(pred_cls[i]).T, label_iter[i][:,4])).asnumpy()[0])
        if one_batch:
            break
    return ious
# ----------------------------------------------------------------------------------------------------- #
########################################### SSD utilities ###############################################
#########################################################################################################