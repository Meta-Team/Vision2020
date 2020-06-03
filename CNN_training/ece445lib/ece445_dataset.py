#########################################################################################################
#                   ece445_dataset version 0.2.4                                                        #
#                   DJIROCO dataset related image loading and preprocessing                             #
#                   Project Member: Jinghua Wang and Jintao Sun.                                        #
#                   Last modified on 2020-04-23.                                                        #
#########################################################################################################


#########################################################################################################
######################################### import libraries ##############################################
# ----------------------------------------------------------------------------------------------------- #
import os
import logging
import warnings
import time
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd, init, ndarray as nd
from mxnet.gluon import loss as gloss, nn
import gluoncv
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
# ----------------------------------------------------------------------------------------------------- #
######################################### import libraries ##############################################
#########################################################################################################


#########################################################################################################
############################### rename Chinese filenames to English #####################################
# ----------------------------------------------------------------------------------------------------- #
def chinese_to_pinyin(src_dir):
    '''
    Convert Chinese filenames of some train images and labels to English, used for DJI_ROCO 2019-12 release.
    Example Usage:
        Assume all .jpeg images are under ~/ml_datasets/DJI_ROCO/image/ directory and 
        all .xml labels are under ~/ml_datasets/DJI_ROCO/image_annotation/ directory, then call:

        chinese_to_pinyin(expanduser(os.path.join("~",'ml_datasets', 'DJI_ROCO', 'image')))
        chinese_to_pinyin(expanduser(os.path.join("~",'ml_datasets', 'DJI_ROCO', 'image_annotation')))

        to convert all images and labels to English filenames. 
    '''
    chinese = ["华南虎","逸仙狮","交龙","狼牙","高巨毅恒","中维动力","火线",
                "风暴","电创","追梦","速加网笃行","新日成","火锅","雷霆","领志科技"]
    pinyin = ["HuaNanHu", "YiXianShi", "JiaoLong","LangYa", "GaoJuYiHeng", "ZhongWeiDongLi", "HuoXian",
     "FengBao", "DianChuang", "ZhuiMeng", "SuJiaWangDuXing", "XinRiCheng", "HuoGuo", "LeiTing", "LingZhiKeJi"]
    all_files = os.listdir(src_dir)
    for src_filename in all_files:
        new_filename = src_filename
        for i in range(len(chinese)):
            if chinese[i] in src_filename:
                new_filename = new_filename.replace(chinese[i], pinyin[i])
        os.rename(os.path.join(src_dir, src_filename), os.path.join(src_dir, new_filename))

# ----------------------------------------------------------------------------------------------------- #
############################### rename Chinese filenames to English #####################################
#########################################################################################################


#########################################################################################################
############################### Load DJI ROCO object detection dataset ##################################
# ----------------------------------------------------------------------------------------------------- #
class DJIROCO(gluon.data.dataset.Dataset):
    """DJI ROCO robot detection Dataset.
    Parameters
    ----------
    root : str, default '~/ml_datasets/DJI_ROCO'
        Path to folder storing the dataset.
    splits : string, the filename of a txt file under the root dataset folder
        that saves all the names of images to be loaded. ('.txt' is not needed in this string)
        If splits is an empty string, then all images in root/image/folder
        will be used by default.
    splits_mode : {'include', 'exclude'}
        Have effects only when a splits file is provided and splits is not empty string.
        'include' means files listed in splits+'.txt' will be included the dataset.
        'exclude' means all the files will be loaded except those listed in splits+'.txt'.
    transform : callable, default None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for more information. 
        Example usage:
        new_image, new_label = transform(image, label)
        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    index_map : dict, default None
        In default, the 5 classes are mapped into indices from 0 to 4. We can
        customize it by providing a str to int dict specifying how to map class
        names to indices. Use by advanced users only, when you want to swap the orders
        of class labels.
    crop_poss : list, tuple, or array, with shape (dataset_len, 2), default is None.
        Custom cropbox top-left positions for all images in the loaded dataset, used for test 
        dataset mode.
    preload_label : bool, default True
        If True, then parse and load all labels into memory during
        initialization. It often accelerate speed but require more memory
        usage. Typical preloaded labels took tens of MB. You only need to disable it
        when your dataset is extremely large.
    ----------
    Class info:
        There are five classes: ('car', 'watcher', 'base', 'ignore', 'armor').
        The label shape is (N, 8), the 8 terms are:
            (xmin, ymin, xmax, ymax, class_idx, difficulty, armor_class, armor_color)
            ----------------------------------------------------------------------------------------------
            class_idx: float converted from int indices for the five classes above.
            ----------------------------------------------------------------------------------------------
            difficulty: degree of obscuration of the object.
            Only non-armor, non-ignore objects tagged with difficulty.
            0: Obscuration <= 20%; 1: 20%< Obscuration <= 50%; 2: Obscuration > 50%, -1: no tagged.
            ----------------------------------------------------------------------------------------------
            armor_class: ID number of armor plate. Cars:1 to 5, watcher:6, base:8, none: hard to identify.
            when loading dataset, armor_class of non-armor bboxes are mapped to -1,
            armor_class that is hard to identify is mapped to value 0,
            ----------------------------------------------------------------------------------------------
            armor_color: red, blue, or gray.
            when loading dataset, armor_color of non-armor bboxes are mapped to -1.
            for armor_color, red: 0, blue: 1, gray: 2.
    """

    # There are five attribute values:  car (robot), watcher (sentry), base, ignore, armor.
    # Reflected image of objects or robots outside the Battlefiled will be annotated as ignore.
    CLASSES = ('car', 'watcher', 'base', 'ignore', 'armor')

    def __init__(self, root=os.path.join('~','ml_datasets','DJI_ROCO'), splits="", splits_mode = 'include',
                 transform=None, index_map=None, crop_poss=None, preload_label=True, **kwargs):
        if not os.path.isdir(os.path.expanduser(root)):
            raise OSError(helper_msg = "{} is not a valid dir, cannot initialize dataset.".format(root))
        if splits_mode not in ['include', 'exclude']:
            raise ValueError("split_mode can only be 'include' or 'exclude', but is: "+str(splits_mode))
        super(DJIROCO, self).__init__(**kwargs)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._splits = splits
        self._splits_mode = splits_mode
        self._crop_poss = crop_poss
        self._items = self._load_items(splits, splits_mode)
        self._anno_path = os.path.join('{}', 'image_annotation', '{}.xml')
        self._image_path = os.path.join('{}', 'image', '{}.jpg')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        self._label_cache = self._preload_labels() if preload_label else None
        # sanity check to make sure we have valid number of input samples in crop_poss 
        if self._crop_poss != None and len(self._crop_poss) != len(self._items):
            raise RuntimeError("input cropbox top-left positions for all images 'crop_poss' has different length to actual loaded dataset!")

    def __str__(self):
        return self.__class__.__name__ + "."+self._splits

    @property
    def classes(self):
        """Category names."""
        try:
            self._validate_class_names(self.CLASSES)
        except AssertionError as e:
            raise RuntimeError("Class names must not contain {}".format(e))
        return type(self).CLASSES

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        if np.isscalar(idx):
            img_id = self._items[idx]
            img_path = self._image_path.format(*img_id)
            label = self._label_cache[idx] if self._label_cache else self._load_label(idx)
            img = mx.image.imread(img_path, 1)
            if self._transform is not None:
                if self._crop_poss is not None:
                    return self._transform(img, label, self._crop_poss[idx])
                else:
                    return self._transform(img, label)
            return img, label.copy()
        else:
            ret = []
            img_ids = self._items[idx]
            for i in range(len(img_ids)):
                img_path = self._image_path.format(*img_ids[i])
                label = self._label_cache[i] if self._label_cache else self._load_label(i)
                img = mx.image.imread(img_path, 1)  # 1 for colored output
                if self._transform is not None:
                    if self._crop_poss is not None:
                        img, label = self._transform(img, label, self._crop_poss[i])
                    else:
                        img, label = self._transform(img, label)
                else:
                    label = label.copy() # make sure label is a new copy
                ret.append((img,label))
            return ret

    @property
    def num_class(self):
        """Number of categories."""
        return len(self.classes)

    def _load_items(self, splits, splits_mode):
        """Load individual image indices from splits."""
        ids = []
        root = os.path.join(self._root)
        if splits != "":
            lf = os.path.join(root, splits+'.txt')
            if splits_mode == 'include':
                with open(lf, 'r') as f:
                    ids += [(root, line.strip()) for line in f.readlines()]
            else:
                with open(lf, 'r') as f:
                    ex_list = [line.strip() for line in f.readlines()]
                ids += [(root, file_str[:-4]) for file_str in os.listdir(os.path.join(root, "image")) if file_str[:-4] not in ex_list]
        else:
            ids += [(root, file_str[:-4]) for file_str in os.listdir(os.path.join(root, "image"))]
        return ids

    def _load_label(self, idx):
        """Parse xml file and return labels."""
        img_id = self._items[idx]
        anno_path = self._anno_path.format(*img_id)
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            # read difficulty, armor_class and armor_color
            obj_class_name_str = obj.find('name').text
            if obj_class_name_str == 'armor':
                difficulty = -1
                armor_class_str = obj.find('armor_class').text
                if armor_class_str == 'none':
                    armor_class = 0
                else:
                    armor_class = int(armor_class_str)
                armor_color_str = obj.find('armor_color').text
                if armor_color_str == 'red':
                    armor_color = 0
                elif armor_color_str == 'blue':
                    armor_color = 1
                elif armor_color_str == 'gray':
                    armor_color = 2
                else:
                    armor_color = -1
            else:
                if obj.find('name').text == 'ignore':
                    difficulty = -1
                else:
                    difficulty = int(obj.find('difficulty').text)
                armor_class = -1
                armor_color = -1
            cls_name = obj.find('name').text.strip().lower()    # class name
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]    # class id
            xml_box = obj.find('bndbox')
            # extract bbox positions, some notations use float(xml_value)-1 instead of float(xml_value).
            xmin = max(float(xml_box.find('xmin').text), 0)
            ymin = max(float(xml_box.find('ymin').text), 0)
            xmax = min(float(xml_box.find('xmax').text), width)
            ymax = min(float(xml_box.find('ymax').text), height)
            if xmin >= width or ymin >= height:
                continue
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
            label.append([xmin, ymin, xmax, ymax, cls_id, difficulty, armor_class, armor_color])
        return np.array(label, dtype=np.float32)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    def _validate_class_names(self, class_list):
        """Validate class names."""
        assert all(c.islower() for c in class_list), "uppercase characters"
        stripped = [c for c in class_list if c.strip() != c]
        if stripped:
            warnings.warn('white space removed for {}'.format(stripped))

    def _preload_labels(self):
        """Preload all labels into memory."""
        logging.debug("Preloading %s labels into memory...", str(self))
        return [self._load_label(idx) for idx in range(len(self))]

def batchify_fn_DJIROCO_SSD_val(data):
    batch_size = len(data)
    max_label_len = 0
    src_images = []
    src_labels = []
    for data_item in data:
        image, label = data_item
        src_images.append(image)
        src_labels.append(label)
        if len(label) > max_label_len:
            max_label_len = len(label)
    if max_label_len == 0:
        print("Warning in batchify_fn: a whole batch without any valid bboxes returned.")
    batch_images = nd.stack(*src_images)
    batch_labels = -nd.ones((batch_size, max_label_len, src_labels[0].shape[1]),dtype=np.float32)
    for i in range(batch_size):
        if len(src_labels[i]) > 0:
            batch_labels[i, :src_labels[i].shape[0], :] = src_labels[i][:,:]
    return batch_images, batch_labels

def get_batchify_fn_DJIROCO_SSD(mode):
    '''
    Get the batchify function on DJIROCO dataset for training or validation of SSD, collecting data into batches.
    ------------------------
    mode: {'train', 'val'}.
        The mode of batchify_fn returned.
    ------------------------
    Return : a batchify_fn function, either for train or for validation.
    Example usage:
    dataset = DJIROCO(rootpath)
        dataloader = gluon.data.DataLoader(dataset, batch_size=64, batchify_fn=get_batchify_fn_DJIROCO_SSD('train'))
        for dataset_iter in dataloader:
            image_batch, label_batch = dataset_iter
            train or test model in current batch...
    '''
    from gluoncv.data.batchify import Stack, Tuple, Pad
    if mode == 'train':
        return Tuple(Stack(), Stack(), Stack())
    elif mode == 'val':
        return batchify_fn_DJIROCO_SSD_val
    else:
        raise RuntimeError("mode must be 'train' or 'val', but is: "+str(mode))

def get_batchify_fn_DJIROCO_YOLOv3(mode):
    '''
    Get the batchify function on DJIROCO dataset for training or validation of YOLOv3, collecting data into batches.
    ------------------------
    mode: {'train', 'val'}.
        The mode of batchify_fn returned.
    ------------------------
    Return : a batchify_fn function, either for train or for validation.
    Example usage:
    dataset = DJIROCO(rootpath)
        dataloader = gluon.data.DataLoader(dataset, batch_size=64, batchify_fn=get_batchify_fn_DJIROCO_YOLOv3('train'))
        for dataset_iter in dataloader:
            image_batch, label_batch = dataset_iter
            train or test model in current batch...
    '''
    from gluoncv.data.batchify import Tuple, Stack, Pad
    if mode == 'train':
        return Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))
    elif mode == 'val':
        return Tuple(Stack(), Pad(axis=0, pad_val=-1))
    else:
        raise RuntimeError("mode must be 'train' or 'val', but is: "+str(mode))
# ----------------------------------------------------------------------------------------------------- #
############################### Load DJI ROCO object detection dataset ##################################
#########################################################################################################


#########################################################################################################
################################## image and label preprocessing ########################################
# ----------------------------------------------------------------------------------------------------- #
def crop_image_and_bbox_random(image, label):
    return crop_with_bbox_labels(image, label, crop_mode='random')

def crop_image_and_bbox_random_bbox(image, label):
    return crop_with_bbox_labels(image, label, crop_mode='random_with_bbox')

def crop_image_and_bbox_random_armor(image, label):
    return crop_with_bbox_labels(image, label, crop_mode='random_with_armor')

def crop_image_and_bbox_manual(image, label, top_left):
    return crop_with_bbox_labels(image, label, crop_mode='manual', top_left=top_left)

def crop_with_bbox_labels(image, label, crop_mode, top_left=(0,0), output_size=(640, 480), allow_outside_center=False):
    '''
    Function that crops one image into resolution 640x480, bboxes are cropped together with image.
    image : 
        array-like object, the image.
    label : 
        array-like object, the bbox labels of the image.
    crop_mode : {'random', 'random_with_bbox', 'random_with_armor', 'manual'}
        'random' : image is cropped randomly within the original image.
        'random_with_bbox' : randomly crop image such that it contains at least one full size bbox.
        'random_with_armor' : randomly crop image such that it contains at least one full size armor bbox.
        'manual' : crop image with input top-left absolute positions in pixels, require input parameter 'top_left'.
        Note: in some cases the whole original image has no armor bbox, even when you use 'random_with_armor',
        it is still possible that armor bbox will not show up in some samples.
    top_left : tuple of 2 ints, optional, default (0,0), only used when crop_mode='manual'.
        The top-left absolute position in pixels of output cropbox in the original image.
    output_size : tuple of 2 ints, optional, default (640, 480).
        The output cropbox size in (width, height).
    allow_outside_center : Boolean, optional, default False.
        Whether we allow bbox with its center out of range.
    -----------------------
    return : (processed_image, processed_label)
    '''
    if crop_mode not in ['random', 'random_with_bbox', 'random_with_armor', 'manual']:
        raise ValueError("crop_mode can only be 'random', 'random_with_bbox', 'random_with_armor', 'manual', but is: "+str(crop_mode))
    # for random modes, if there is no bbox(empty label), do random cropping and return empty label.
    if label.shape[0] == 0 and crop_mode != "manual":   
        crop_mode = 'random'
    # check different cases and set rand_xmin, rand_ymin for cropbox.
    if crop_mode == 'random':
        rand_xmin = np.random.randint(low=0, high=1920-output_size[0])
        rand_ymin = np.random.randint(low=0, high=1080-output_size[1])
    elif crop_mode == 'manual':
        rand_xmin, rand_ymin = top_left
        # sanity check on ranges of rand_xmin and rand_ymin
        if rand_xmin < 0 or rand_xmin > 1920-output_size[0]:
            raise ValueError("rand_xmin out of range! valid range is: [0,"+str(1920-output_size[0])+"), but rand_xmin is: "+str(rand_xmin)+".")
        if rand_ymin < 0 or rand_ymin > 1080-output_size[1]:
            raise ValueError("rand_ymin out of range! valid range is: [0,"+str(1080-output_size[1])+"), but rand_ymin is: "+str(rand_ymin)+".")
    else:
        if crop_mode == 'random_with_armor' and len(np.where(label[:,4] == 4)[0])>0:
            rand_bbox_idx = np.random.choice(np.where(label[:,4] == 4)[0])
            rand_bbox_idx = np.random.randint(low=0, high=label.shape[0])
        else:
            rand_bbox_idx = np.random.randint(low=0, high=label.shape[0])
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = label[rand_bbox_idx][:4]
        bbox_xmin, bbox_ymin = np.ceil(bbox_xmin),  np.ceil(bbox_ymin)
        bbox_xmax, bbox_ymax = np.floor(bbox_xmax), np.floor(bbox_ymax)
        rand_xmin = np.random.randint(low=min(max(0, bbox_xmax-output_size[0]-1), bbox_xmin), high=min(1920-output_size[0], bbox_xmin+1))
        rand_ymin = np.random.randint(low=min(max(0, bbox_ymax-output_size[1]-1), bbox_ymin), high=min(1080-output_size[1], bbox_ymin+1))
    # crop image and label using rand_xmin, rand_ymin.
    cropped_image = image[rand_ymin:(rand_ymin+output_size[1]), rand_xmin:(rand_xmin+output_size[0])].copy()
    if label.shape[0] == 0:
        cropped_label = label.copy()
    else:
        cropped_label = gluoncv.data.transforms.bbox.crop(label, crop_box=(rand_xmin,rand_ymin,output_size[0],output_size[1]),
                                                            allow_outside_center=allow_outside_center)
    return cropped_image, cropped_label

def label_filter_armor_2_class(label):
    '''
    Function that filters all the bbox labels and only return armor bboxes or an empty numpy.ndarray.
    ----------------------------
    Parameters:
    label : tuple or numpy.ndarray.
        The bbox labels we need to process and filter, can be label for single image, or label for a batch.
        Label for a single image is an numpy.ndarray with shape (N-bboxes, 8);
        label for a batch is a tuple of numpy.ndarrays, tuple with length N, array with shape (N-bboxes, 8).
        Should have dim (N, N-bboxes, 8) or (N-bboxes, 8), the last 8-term entry is of format:
            (xmin, ymin, xmax, ymax, class_idx, difficulty, armor_class, armor_color).
    -----------------------------
    Return : new copy of tuple or numpy.ndarray, modification of returned label does not change input label.
        Label for a simple image:
            numpy.ndarray with shape (N-bboxes-armor, 5)
        Label for a batch:
            tuple of numpy.ndarray, tuple with length N, array with shape (N-bboxes-armor, 5)
        The 5-term entry is of format:
            (xmin, ymin, xmax, ymax, class_idx)
            Where class_idx represents:
                0: red armor, 1: blue armor.
    '''
    if type(label) == tuple:
        retlist = []
        for batch_item in label:
            if len(batch_item) > 0:
                # the 'armor_color' term of non-armor bboxes is set to -1, and we remove those with -1 in 'armor_color'.
                # remove tall bboxes with width/height > 1.5
                batch_item[(batch_item[:, 3] - batch_item[:, 1]) > (batch_item[:, 2] - batch_item[:, 0]) * 1.2, -1] = -1
                # remove gray bboxes
                batch_item[batch_item[:, -1] == 2, -1] = -1
                retlist.append(np.array(batch_item[:,[0,1,2,3,-1]][np.where(batch_item[:,-1] != -1)[0]], dtype=np.float32))
            else:
                retlist.append(np.array([]).reshape(0,5))
        return tuple(retlist)
    elif type(label) == np.ndarray:
        if len(label) > 0:
            # the 'armor_color' term of non-armor bboxes is set to -1, and we remove those with -1 in 'armor_color'.
            # remove tall bboxes with width/height > 1.5
            label[(label[:, 3] - label[:, 1]) > (label[:, 2] - label[:, 0]) * 1.2, -1] = -1
            # remove gray bboxes
            label[label[:, -1] == 2, -1] = -1
            return np.array(label[:,[0,1,2,3,-1]][np.where(label[:,-1] != -1)[0]], dtype=np.float32)
        else:
            return np.array([]).reshape(0,5)
    else:
        raise TypeError("label should be tuple or numpy.ndarray, but is: "+str(type(label))+".")
# ----------------------------------------------------------------------------------------------------- #
################################## image and label preprocessing ########################################
#########################################################################################################


#########################################################################################################
########################################### test dataset ################################################
# ----------------------------------------------------------------------------------------------------- #
def get_test_DJIROCO(crop_poss_filepath=os.path.expanduser(os.path.join('~','ml_datasets','DJI_ROCO', 'test_crop_poss.list')),
                    DJIROCO_dir=os.path.join('~','ml_datasets','DJI_ROCO'), test_split_name="testFiles",
                    splits_mode='include'):
    """
    Get the DJIROCO dataset for testing.
    The DJIROCO test dataset has 2074 images, about 20% of total number of images.
    ----------------------
    Parameters:
    crop_poss_filepath : String, optional,
        default is os.path.expanduser(os.path.join('~','ml_datasets','DJI_ROCO', 'test_crop_poss.list'))
        the absolute filepath of the crop_poss file.
    DJIROCO_dir : String, optional,
        default is os.path.join('~','ml_datasets','DJI_ROCO')
        the directory path of the DJIROCO dataset, can be relative or absolute path.
    test_split_name : String, optional, default is 'testFiles'.
        the filename of the split .txt file, parameter passed to constructor of Class DJIROCO.
    splits_mode : String, optional, default is 'include'.
        the mode of splits in {'include', 'exclude'}, parameter passed to contructor of Class DJIROCO.
    """
    return DJIROCO(DJIROCO_dir, splits=test_split_name, splits_mode = splits_mode,
                transform=crop_image_and_bbox_manual, crop_poss=get_test_crop_poss(crop_poss_filepath))

def get_test_crop_poss(filepath):
    import pickle
    with open(filepath, 'rb') as filehandle:
        test_crop_poss = pickle.load(filehandle)
    return test_crop_poss
# ----------------------------------------------------------------------------------------------------- #
########################################### test dataset ################################################
#########################################################################################################