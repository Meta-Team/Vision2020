import sys
sys.path.insert(0,'..')
from ece445lib import *

#########################################################################################################
##################################### hyperparameter settings ###########################################
# ----------------------------------------------------------------------------------------------------- #
num_epochs = 2
batch_size = 12
lr = 0.0001
ctx = [mx.gpu(0)]
model_filename = "YOLOv3centernet0_25"
# ----------------------------------------------------------------------------------------------------- #
##################################### hyperparameter settings ###########################################
#########################################################################################################

print("Total number of train epochs is: ", num_epochs)
print("Train batch size is: ", batch_size)
print("The learning rate is: ", lr)
print("The training device is:", str(ctx))
print("The name of YOLOv3 model will be set as:", model_filename)

# load dataset
print("Loading dataset...")
DJIROCO_dataset = DJIROCO(os.path.join('~', 'ml_datasets', "DJI_ROCO"),
                              #splits='mini_train', splits_mode='include',
                              #splits='trainFiles', splits_mode='include',
                         transform=crop_image_and_bbox_random_armor)
num_batches = len(DJIROCO_dataset) // batch_size
print("Dataset loaded, the size of dataset is:", len(DJIROCO_dataset))

# net initialization
print("Initializing YOLOv3 model...")
net = gluoncv.model_zoo.yolo3_mobilenet0_25_custom(classes=['red armor','blue armor'], transfer=None, pretrained_base=False, pretrained=False)
net.initialize(force_reinit=True, ctx=ctx)
net.hybridize()
print("YOLOv3 model intialized.")

# load dataloader
print("Initializing dataloader...")
train_transform = YOLOv3TrainTransform_DJIROCO(width=416, height=416, net=net, label_filter=label_filter_armor_2_class,
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

train_dataloader = gluon.data.DataLoader(DJIROCO_dataset.transform(train_transform), batch_size=batch_size, shuffle=True,
                                         batchify_fn=get_batchify_fn_DJIROCO_YOLOv3("train"))
print("DataLoader initialized.")

# train the model, save the model, and save the train records.
train_record = train_YOLOv3(net, ctx, train_dataloader, num_epochs, num_batches, lr, batch_size, model_filename)