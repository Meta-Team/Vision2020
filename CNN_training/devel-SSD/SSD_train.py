import sys
sys.path.insert(0,'..')
from ece445lib import *


#########################################################################################################
##################################### hyperparameter settings ###########################################
# ----------------------------------------------------------------------------------------------------- #
batch_size = 64
ctx = mx.gpu(0)
num_epochs = 50
train_algorithm = 'Adam'
train_alg_params = {'learning_rate': 0.0001, 'wd': 1e-7}
# ----------------------------------------------------------------------------------------------------- #
##################################### hyperparameter settings ###########################################
#########################################################################################################

print("Total number of train epochs is: ", num_epochs)
print("Train batch size is: ", batch_size)
print("The train algorithm is: ", train_algorithm)
print("The train algorithm settings are: ", train_alg_params)
print("The training device is:", str(ctx))

# load dataset
DJIROCO_dataset = DJIROCO(os.path.join('~', 'ml_datasets', "DJI_ROCO"),
                               splits='trainFiles', splits_mode='include',
                         transform=crop_image_and_bbox_random_armor)
print("Dataset loaded, the size of dataset is:", len(DJIROCO_dataset))

sizes, ratios = get_SSD_sizes_and_ratios(config='original')
net_blank = SSD(num_classes=2, sizes=sizes, ratios=ratios, forward_mode='return_anchors')
net_blank.initialize()
X = nd.zeros((batch_size, 3, 480, 640))
anchors, cls_preds, bbox_preds = net_blank(X)
# print('output anchors:', anchors.shape)
# print('output class preds:', cls_preds.shape)
# print('output bbox preds:', bbox_preds.shape)

# load dataloader
train_transform = SSDTrainTransform_DJIROCO(width=640, height=480, anchors=anchors, label_filter=label_filter_armor_2_class,
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), iou_thresh=0.5, box_norm=(0.1, 0.1, 0.2, 0.2))

train_dataloader = gluon.data.DataLoader(DJIROCO_dataset.transform(train_transform), batch_size=batch_size, shuffle=True,
                                         #batchify_fn = batchify_fn_DJIROCO)
                                         batchify_fn=get_batchify_fn_DJIROCO_SSD('train'))
print("DataLoader initialized.")

# net initialization
net = SSD(num_classes=2, sizes=sizes, ratios=ratios, forward_mode='train')
net.initialize(init=init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), train_algorithm, train_alg_params)
print("SSD model initialized.")

# train the net
print("Training the SSD model...")
train_record = train_SSD(net=net, ctx=ctx, trainer=trainer, batch_size=batch_size,
                        dataloader=train_dataloader, num_epochs=num_epochs, model_filename='SSD',
                        cls_bbox_ctx=mx.cpu(0), print_info=True)