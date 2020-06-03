import sys
sys.path.insert(0,'..')
from ece445lib import *

#########################################################################################################
##################################### hyperparameter settings ###########################################
# ----------------------------------------------------------------------------------------------------- #
batch_size = 32
ctx = mx.gpu(0)
model_filename = "SSD.params"
# ----------------------------------------------------------------------------------------------------- #
##################################### hyperparameter settings ###########################################
#########################################################################################################
print("Test batch size is: ", batch_size)
print("The default testing device is:", str(ctx))
print("Testing model filename: ", model_filename)

dataset_test = get_test_DJIROCO()

print("The size of dataset is:", len(dataset_test))

sizes, ratios = get_SSD_sizes_and_ratios(config='original')

test_transform = SSDTrainTransform_DJIROCO(width=640, height=480, anchors=None, label_filter=label_filter_armor_2_class,
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), iou_thresh=0.5, box_norm=(0.1, 0.1, 0.2, 0.2))

test_dataloader = gluon.data.DataLoader(dataset_test.transform(test_transform), batch_size=batch_size, shuffle=False, last_batch='discard',
                                         batchify_fn=get_batchify_fn_DJIROCO_SSD('val'))

net = SSD(num_classes=2, sizes=sizes, ratios=ratios, forward_mode="val")
net.load_parameters(model_filename, ctx=mx.gpu(0))
print("Model parameters successfully loaded.")

print("Doing model inference and evaluating performance...")

# print("Testing model inference IoUs...")
# res = get_ious_SSD(net, test_dataloader, ctx=mx.gpu(0), one_batch=False)
# print("The maximum IOU metric is: ", np.max(res))
# print("The avg IOU metric is: ", np.mean(res))

ROCOAID_acc_list = []
ROCOAUTO_acc_list = []
FPS_list = []
i = 0
for batch in test_dataloader:
    batch_start_time = time.time()
    image_iter, label_iter = batch
    image_iter, label_iter = image_iter.as_in_context(ctx), label_iter.as_in_context(ctx)
    pred_cls, pred_score, pred_bbox = net(image_iter)
    model_inference_res = nd.concat(pred_bbox, pred_cls.reshape(pred_cls.shape[0], pred_cls.shape[1], 1), dim=2)
    Y_true = bbox_2_class_to_ROCOACC_batch(label_iter, absolute_size=(640,480), label_mode='absolute')
    Y_pred = bbox_2_class_to_ROCOACC_batch(model_inference_res, absolute_size=(640, 480), label_mode='relative')
    batch_ROCOAID_acc, batch_ROCOAUTO_acc = nd.mean(ROCOAID_acc(Y_pred, Y_true)).asnumpy()[0], nd.mean(ROCOAUTO_acc(Y_pred, Y_true)).asnumpy()[0]
    ROCOAID_acc_list.append(batch_ROCOAID_acc)
    ROCOAUTO_acc_list.append(batch_ROCOAUTO_acc)
    batch_time = time.time() - batch_start_time
    batch_FPS = batch_size / batch_time
    FPS_list.append(batch_FPS)
    i += 1
    print("Batch "+str(i)+", ROCOAID_acc: "+"{:.6}".format(batch_ROCOAID_acc)+", ROCOAUTO_acc: "
            +"{:.6}".format(batch_ROCOAUTO_acc, end='\r')+", FPS: "+"{:.6}".format(batch_FPS), end='\r')
print("\nOverall ROCOAID_acc:", "{:.6}".format(np.mean(ROCOAID_acc_list)), "overall ROCOAUTO_acc:", 
        "{:.6}".format(np.mean(ROCOAUTO_acc_list)), "overall FPS:", "{:.6}".format(np.mean(FPS_list)))