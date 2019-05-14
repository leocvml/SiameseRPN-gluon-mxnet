from DateLoader import VOT_DataLoader, LoadDataset, plotImg
import mxnet as mx
from Net_Siamese import SiameseRPN,SiameseForward

import numpy as np
from mxnet import nd
import time
from mxnet import autograd

def SiameseRPN_Target(cls_forward,labels, anchors, ignore_label=-1 ):
    b, h, w, n, _ = cls_forward.shape
    anchors = np.reshape(np.array(anchors), (-1, 2))
    target_score = nd.zeros((b, h, w, n, 1), ctx=cls_forward.context)
    target_id = nd.ones_like(target_score, ctx=cls_forward.context) * ignore_label
    target_box = nd.zeros((b, h, w, n, 4), ctx=cls_forward.context)
    sample_weight = nd.zeros((b, h, w, n, 1), ctx=cls_forward.context)

    for b in range(cls_forward.shape[0]):
        # find the best match for each ground-truth
        l = labels[b].asnumpy()

        gx, gy, gw, gh = (l[0] + l[2]) / 2, (l[1] + l[3]) / 2, l[2] - l[0], l[3] - l[1]

        ind_x = int(gx * w)
        ind_y = int(gy * h)
        tx = gx * w - ind_x
        ty = gy * h - ind_y
        gw = gw * w
        gh = gh * h
        # find the best match using width and height only, assuming centers are identical
        intersect = np.minimum(anchors[:, 0], gw) * np.minimum(anchors[:, 1], gh)
        ovps = intersect / (gw * gh + anchors[:, 0] * anchors[:, 1] - intersect)
        best_match = int(np.argmax(ovps))
        target_id[b, ind_y, ind_x, best_match, :] = 0.0
        target_score[b, ind_y, ind_x, best_match, :] = 1.0
        tw = np.log(gw / anchors[best_match, 0])
        th = np.log(gh / anchors[best_match, 1])
        target_box[b, ind_y, ind_x, best_match, :] = mx.nd.array([tx, ty, tw, th])
        sample_weight[b, ind_y, ind_x, best_match, :] = 1.0
    #print('ind_y', ind_y, 'ind_x', ind_x, 'best_match', best_match, 't', tx, ty, tw, th, 'ovp', ovps[best_match], 'gt', gx, gy, gw/w, gh/h, 'anchor', anchors[best_match, 0], anchors[best_match, 1])
    return target_id, target_score, target_box, sample_weight



ctx = mx.gpu()
batch_size = 30
train_iter = LoadDataset(batch_size,test=False)

# for TempImgs, DetImgs, bboxes in train_iter:
#     break
#
# print(TempImgs.shape)
# print(DetImgs.shape)
# print(bboxes.shape)



##############
##  Net
##############
ctx = mx.gpu()
scales = [[5,5],[7,7],[9,9]]
class_name = ['obj', 'back']
Net = SiameseRPN(anchor_k=len(scales))
Net.initialize(ctx=ctx)

sce_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss(from_logits=False)
#sce_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
l1_loss = mx.gluon.loss.L1Loss()

trainer =mx.gluon.Trainer(Net.collect_params(),'sgd',{'learning_rate': 0.01, 'wd': 5e-4})

epochs = 400
positive_weight = 5.0
negative_weight = 0.1
class_weight = 1.0
box_weight = 5.0
paramsName = 'SiameseRPN_W.params'
#Net.load_parameters('SiameseRPN_W.params')
for epoch in range(epochs):

    tic = time.time()
    if (epoch >0 ) and (epoch % 400 ==0):
        trainer.learning_rate *=0.5
    for t_img, d_img, y in train_iter:

        t_img = t_img.as_in_context(ctx)
        d_img = d_img.as_in_context(ctx)
        y = y.as_in_context(ctx)

        #print(t_img.dtype, d_img.dtype, y.dtype)

        with autograd.record():

            cls_branch, bbox_branch  = Net(t_img, d_img)
            #output, cls_forward, bbox_forward = SiameseForward(cls_branch, bbox_branch, scales)

            cls_forward, bbox_forward = SiameseForward(cls_branch, bbox_branch, scales, Training=True)

            with autograd.pause():
                tid, tscore, tbox, sample_weight = SiameseRPN_Target(cls_forward ,y, scales, ignore_label=-1 )

            loss1 = sce_loss(cls_forward, tid, sample_weight * class_weight)
            score_weight = nd.where(sample_weight > 0,
                                    nd.ones_like(sample_weight) * positive_weight,
                                    nd.ones_like(sample_weight) * negative_weight)

            loss3 = l1_loss(bbox_forward, tbox, sample_weight * box_weight)
            loss = loss1 + loss3
        loss.backward()

        trainer.step(batch_size)

    print('Epoch %2d , class:  %.5f, bbox: %.5f time %.1f sec'
          % (
          epoch, mx.ndarray.mean(loss1).asscalar() , mx.ndarray.mean(loss3).asscalar(),
          time.time() - tic))
    Net.save_parameters(paramsName)




from mxnet import image
import matplotlib.pyplot as plt
import cv2

data_shape = [256,256]
batch_size = 20
rgb_mean = nd.array([123, 117, 104])
rgb_std = nd.array([58.395, 57.12, 57.375])

def box_to_rect(box, color, linewidth=3):
    """convert an anchor box to a matplotlib rectangle"""
    box = box.asnumpy()
    return plt.Rectangle(
        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
        fill=False, edgecolor=color, linewidth=linewidth)

colors = ['blue', 'green', 'red', 'black', 'magenta']



def display(im, out, threshold=0.99):

    im = mx.ndarray.transpose(im,(1,2,0)).astype('uint8')
    im = im.asnumpy()
    plt.imshow(im)

    class_names = ['obj','back']
    for row in out:
        row = row.asnumpy()
        class_id, score = int(row[0]), row[1]


        if class_id < 0 or score < threshold:
            continue
        color = colors[class_id%len(colors)]
        box = row[2:6] * np.array([im.shape[1],im.shape[0]]*2)

        rect = box_to_rect(nd.array(box), color, 2)
        plt.gca().add_patch(rect)
        text = class_names[class_id]
        plt.gca().text(box[0], box[1],
                       '{:s} {:.2f}'.format(text, score),
                       bbox=dict(facecolor=color, alpha=0.5),
                       fontsize=10, color='white')
    plt.show()


batch_size = 30
test_iter = LoadDataset(batch_size,shuffle=True)

for TempImgs, DetImgs, bboxes in test_iter:
    break

Net.load_parameters('SiameseRPN_W.params')

#plotImg(TempImgs, DetImgs, bboxes)

test_t = TempImgs[0].expand_dims(0)
for i in range(DetImgs.shape[0]):
    test_d = DetImgs[i].expand_dims(0)
    test_t = test_t.as_in_context(ctx)
    test_d = test_d.as_in_context(ctx)

    cls_branch, bbox_branch = Net(test_t, test_d)
    cid, score, corner ,xywh = SiameseForward(cls_branch, bbox_branch, scales, Training=False)

    output = nd.concat(*[cid, score, corner], dim=4)
    print(xywh.shape)
    result = nd.contrib.box_nms(output.reshape((0, -1, 6)))


    #for idx in range(len(DetImgs)):
    display(test_d[0], result[0], threshold=0.3)
