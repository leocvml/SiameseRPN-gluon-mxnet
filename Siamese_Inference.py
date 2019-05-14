from DateLoader import VOT_DataLoader, LoadDataset, plotImg
import mxnet as mx
from Net_Siamese import SiameseRPN,SiameseForward
from mxnet import image
import matplotlib.pyplot as plt
import numpy as np
from mxnet import nd
import time
import cv2

default_w = 85
default_h = 100

ctx = mx.gpu()
scales = [ [5,5],[7,7],[9,9] ]
num_anchor = len(scales)
class_name = ['obj', 'back']
Net = SiameseRPN(anchor_k=len(scales))
Net.initialize(ctx=ctx)
Net.load_parameters('SiameseRPN_W.params')

data_shape = [256, 256]
batch_size = 20
rgb_mean = nd.array([123, 117, 104])
rgb_std = nd.array([58.395, 57.12, 57.375])

def _FarAwayCenter(output):
    half = output.shape[1] //2
    mask = mx.ndarray.zeros_like(output)
    mask[:, half - 3 :half + 3, half - 3 :half + 3, :, :] = 1
    return mask

def _cosine_window(output):
    size = output.shape[1]
    h = np.hamming(size)
    cosWindow = np.sqrt(np.outer(h, h))
    window = mx.ndarray.array(cosWindow, ctx=output.context).expand_dims(2)
    cos_list =[]
    for _ in range(output.shape[3]):
        cos_list.append(window)
    cosWindow = mx.ndarray.concat(*cos_list,dim=2)
    cosWindow = cosWindow.expand_dims(0)
    return cosWindow

def _penaltyfun(last_w, last_h, pimg_w, pimg_h,num_anchor,k= 0.25):

    eps = 10e-5
    b,w,h,a,s = pimg_w.shape
    # last_w = last_w.asnumpy()
    # last_h = last_h.asnumpy()
    ctx = pimg_w.context
    pimg_w = pimg_w.asnumpy()
    pimg_h = pimg_h.asnumpy()
    pimg_w = np.reshape(pimg_w,-1)
    pimg_h = np.reshape(pimg_h,-1)
   # print(pimg_w.shape)

    padd = (pimg_w + pimg_h) /2
    s = (pimg_w + padd) * (pimg_h + padd)
    s = np.sqrt(s)

    padd_bar = (last_w + last_h) /2
    s_bar =  (last_w + padd_bar) * (last_h + padd_bar)
    s_bar = np.sqrt(s_bar)

    r = (last_h / (last_w +eps))
    r_bar = (pimg_h / (pimg_w+eps))


    maxr =  np.maximum((r/(r_bar+eps)), (r_bar / (r+eps)))
    maxs =  np.maximum((s/(s_bar+eps)), (s_bar / (s+eps)))



    penalty = np.exp(-((maxr * maxs) - 1.) * k)
    penalty[np.isnan(penalty)] = 0
    penalty = penalty.reshape(1,17,17,num_anchor,1)
    penalty = mx.ndarray.array(penalty,ctx=ctx)
    return penalty

def rescore(score, num_anchor):

    discard = _FarAwayCenter(score)
    score = discard * score

    score = mx.ndarray.reshape(score,(0,0,0,-1))
    #print(score.shape)
    cos_window = _cosine_window(score)
    score = score * cos_window
    score = mx.ndarray.reshape(score, (0, 0, 0, num_anchor, -1))
    return score



def box_to_rect(box, color, linewidth=3):
    """convert an anchor box to a matplotlib rectangle"""
    box = box.asnumpy()
    return plt.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
        fill=False, edgecolor=color, linewidth=linewidth)


colors = ['blue', 'green', 'red', 'black', 'magenta']


def display(im, out, default_w, default_h,threshold=0.5):
    print(im.dtype)
    im = mx.ndarray.transpose(im, (1, 2, 0)).astype('uint8')
    im = im[:,:,::-1]
    im = im.asnumpy()

    cv2.imshow('img', im)
    cv2.waitKey(0)

    class_names = ['obj', 'back']
    for row in out:
        row = row.asnumpy()
        class_id, score = int(row[0]), row[1]

        if class_id < 0 or score < threshold:
            continue
        color = colors[class_id % len(colors)]
        box = row[2:6] * np.array([im.shape[1], im.shape[0]] * 2)

        cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)


        cv2.putText(im, str(score), (int(box[0]) + 20, int(box[1]) + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)

        default_w = box[2] - box[0]
        default_h = box[3] - box[1]
    cv2.imshow('result',im)
    cv2.waitKey(0)

    return default_w,default_h

batch_size = 150
test_iter = LoadDataset(batch_size,shuffle=False,test=True)


def cropImg(img, bbox):
    x,y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    w = bbox[3]
    h = bbox[2]

    p = ((w + h) / 2)
    A = int((np.sqrt((w + p) * (h + p))))
    center =np.array([x + (0.5 * w),y + (0.5 * h)])
    center = center.astype('int32')

    luy = np.clip((center[1] - (A // 2)), 0, img.shape[0])
    rdy = np.clip((center[1] + (A // 2)), 0, img.shape[0])
    lux = np.clip((center[0] - (A // 2)), 0, img.shape[1])
    rdx = np.clip((center[0] + (A // 2)), 0, img.shape[1])
    img = img[luy:rdy, lux:rdx, :]

    img = image.imresize(img, 127, 127)

    img = img.astype('float32') / 255
    norm_img = mx.image.color_normalize(img,
                                          mean=mx.nd.array([0.485, 0.456, 0.406]),
                                          std=mx.nd.array([0.229, 0.224, 0.225]))
    norm_img = norm_img.expand_dims(0)
    norm_img = mx.ndarray.transpose(norm_img,(0,3,1,2))
    return norm_img

for TempImgs, DetImgs, bboxes in test_iter:
    break

fake_t = mx.ndarray.random_uniform(shape=(1,3,127,127), ctx=ctx)
fake_d = mx.ndarray.random_uniform(shape=(1,3,255,255), ctx=ctx)

cls_branch, bbox_branch = Net(fake_t, fake_d)
frame = cv2.imread('VOT/bag/00000001.jpg')
mxImg = mx.image.imread('VOT/bag/00000001.jpg')

while True:

    fromCenter = False
    print("draw init BBOX -> press enter")
    Init_xywh = cv2.selectROI('MultiTracker', frame, fromCenter)


    # colors.append((255, 0, 0))

    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv2.waitKey(0) & 0xFF
    if (k == 113):  # q is pressed
        break

t_img = cropImg(mxImg,Init_xywh)

for i in range(DetImgs.shape[0]):
    test_d = DetImgs[i].expand_dims(0)
    test_t = t_img.as_in_context(ctx)
    test_d = test_d.as_in_context(ctx)

    print(test_t.shape, test_d.shape)

    cls_branch, bbox_branch = Net(test_t, test_d)



    cid, score, corner ,p_w,p_h = SiameseForward(cls_branch, bbox_branch, scales, Training=False)
    score = rescore(score,num_anchor)
    print(score.shape)
    _p = _penaltyfun(default_w / 255, default_h / 255, p_w, p_h, len(scales), k=0.25)
    score = score * _p



    output = nd.concat(*[cid, score, corner], dim=4)

    result = nd.contrib.box_nms(output.reshape((0, -1, 6)),topk =1) #,topk =1

    # for idx in range(len(DetImgs)):
    w,h = display(test_d[0], result[0],default_w,default_h,threshold=0.5)

    default_w = w
    default_h = h


#
# import numpy as np
# import cv2
# import mxnet as mx
# from mxnet import gluon, image

