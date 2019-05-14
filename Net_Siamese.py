import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import nd
import numpy as np




class SiameseRPN(gluon.nn.HybridBlock):
    def __init__(self,anchor_k):
        super(SiameseRPN, self).__init__()
        self.k = anchor_k
        self.Siamese = gluon.nn.HybridSequential()
        with self.Siamese.name_scope():
            self.Siamese.add(
                nn.Conv2D(64, kernel_size=11, strides=2),
                nn.Activation('relu'),
                nn.MaxPool2D(pool_size=(3,3), strides=2),
                nn.Conv2D(192, kernel_size=5),
                nn.Activation('relu'),
                nn.MaxPool2D(pool_size=(3, 3), strides=2),
                nn.Conv2D(384, kernel_size=3),
                nn.Activation('relu'),
                nn.Conv2D(256, kernel_size=3),
                nn.Activation('relu'),
                nn.Conv2D(256, kernel_size=3)
            )

        self.ClsBr_Kernel = nn.HybridSequential()
        with self.ClsBr_Kernel.name_scope():
            self.ClsBr_Kernel.add(
                nn.Conv2D(2 * self.k * 256, kernel_size=3),  #2*anchor
                nn.BatchNorm(),
                nn.Activation('relu')
            )

        self.RegBr_kernel = nn.HybridSequential()
        with self.RegBr_kernel.name_scope():
            self.RegBr_kernel.add(
                nn.Conv2D(4* self.k * 256, kernel_size=3),
                nn.BatchNorm(),
                nn.Activation('relu')
            )

        self.ClsBr_FeatMap = nn.HybridSequential()
        with self.ClsBr_FeatMap.name_scope():
            self.ClsBr_FeatMap.add(
                nn.Conv2D(256, kernel_size=3),
                nn.BatchNorm(),
                nn.Activation('relu')
            )

        self.RegBr_FeatMap = nn.HybridSequential()
        with self.RegBr_FeatMap.name_scope():
            self.RegBr_FeatMap.add(
                nn.Conv2D(256, kernel_size=3),
                nn.BatchNorm(),
                nn.Activation('relu')
            )

    def hybrid_forward(self, F, template, detection, *args, **kwargs):

        template = self.Siamese(template)
        detection = self.Siamese(detection)

        cls_kernel = self.ClsBr_Kernel(template)
        cls_featmap = self.ClsBr_FeatMap(detection)

        reg_kernel = self.RegBr_kernel(template)
        reg_featmap = self.RegBr_FeatMap(detection)

        RegOut =[]
        ClsOut =[]

        for cls_f, cls_k, reg_f, reg_k in zip(cls_featmap, cls_kernel, reg_featmap, reg_kernel):
            cls_k =cls_k.reshape(2 * self.k, 256, 4, 4)  # 2*anchor
            reg_k =reg_k.reshape(4 * self.k, 256, 4, 4)

            cls_f = cls_f.expand_dims(0)
            reg_f = reg_f.expand_dims(0)

            cls_out = F.Convolution(data = cls_f, weight= cls_k,
                                    no_bias=True, kernel=[cls_k.shape[2],cls_k.shape[3]],
                                    num_filter=cls_k.shape[0])

            reg_out = F.Convolution(data = reg_f, weight= reg_k,
                                    no_bias=True, kernel=[reg_k.shape[2],reg_k.shape[3]],
                                    num_filter=reg_k.shape[0])



            RegOut.append(reg_out)
            ClsOut.append(cls_out)


        # RegOut = F.concatenate(RegOut)
        #
        # ClsOut = F.concatenate(ClsOut).

        ClsOut = F.concat(*ClsOut, dim=0)
        RegOut = F.concat(*RegOut, dim=0)

        return ClsOut, RegOut


def transform_center(xy):
    """Given x, y prediction after sigmoid(), convert to relative coordinates (0, 1) on image."""
    b, h, w, n, s = xy.shape
    offset_y = nd.tile(nd.arange(0, h, repeat=(w * n * 1), ctx=xy.context).reshape((1, h, w, n, 1)), (b, 1, 1, 1, 1))
    # print(offset_y[0].asnumpy()[:, :, 0, 0])
    offset_x = nd.tile(nd.arange(0, w, repeat=(n * 1), ctx=xy.context).reshape((1, 1, w, n, 1)), (b, h, 1, 1, 1))
    # print(offset_x[0].asnumpy()[:, :, 0, 0])
    x, y = xy.split(num_outputs=2, axis=-1)
    x = (x + offset_x) / w
    y = (y + offset_y) / h
    return x, y

def transform_size(wh, anchors):
    """Given w, h prediction after exp() and anchor sizes, convert to relative width/height (0, 1) on image"""
    b, h, w, n, s = wh.shape
    aw, ah = nd.tile(nd.array(anchors, ctx=wh.context).reshape((1, 1, 1, -1, 2)), (b, h, w, 1, 1)).split(num_outputs=2,
                                                                                                         axis=-1)
    w_pred, h_pred = nd.exp(wh).split(num_outputs=2, axis=-1)
    w_out = w_pred * aw / w
    h_out = h_pred * ah / h
    return w_out, h_out

def SiameseForward(cls_pred, bbox_pred, anchor_scales, Training = True):

    num_anchor = len(anchor_scales)
    cls_pred = mx.ndarray.transpose(cls_pred, (0, 2, 3, 1))
    cls_pred = mx.ndarray.reshape(cls_pred, (0, 0, 0, num_anchor, -1))


    bbox_pred = mx.ndarray.transpose(bbox_pred, (0, 2, 3, 1))
    bbox_pred = mx.ndarray.reshape(bbox_pred, (0, 0, 0, num_anchor, -1))

    #print(bbox_pred.shape )

    xy = bbox_pred.slice_axis(begin=0, end=2, axis=-1)
    xy = mx.ndarray.sigmoid(xy)
    x, y = transform_center(xy)

    wh = bbox_pred.slice_axis(begin=2, end=4 , axis=-1)
    w, h = transform_size(wh, anchor_scales)
    # cid is the argmax channel

    cid = nd.argmax(cls_pred, axis=-1, keepdims=True)

    # print(cls_pred.shape)
    # print(cid.shape)
    half_w = w / 2
    half_h = h / 2
    left = nd.clip(x - half_w, 0, 1)
    top = nd.clip(y - half_h, 0, 1)
    right = nd.clip(x + half_w, 0, 1)
    bottom = nd.clip(y + half_h, 0, 1)
    #output = nd.concat(*[cid,left, top, right, bottom], dim=4)
    if Training:
        return cls_pred, nd.concat(*[xy, wh], dim=4)

    if not Training:

        score = nd.softmax(cls_pred, axis=-1)
        score = nd.max(score,axis=-1, keepdims=True)
        # discard = _FarAwayCenter(score)
        # score = discard * score
        #
        # score = mx.ndarray.reshape(score,(0,0,0,-1))
        # print(score.shape)
        # cos_window =_cosine_window(score)
        # score = score *cos_window
        # score = mx.ndarray.reshape(score, (0, 0, 0, num_anchor,-1))
        # #output = nd.concat(*[cid, score, left, top, right, bottom], dim=4)
        p_w = right - left
        p_h = bottom - top
        return cid, score , nd.concat(*[ left, top, right, bottom], dim=4), p_w, p_h


    #print(cls_pred.shape, bbox_pred.shape)

if __name__ == '__main__':

    temp_img = mx.ndarray.random_uniform(shape=(10,3,127,127))
    Detec_img = mx.ndarray.random_uniform(shape=(10,3,256,256))

    scales = [[0.34923, 0.34923], [0.45, 0.45]]

    Net = SiameseRPN(anchor_k=len(scales))
    Net.initialize()

    cls_branch, bbox_branch = Net(temp_img, Detec_img)

    print(cls_branch.shape, bbox_branch.shape)

    #cls_forward, bbox_forward = SiameseForward(cls_branch, bbox_branch, scales,Training = False)
    # print(cls_forward.shape)
    # print(bbox_forward.shape)

    output,xywh = SiameseForward(cls_branch, bbox_branch, scales, Training=False)
    print(output.shape, xywh.shape)













