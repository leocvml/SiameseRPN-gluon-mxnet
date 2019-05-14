import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import nd





class YOLOTrack(gluon.nn.HybridBlock):
    def __init__(self,anchor_k, num_class):
        super(YOLOTrack, self).__init__()
        self.k = anchor_k
        self.numcls = num_class
        self.YOLOTrack = gluon.nn.HybridSequential()
        with self.YOLOTrack.name_scope():
            self.YOLOTrack.add(
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


        self._kernel = nn.HybridSequential()
        with self._kernel.name_scope():
            self._kernel.add(
                nn.Conv2D((self.numcls +1 + 4) * self.k * 256, kernel_size=3),
                nn.BatchNorm(),
                nn.Activation('relu')
            )

        self._featMap = nn.HybridSequential()
        with self._featMap.name_scope():
            self._featMap.add(
                nn.Conv2D(256, kernel_size=3),
                nn.BatchNorm(),
                nn.Activation('relu')
            )



    def hybrid_forward(self, F, template, detection, *args, **kwargs):

        template = self.YOLOTrack(template)
        detection = self.YOLOTrack(detection)

        kernelmap  = self._kernel(template)
        featuremap = self._featMap(detection)

        OutPut = []


        for k, f  in zip(kernelmap, featuremap ):
            k = k.reshape(7 * self.k, 256, 4, 4)  # 2*anchor
            f = f.expand_dims(0)
            out = F.Convolution(data=f, weight=k,
                                    no_bias=True, kernel=[k.shape[2], k.shape[3]],
                                    num_filter=k.shape[0])

            OutPut.append(out)

        result = F.concat(*OutPut, dim=0)

        return result


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



def YOLOTrackForward(x, num_class, anchor_scales):
    """Transpose/reshape/organize convolution outputs."""
    stride = num_class + 5
    # transpose and reshape, 4th dim is the number of anchors
    x = x.transpose((0, 2, 3, 1))
    x = x.reshape((0, 0, 0, -1, stride))
    # now x is (batch, m, n, stride), stride = num_class + 1(object score) + 4(coordinates)
    # class probs
    cls_pred = x.slice_axis(begin=0, end=num_class, axis=-1)
    # object score
    score_pred = x.slice_axis(begin=num_class, end=num_class + 1, axis=-1)
    score = nd.sigmoid(score_pred)
    # center prediction, in range(0, 1) for each grid
    xy_pred = x.slice_axis(begin=num_class + 1, end=num_class + 3, axis=-1)
    xy = nd.sigmoid(xy_pred)
    # width/height prediction
    wh = x.slice_axis(begin=num_class + 3, end=num_class + 5, axis=-1)
    # convert x, y to positions relative to image
    x, y = transform_center(xy)
    # convert w, h to width/height relative to image
    w, h = transform_size(wh, anchor_scales)
    # cid is the argmax channel
    cid = nd.argmax(cls_pred, axis=-1, keepdims=True)
    # convert to corner format boxes
    half_w = w / 2
    half_h = h / 2
    left = nd.clip(x - half_w, 0, 1)
    top = nd.clip(y - half_h, 0, 1)
    right = nd.clip(x + half_w, 0, 1)
    bottom = nd.clip(y + half_h, 0, 1)
    output = nd.concat(*[cid, score, left, top, right, bottom], dim=4)
    return output, cls_pred, score, nd.concat(*[xy, wh], dim=4)

if __name__ == '__main__':

    temp_img = mx.ndarray.random_uniform(shape=(10,3,127,127))
    Detec_img = mx.ndarray.random_uniform(shape=(10,3,256,256))

    scales = [[0.34923, 0.34923], [0.45, 0.45]]
    class_name=['obj', 'back']

    Net = YOLOTrack(anchor_k=len(scales), num_class=len(class_name))
    Net.initialize()

    out = Net(temp_img, Detec_img)


    output, cls_pred, score, xywh = YOLOTrackForward(out, len(class_name), scales)
    print(output.shape,
          cls_pred.shape,
          score.shape,
          xywh.shape)

