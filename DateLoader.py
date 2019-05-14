
import glob
import cv2
import numpy as np
import mxnet as mx
from mxnet import gluon, image
from mxnet.gluon import nn

from mxnet.gluon import data as gdata

class VOT_DataLoader(gluon.data.Dataset):
    def __init__(self,rootPath='VOT/',numofSeq=1,seqrange=100,test=True):
        self.rootPath = rootPath
        self.numofSeq = numofSeq
        self.seqrange = seqrange
        self.data, self.GTinfo = self.getDataset()
        self.test = test
       # self.cropSize = 128


    def getDataset(self):
        rootPath = self.rootPath
        with open(rootPath + 'list2.txt', 'r') as f:
            FolderList = f.readlines()
        for folderName in FolderList[:self.numofSeq ]:
            folder_Imgs = glob.glob(rootPath + folderName[:-1] + "/*.jpg")
            with open(rootPath + folderName[:-1] + "/groundtruth.txt", 'r') as f:
                GTinfo = f.readlines()
        data = []
        for idx in range(len(folder_Imgs)):
            img = mx.image.imread(folder_Imgs[idx])
            data.append(img)

        #print(GTinfo)
        return  data, GTinfo

    def cropImg(self,img,bbox,Det=False):

        bboxes = bbox.strip('\n')
        bboxes = bboxes.split(',')
        bboxes = [int(float(x)) for x in bboxes]
        coord = np.array(bboxes).reshape(-1, 2)

        xy_max = np.max(coord, axis=0)
        xy_min = np.min(coord, axis=0)

        w = xy_max[0] - xy_min[0]
        h = xy_max[1] - xy_min[1]

        p = ((w + h) / 2)
        if Det:
            A = int((np.sqrt((w + p) * (h + p)))) * 2
        else:
            A = int((np.sqrt((w + p) * (h + p))))


        center = xy_max / 2 + xy_min / 2
        center = center.astype('int32')

        luy = np.clip((center[1] - (A // 2)), 0, img.shape[0])
        rdy = np.clip((center[1] + (A // 2)), 0, img.shape[0])

        lux = np.clip((center[0] - (A // 2)), 0, img.shape[1])
        rdx = np.clip((center[0] + (A // 2)), 0, img.shape[1])
        img = img[luy:rdy, lux:rdx, :]


        if Det:
           # print(img.shape)
            scale_w= 255/img.shape[1]
            scale_h=255/img.shape[0]
            img = image.imresize(img, 255, 255)
            bboxInDet = mx.ndarray.array([(((center[0]) - lux)*scale_w) / 255, (((center[1] - luy )*scale_h)) / 255, ((w*scale_w) / 255), ((h*scale_h) / 255)])
            #coord = mx.ndarray.array([(center[0] ) /255, center[1] /255, w/255, h/255])
            return img,bboxInDet

        else:
            img = image.imresize(img,127,127)
            return img



    def cropBoundingbox_center(self, tempImg, detImg, tempbbox, detbbox):

        cropTempImg = self.cropImg(tempImg, tempbbox)
        cropDetImg,coord = self.cropImg(detImg, detbbox,Det=True)

        return cropTempImg, cropDetImg,coord

    def normalize_image(self, data):
        data = data.astype('float32') / 255
        normalized = mx.image.color_normalize(data,
                                              mean=mx.nd.array([0.485, 0.456, 0.406]),
                                              std=mx.nd.array([0.229, 0.224, 0.225]))
        return normalized


    def __getitem__(self, item):

        endidx = item + self.seqrange if item + self.seqrange < len(self.data) else len(self.data)
        if not self.test:
            detIdx = np.random.randint(item, endidx)
        if self.test:
            detIdx = item+1
        tempImg = self.data[item]
        detImg = self.data[detIdx]

        cropTempImg, cropDetImg,Gtbbox = self.cropBoundingbox_center(tempImg, detImg, self.GTinfo[item], self.GTinfo[detIdx])
        #cropTempImg = self.normalize_image(cropTempImg)
        #cropDetImg = self.normalize_image(cropDetImg)

        x, y, w, h = Gtbbox[0], Gtbbox[1], Gtbbox[2], Gtbbox[3]
        #print(w,h)
        lux = x-w/2
        luy = y-h/2
        rdx = x+w/2
        rdy = y+h/2

        Gtbbox[0] =lux
        Gtbbox[1] =luy
        Gtbbox[2] =rdx
        Gtbbox[3] =rdy

        #x, y, w, h = Gtbbox[0], Gtbbox[1], Gtbbox[2], Gtbbox[3]
        # lux = (Gtbbox[0] - Gtbbox[2]) /2
        # luy = (Gtbbox[1] - Gtbbox[3]) /2
        # rdx = (Gtbbox[0] + Gtbbox[2]) /2
        # rdy = (Gtbbox[1] + Gtbbox[3]) /2
        # Gtbbox[0] =  lux
        # Gtbbox[1] =  luy
        # Gtbbox[2] =  rdx
        # Gtbbox[3] =  rdy

        return cropTempImg.transpose((2, 0, 1)).astype('float32'), cropDetImg.transpose((2, 0, 1)).astype('float32'), Gtbbox

    def __len__(self):
        return len(self.data)

    # def show(self):
    #     tempIdx = np.random.randint(len(self.folder_Imgs)-100)
    #     detIdx = np.random.randint(self.seqrange) +tempIdx
    #     print(tempIdx,detIdx)
    #     tempImg = mx.image.imread(self.folder_Imgs[tempIdx])
    #     detImg = mx.image.imread(self.folder_Imgs[detIdx])
    #
    #     cropTempImg, cropDetImg = self.cropBoundingbox_center(tempImg,detImg , self.GTinfo[tempIdx], self.GTinfo[detIdx])
    #     print(cropTempImg.shape,cropDetImg.shape)
    #     # for idx, f_img in enumerate(self.folder_Imgs):
    #     #    # img = cv2.imread(f_img)
    #     #     img = mx.image.imread(f_img )
    #     #    tempImg, detImg =self.cropBoundingbox_center(img, self.GTinfo[idx],crop_size=128)  # ori or center

def LoadDataset(batchsize,shuffle=False,test=True):
    dataset = VOT_DataLoader(test=test)

    data_iter = gdata.DataLoader(dataset, batchsize, shuffle=shuffle)

    return data_iter

def showndimg(img):
    img = mx.ndarray.transpose(img, (0, 2, 3, 1)).astype('uint8')

    img = img.asnumpy()

    return img


def center2cornr(bbox):
    bbox = [int(x * 255) for x in bbox]
    lux, luy, rdx, rdy = bbox[0], bbox[1], bbox[2], bbox[3]
    #x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    # lux = x-w//2
    # luy = y-h//2
    # rdx = x+w//2
    # rdy = y+h//2

    return lux, luy, rdx, rdy


def plotImg(Tempimgs, Detimgs, bbox):
    bbox = bbox.asnumpy()
    Detimgs = showndimg(Detimgs)
    Tempimgs = showndimg(Tempimgs)

    for idx, detimg in enumerate(Detimgs):
        lux, luy, rdx, rdy = center2cornr(bbox[idx])
        cv2.rectangle(detimg, (lux, luy), (rdx, rdy), (0, 255, 0), 2)
        print(Tempimgs.shape)
        cv2.imshow('Tempimg', Tempimgs[idx, :, :, ::-1])
        cv2.imshow('Detimg', detimg[:, :, ::-1])
        cv2.waitKey(0)


if __name__ == "__main__":



    ctx = mx.gpu()
    batch_size = 30
    train_iter = LoadDataset(batch_size)





    for TempImgs, DetImgs, bboxes in train_iter:
        break
    print(TempImgs.shape)
    print(DetImgs.shape)
    print(bboxes.shape)
    plotImg(TempImgs, DetImgs, bboxes)


