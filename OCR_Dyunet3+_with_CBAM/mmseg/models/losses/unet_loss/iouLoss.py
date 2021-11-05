import torch
import torch.nn.functional as F

def _iou(pred, target, size_average = True):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

def _miou(pred,target,size_average=True):
    b = pred.shape[0]
    c = pred.shape[1]
    mIoU = 0.0
    for i in range(0,b):
        IoU = 0.0
        for j in range(1,c): # except background
            Iand1 = torch.sum(target[i,j,:,:]*pred[i,j,:,:])
            Ior1 = torch.sum(target[i,j,:,:]) + torch.sum(pred[i,j,:,:])-Iand1
            IoU1 = Iand1/Ior1

            IoU += IoU1
        mIoU1 = IoU/(c-1) # except background
        mIoU = mIoU + (1-mIoU1)
    return mIoU/b


class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        ## custom
        target = F.one_hot(target,11).float() # B x H x W x num_classes
        target = target.permute(0,3,1,2)
        ##
        return _miou(pred, target, self.size_average)

def IOU_loss(pred,label):
    iou_loss = IOU(size_average=True)
    iou_out = iou_loss(pred, label)
    #print("iou_loss:", iou_out.data.cpu().numpy())
    return iou_out
