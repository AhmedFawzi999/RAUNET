import numpy as np
from torch import nn
import torch
import tqdm
import math
import torchvision
from sklearn.metrics import confusion_matrix
from PIL import Image



## Validation Function
def val_multi(model: nn.Module, criterion, valid_loader, num_classes,batch_size,device_ids):

    labels = np.arange(num_classes)
    # confusion matrix to help calculate IOU
    cm = np.zeros((num_classes,num_classes))
    # Folder to save Prediction in
    folder=r"C:\Users\asemi\Desktop\2classes trial\saved_images"
    # Predicting without changing the grads
    with torch.no_grad():
        model.eval()
        losses = []

        dt_size = len(valid_loader.dataset)
        tq = tqdm.tqdm(total=math.ceil(dt_size / batch_size))
        tq.set_description('Validation')
        
        ## Loop on the Validation Loader 
        for idx, (inputs, targets) in enumerate(valid_loader):

            inputs = inputs.cuda(device_ids[0])
            targets = targets.cuda(device_ids[0]).long()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            ## Confusion Matrix Calculation
            tq.set_postfix(loss='{0:.3f}'.format(np.mean(losses)))
            tq.update(1)
            for j in range(len(targets)): 
                true = targets[j].cpu().detach().numpy().flatten()
                pred = preds[j].cpu().detach().numpy().flatten()
                cm += confusion_matrix(true, pred, labels=labels)


            
            ## Displaying the Prediction of the Model
            preds = preds.to('cpu')
            targets = targets.to('cpu')

            ## Mapping the class indices back to colors to save
            preds=preds.numpy()    
            targets=targets.numpy()   
            preds[preds==1]=255
            targets[targets==1]=255     

            ## A function to save the prediciton batches all together
            targets=get_concat_h(targets)
            targets.save(f"{folder}{idx}.png")

            prdddd=get_concat_h(preds)
            prdddd.save(f"{folder}/pred_{idx}.png")

            preds = torch.from_numpy(preds).to(device_ids[0])   

        ## Computing each Class IOU
        class_iou, mean_iou = compute_IoU(cm)
        tq.close()
        valid_loss = np.mean(losses)  # type: float

        print('Valid loss: {:.4f}'.format(valid_loss))
        

        return class_iou, mean_iou


## Computing the IOU Fucntion
def compute_IoU(cm):
    '''
    Adapted from:
        https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py
        https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/metrics.py#L2716-L2844
    '''
    
    sum_over_row = cm.sum(axis=0)
    sum_over_col = cm.sum(axis=1)
    true_positives = np.diag(cm)

    # sum_over_row + sum_over_col = 2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives
    
    iou = true_positives / denominator
    
    return iou, np.nanmean(iou) 


## Concatinating each image and saving Batches together
def get_concat_h(arr):
    dst = Image.new( 'L', (arr.shape[0]*arr.shape[2], arr.shape[2]))
    x=0
    for i in range(arr.shape[0]):
        im = Image.fromarray(np.uint8(arr[i]),mode='L')
        dst.paste(im, (x, 0))
        x+=arr.shape[2]

    return dst

