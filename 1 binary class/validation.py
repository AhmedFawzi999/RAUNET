import numpy as np
from torch import nn
import torch
import tqdm
import math
import torchvision

## Validation Function

def val_multi(model: nn.Module, criterion, valid_loader, num_classes,batch_size,device_ids):

    # A function that is used to save the predictions
    save_predictions_as_imgs(valid_loader, model)
    
    # Predicting without changing the grads
    with torch.no_grad():
        model.eval()
        losses = []

        # Create a confusion matrix
        confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.uint32)
        dt_size = len(valid_loader.dataset)
        ## Print some info
        tq = tqdm.tqdm(total=math.ceil(dt_size / batch_size))
        tq.set_description('Validation')

        ## Loop on the Validation Loader 
        for inputs, targets in valid_loader:

            ## Pridection
            inputs = inputs.cuda(device_ids[0])
            targets = targets.float()
            targets = targets.cuda(device_ids[0])
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            output_classes = outputs.data.cpu().numpy().argmax(axis=1)
            # print(outputs.shape,"el shape ah")
            target_classes = targets.data.cpu().numpy()

            ## Confusion Matrix Calculation
            confusion_matrix += calculate_confusion_matrix_from_arrays(
                output_classes, target_classes, num_classes)
            # Print info
            tq.set_postfix(loss='{0:.3f}'.format(np.mean(losses)))
            tq.update(1)
        tq.close()
        ## Printing the validation loss and dice
        confusion_matrix = confusion_matrix  # exclude background
        valid_loss = np.mean(losses)  # type: float
        ious = {'iou_{}'.format(cls + 1): iou
                for cls, iou in enumerate(calculate_iou(confusion_matrix))}

        dices = {'dice_{}'.format(cls + 1): dice
                 for cls, dice in enumerate(calculate_dice(confusion_matrix))}

        average_iou = np.mean(list(ious.values()))
        average_dices = np.mean(list(dices.values()))

        print('Valid loss: {:.4f}'.format(valid_loss))


        return average_dices, average_iou


## Calculating the IOU
def calculate_iou(confusion_matrix):
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return ious

## Calculating the dice
def calculate_dice(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices
    
## Calculating the Confusion Matrix
def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


## This function is responsible to save the predictions and print the dice metric
def save_predictions_as_imgs(
    loader, model, folder="C:/Users/asemi/Desktop/1 binary class/saved_images", device="cuda"
):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    # loop on the val dataloader
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device).unsqueeze(1)
        with torch.no_grad():
            ## Predict the images
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            # Calculation of the Dice Metric
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y, f"{folder}{idx}.png")

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")

       


