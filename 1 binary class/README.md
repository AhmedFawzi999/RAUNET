# RAUNET Network to Segment Left and Right Ventricle

## The models are trained and weights are saved in the Logs Folder:


<p>&nbsp;</p>
<p>&nbsp;</p>

# Files

1- Train.py:

> This file is responsible to intialize the variable and the model and to train the network it also saves the weights of each epoch in the Logs Folder.

2- Valiation.py

> This file is reponsible to validate the network and to display and save the predictions.

3- load_dataset.py

> This file is responsible to load the dataset and to return the data for Images and Groundtruth

4- loss, lossnew,lovasz_loss,focalloss

> These files contain different loss functions that could be used to experiment different losses and test the network but for the training focall loss with gamma=2 was used to achieve results.


<p>&nbsp;</p>
<p>&nbsp;</p>

# Train.py Functions

1- adjust_learning_rate:

> This Function is responsible to set the learning rate to the initial LR decayed by 10 every 30 epochs

2- load_filename Function
> This Function is responsible to load the file names of the Images and GroundTruth

3- train_model Fucntion
> This Function is train the model and to print and save the weights of the network. It also calls the validation function in each epoch to evaluate the model and save the predictions

4- main Function
>This Function is responsible to initialize the models get the dataset and the loaders, intialize the loss and optimizer and call train_model Function.


<p>&nbsp;</p>
<p>&nbsp;</p>

# load_dataset.py Functions

1- Class Load_Dataset:

> This Class is responsible to load the images and Ground truth images . It also is responsible to transform the RGB groundtruth images to masks wheere each pixel corresponds to a class.




<p>&nbsp;</p>
<p>&nbsp;</p>

# validation.py Functions

1- val_multi Function:

> This Function is responsible to evaluate the model and to save the images. It alsp calculates IOU metrics and dice for the prediction.














