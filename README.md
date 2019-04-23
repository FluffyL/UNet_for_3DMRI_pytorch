# UNet_for_3DMRI_pytorch
An UNet for 3D MRI images treatment with pytorch   
train with LPBA40 dataset  
Images are croped into 32x32x32 patches(with 4 overlap in each side) before training  
4 maxpools and 4 upsamples  
loss function: cross_entropy, simple dice, dice with IOU(tversky loss), focal loss  
the training accuracy of the 1st epoch training with LPBA40 is 87.7 and the testing accuracy is 87.9
