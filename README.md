# UNet_for_3DMRI_pytorch
An UNet for 3D MRI images treatment with pytorch   
train with LPBA40 dataset  
Images are croped into 32x32x32 patches(with 4 overlap in each side) before training  
4 maxpools and 4 upsamples  
loss function: cross_entropy, simple dice(not yet), dice with IOU(not yet), focal loss(not yet)
