import os
import time
import nibabel as nib
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd

from predata_LPBA40 import load_dataset,prepare_validation,crop_data,convert_one_hot



RANDOM_SEED = 123
patch_size = 32
dataset_dir = '/LPBA40/native_space/'
croped_dir = '/LPBA40/train_data/'
s_train = 30#the num of subject used for training

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class MRIDataset(Dataset):
	"""loading MRI data"""
	def __init__(self, croped_dir, mode,t_num, img_transform=None, labels_transform=None):

		self.croped_dir = croped_dir
		self.mode = mode
		self.t_num = t_num
		#self.labels = convert_labels(labels)
		self.img_transform = img_transform
		self.labels_transform = labels_transform

	def __getitem__(self, index):

		[T1,label] = load_dataset(self.croped_dir, index, self.mode)
		#if read .hdr files
		T1 = T1[:,:,:,0]
		label = label[:,:,:,0]

		if self.img_transform is not None:
			T1 = self.img_transform(T1)
		if self.labels_transform is not None:
			label = self.labels_transform(label)

		return T1, label

	def __len__(self):
		return self.t_num



img_transform = transforms.Compose([
	#transforms.RandomAffine(degrees=(-20, 20), translate=(0.15, 0.15)),
	transforms.ToTensor(), 
	])

labels_transform = transforms.Compose([
	#transforms.RandomAffine(degrees=(-20, 20), translate=(0.15, 0.15)),
	transforms.ToTensor(), 
	])



total_train = 0
total_test = 0

for subject_id in range(1,s_train+1):
	total_train = total_train + crop_data(dataset_dir, croped_dir, subject_id, total_train, mode=0, patch_size=32)

for subject_id in range(s_train+1,41):
	total_test = total_test + crop_data(dataset_dir, croped_dir, subject_id, total_test, mode=1, patch_size=32)

BATCH_SIZE = 8
train_dataset = MRIDataset(croped_dir,
			t_num = total_train,
			mode = 0,
			img_transform = img_transform,
			labels_transform = img_transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)

valid_dataset = MRIDataset(croped_dir,
			t_num = total_test,
			mode = 1,
			img_transform = img_transform,
			labels_transform = img_transform)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)
'''
predict_dataset = predictDataset(dataset_dir = dataset_dir,
			num_s = 31,
			num_e = 40,
			img_transform = None,
			labels_transform = None)

predict_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)
'''
torch.manual_seed(0)

num_epochs = 2
for epoch in range(num_epochs):

	for batch_idx, (T1, labels) in enumerate(train_loader):

		print('Epoch:', epoch+1, end='')
		print(' | Batch index:', batch_idx, end='')
		print(' | Batch size:', labels.size()[0])
		
		T1 = T1.view(-1, 1,patch_size,patch_size,patch_size).to(DEVICE)
		labels = labels.view(-1, 1,patch_size,patch_size,patch_size).to(DEVICE)
		print('Image shape', T1.shape)
		print('break minibatch for-loop')
		break

#################################
### UNet Model
#################################

class double_conv(nn.Module):
	'''(conv => BN => ReLU) * 2'''
	def __init__(self, in_ch, out_ch):
		super(double_conv, self).__init__()
		self.conv = nn.Sequential(
		nn.Conv3d(in_ch, out_ch, 3, padding=1),
		nn.BatchNorm3d(out_ch),
		nn.ReLU(inplace=True),
		nn.Conv3d(out_ch, out_ch, 3, padding=1),
		nn.BatchNorm3d(out_ch),
		nn.ReLU(inplace=True))
	
	def forward(self, x):
		x = self.conv(x)
		return x


class inconv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(inconv, self).__init__()
		self.conv = double_conv(in_ch, out_ch)

	def forward(self, x):
		x = self.conv(x)
		return x


class outconv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(outconv, self).__init__()
		self.conv = nn.Conv3d(in_ch, out_ch, 1)

	def forward(self, x):
		x = self.conv(x)
		return x


class down(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(down, self).__init__()
		self.mpconv = nn.Sequential(
		nn.MaxPool3d(2),
 		double_conv(in_ch, out_ch))

	def forward(self, x):
		x = self.mpconv(x)
		return x

class up(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(up, self).__init__()
		self.upconv = nn.Sequential(
 		double_conv(in_ch, out_ch),
		nn.Upsample(scale_factor=2, mode='nearest'))

	def forward(self, x):
		x = self.upconv(x)
		return x

class UnetConvBlock(nn.Module):
	def __init__(self, in_size, out_size, is_batchnorm, num_layers=2):
		super(UnetConvBlock, self).__init__()

		self.convs = nn.ModuleList()
		if is_batchnorm:
			conv = nn.Sequential(nn.Conv3d(in_size, out_size, 3, 1, padding=1),
				nn.BatchNorm3d(out_size),
				nn.ReLU())
			self.convs.append(conv)
			for i in range(1, num_layers):
				conv = nn.Sequential(nn.Conv3d(out_size, out_size, 3, 1, padding=1),
					nn.BatchNorm3d(out_size),
					nn.ReLU())
				self.convs.append(conv)
		else:
			conv = nn.Sequential(nn.Conv3d(in_size, out_size, 3, 1, padding=1),
					nn.ReLU())

			self.convs.append(conv)
			for i in range(1, num_layers):
				conv = nn.Sequential(nn.Conv3d(out_size, out_size, 3, 1, padding=1),
						nn.ReLU())
				self.convs.append(conv)

	def forward(self, inputs):
		outputs = inputs
		for conv in self.convs:
			outputs = conv(outputs)
		return outputs
	
class UNet(nn.Module):
	def __init__(self, n_channels, n_classes):
		super(UNet, self).__init__()
		self.inc = inconv(n_channels, 32)
		self.down1 = down(32, 64)
		self.down2 = down(64, 128)
		self.down3 = down(128, 256)
		#self.down4 = down(256, 512)
		self.center = UnetConvBlock(256,512,is_batchnorm = True)
		self.up4 = up(512, 256)
		self.up3 = up(256, 128)
		self.up2 = up(128, 64)
		self.up1 = up(64, 32)
		self.up0 = up(32,  n_classes)

	def forward(self, x):
		x1 = self.inc(x.float())
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.center(x4)
		x4 = self.up4(x5)
		x3 = self.up3(x4)
		x2 = self.up2(x3)
		x1 = self.up1(x2)
		x0 = self.up0(x1)
		s0 = x.size()[2]
		s1 = x.size()[3]
		s2 = x.size()[4]
		self.out = nn.Upsample(size = (s0,s1,s2),mode='trilinear')
		logists = self.out(x0)
		probas = F.softmax(logists)
		return logists,probas

#################################
### Model Initialization
#################################
torch.manual_seed(RANDOM_SEED)

model = UNet(n_channels=1, n_classes=4)
model = model.float()

optimizer = torch.optim.SGD(model.parameters(), lr=0.5,momentum=0.9)

#################################
### Training
#################################

def dice_IOU(predict, label):
	#define loss function
	alpha = 0.5
	beta = 0.5
	
	ones = np.ones(np.shape(label))
	num = np.sum(p0*g0)
	den = num + alpha * np.sum(p0*g1) + beta * np.sum(p1*g0)
	T = np.sum(num/den)
	label = label.view(predict.shape[-1])
	return label-T

def Dice_3D(predict,label):
	label = convert_one_hot(label)
	label_num = label.shape
	predict = predict.detach().numpy()
	cost = 0 
	for i in range(label_num[1]):
		p0 = np.zeros(np.shape(label[:,i,:,:,:]))
		p0 = label[:,i,:,:,:]
		ones = np.ones(np.shape(p0))
		g0 = predict[:,0,:,:,:]

		p_true = p0*g0
		p_true[p_true < i] = 0

		p_true_ones = np.zeros(np.shape(p_true))
		p_true_ones[p_true > 0] = 1
		
		p_predict = np.zeros(np.shape(p_true))
		p_predict[p0 - i > 0] = 1
		p_predict[p0 - i > 1] = 0

		dice = 2*np.sum(p_true)/(np.sum(g0) + np.sum(p_predict))
		cost = cost + dice
	return (-1)*cost

def focal_loss(predict, label):
	gamma = 2
	alpha = 0.25

	return 0
						

def compute_epoch_loss(model, data_loader):
	curr_loss, num_examples = 0., 0
	loss_func = dice()
	with torch.no_grad():
		for T1, labels in data_loader:
			T1 = T1.view(-1,1,patch_size,patch_size,patch_size).to(DEVICE)
			labels = labels.view(-1,1,patch_size,patch_size,patch_size).to(DEVICE)
			logits, probas = model(T1)
			#labels = convert_one_hot(probas)
			#loss = 0
			#loss = Dice_3D(probas,labels)/BATCH_SIZE
			cost = F.cross_entropy(logits, labels)
			
			num_examples += labels.size(0)
			curr_loss += loss

		curr_loss = curr_loss / num_examples
		return curr_loss

################################################
# THE compute_accuracy function
###############################################

def compute_accuracy(model, data_loader):
	correct_pred, num_examples = 0, 0
	with torch.no_grad():
		for T1, labels in data_loader:
			T1 = T1.view(-1,1,patch_size,patch_size,patch_size).to(DEVICE)
			labels = labels.view(-1,1,patch_size,patch_size,patch_size).to(DEVICE)
			logits, probas = model.forward(T1)
			predicted_labels = torch.argmax(probas, 1)
			num_examples += labels.size(0)
			correct_pred += (predicted_labels == labels).sum()
		return correct_pred.float()/num_examples * 100


start_time = time.time()
minibatch_cost = []
epoch_cost = []

NUM_EPOCHS = 60

for epoch in range(NUM_EPOCHS):
	model.train()
	cost = 0
	for batch_idx, (T1, labels) in enumerate(train_loader):
		T1 = T1.view(-1,1,patch_size,patch_size,patch_size).to(DEVICE)
		#labels = labels.view(-1,patch_size,patch_size,patch_size).to(DEVICE)
		labels = labels.long().to(DEVICE)
            
		### FORWARD AND BACK PROP
		logits, probas = model(T1)
		
		#cost = Dice_3D(probas, labels)/BATCH_SIZE
		cost = F.cross_entropy(logits, labels)
		optimizer.zero_grad()
        
		cost.backward()
		minibatch_cost.append(cost)
		### UPDATE MODEL PARAMETERS
		optimizer.step()
        
		### LOGGING
		if not batch_idx % 50:
			print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
 				%(epoch+1, NUM_EPOCHS, batch_idx, 
				len(train_loader), cost))
        
	model.eval()
    
	cost = compute_epoch_loss(model, train_loader)
	epoch_cost.append(cost)
    
	train_accuracy = compute_accuracy(model, train_loader)
	valid_accuracy = compute_accuracy(model, valid_loader)
    
	print('Epoch: %03d/%03d Train Cost: %.4f' % (
		epoch+1, NUM_EPOCHS, cost))
	print('Train Accuracy: %.3f | Validation Accuracy: %.3f' % (train_accuracy, valid_accuracy))
	print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

'''

plt.plot(range(len(minibatch_cost)), minibatch_cost)
plt.ylabel('Cross Entropy')
plt.xlabel('Minibatch')
plt.show()

plt.plot(range(len(epoch_cost)), epoch_cost)
plt.ylabel('Cross Entropy')
plt.xlabel('Epoch')
plt.show()


print('Test Accuracy: %.2f' % compute_accuracy(model, test_loader))
'''
