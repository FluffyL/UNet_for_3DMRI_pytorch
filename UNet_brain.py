import os
import time
import nibabel as nib
import gzip
import shutil
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt



RANDOM_SEED = 123
dataset_dir = '/home/lly/Desktop/intern_Xidian/WM&GM_segmentation/dataset/LPBA40/LPBA40/native_space/'


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def load_dataset(dataset_dir, subject_id, mode = 1):

	subject_name = 'S%d' % subject_id
	if subject_id<10:
		T1_name = 'S0%d' % subject_id
	else:
		T1_name = 'S%d' % subject_id
	
	#get T1 images
	f1 = os.path.join(dataset_dir, subject_name+'/'+T1_name+'.native.mri.hdr')
	f1gz = os.path.join(dataset_dir, subject_name+'/'+T1_name+'.native.mri.img.gz')
	f1img = os.path.join(dataset_dir, subject_name+'/'+T1_name+'.native.mri.img')

	img_T1 = nib.load(f1)
	with gzip.open(f1gz, 'rb') as f_in:
		with open(f1img, 'wb') as f_out:
        		shutil.copyfileobj(f_in, f_out)
	inputs_T1 = img_T1.get_data()

	#get labels for training data
	if mode == 1:
		fl = os.path.join(dataset_dir, subject_name+'/tissue/'+T1_name+'.native.tissue.hdr')
		flgz = os.path.join(dataset_dir, subject_name+'/tissue/'+T1_name+'.native.tissue.img.gz')
		flimg = os.path.join(dataset_dir, subject_name+'/tissue/'+T1_name+'.native.tissue.img')
		with gzip.open(flgz, 'rb') as f_in:
			with open(flimg, 'wb') as f_out:
        			shutil.copyfileobj(f_in, f_out)
		img_label = nib.load(fl)
		inputs_label = img_label.get_data()
		plt.show(inputs_label)

	#for validation data
	else:
		inputs_label = None
	return [inputs_T1, inputs_label]


def convert_labels(labels):
	
	D, H, W, C = labels.shape
	labels_hist = np.reshape(labels,[D*H*W*C,1])
	plt.xlim([np.min(labels_hist)-0.5, np.max(labels_hist)+0.5])
 
	plt.hist(labels_hist, bins=4, alpha=0)
	plt.title('labels hist')
	plt.xlabel('label')
	plt.ylabel('frequent')
 
	plt.show()
	for d in range(D):
		for h in range(H):
			for w in range(W):
				for c in range(C):
					if labels[d,h,w,c] == 2:
						labels[d,h,w,0] = 1
					elif labels[d,h,w,c] == 1:
						labels[d,h,w,0] = 1
					elif labels[d,h,w,c] == 3:
						labels[d,h,w,0] = 3
	return labels


def prepare_validation(img, patch_size, overlap_stepsize):

	patch_ids = []

	D, H, W, _ = cutted_image.shape

	drange = list(range(0, D-patch_size+1, overlap_stepsize))
	hrange = list(range(0, H-patch_size+1, overlap_stepsize))
	wrange = list(range(0, W-patch_size+1, overlap_stepsize))

	if (D-patch_size) % overlap_stepsize != 0:
		drange.append(D-patch_size)
	if (H-patch_size) % overlap_stepsize != 0:
		hrange.append(H-patch_size)
	if (W-patch_size) % overlap_stepsize != 0:
		wrange.append(W-patch_size)

	for d in drange:
		for h in hrange:
			for w in wrange:
				patch_ids.append((d, h, w))

	return patch_ids


class MRIDataset(Dataset):
	"""loading MRI data"""
	def __init__(self, dataset_dir, num_s, num_e, img_transform=None, labels_transform=None):
		global T1,labels
		self.dataset_dir = dataset_dir
		self.T1_index = np.array(range(num_s, num_e + 1))
		#self.labels = convert_labels(labels)
		self.img_transform = img_transform
		self.labels_transform = labels_transform

	def __getitem__(self, index):

		[T1,label] = load_dataset(self.dataset_dir,self.T1_index[index])
		#if read .hdr files
		T1 = T1[:,:,:,0]
		label = label[:,:,:,0]

		if self.img_transform is not None:
			T1 = self.img_transform(T1)
		if self.labels_transform is not None:
			label = self.labels_transform(label)

		return T1, label

	def __len__(self):
		return len(self.T1_index)



img_transform = transforms.Compose([
	#transforms.RandomAffine(degrees=(-20, 20), translate=(0.15, 0.15)),
	transforms.ToTensor(), 
	])

labels_transform = transforms.Compose([
	#transforms.RandomAffine(degrees=(-20, 20), translate=(0.15, 0.15)),
	transforms.ToTensor(), 
	])


BATCH_SIZE = 2
train_dataset = MRIDataset(dataset_dir = dataset_dir,
			num_s = 1,
			num_e = 30,
			img_transform = img_transform,
			labels_transform = img_transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)

valid_dataset = MRIDataset(dataset_dir = dataset_dir,
			num_s = 31,
			num_e = 40,
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
	print(train_loader)

	for batch_idx, (T1, labels) in enumerate(train_loader):

		print('Epoch:', epoch+1, end='')
		print(' | Batch index:', batch_idx, end='')
		print(' | Batch size:', labels.size()[0])
		
		T1 = T1.view(-1, 1,256,256,124).to(DEVICE)
		labels = labels.view(-1, 1,256,256,124).to(DEVICE)
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
		self.outc = outconv(32, n_classes)

	def forward(self, x):
		print(x.shape)
		x1 = self.inc(x.float())
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.center(x4)
		x4 = self.up4(x5)
		x3 = self.up3(x4)
		x2 = self.up2(x3)
		x1 = self.up1(x2)
		x = self.outc(x1)
		return F.sigmoid(x)

#################################
### Model Initialization
#################################
torch.manual_seed(RANDOM_SEED)

model = UNet(n_channels=1, n_classes=1)
model = model.float()

optimizer = torch.optim.SGD(model.parameters(), lr=0.5,momentum=0.9)

#################################
### Training
#################################

def compute_epoch_loss(model, data_loader):
	curr_loss, num_examples = 0., 0
	with torch.no_grad():
		for T1, labels in data_loader:
			print(T1.shape)
			T1 = T1.view(-1,1,256,256,124).to(DEVICE)
			labels = labels.view(-1,1,256,256,124).to(DEVICE)
			print(T1.shape)
			logits, probas = model(T1)
			loss = F.cross_entropy(logits, labels, reduction='sum')
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
			print(T1.shape)
			T1 = T1.view(-1,1,256,256,124).to(DEVICE)
			print(T1.shape)
			labels = labels.view(-1,1,256,256,124).to(DEVICE)
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
	for batch_idx, (T1, labels) in enumerate(train_loader):
		#features = features.view(-1,28*28).to(DEVICE)
		T1 = T1.view(-1,1,256,256,124).to(DEVICE)
		labels = labels.view(-1,1,256,256,124).to(DEVICE)
            
		### FORWARD AND BACK PROP
		logits, probas = model(T1)
        
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

