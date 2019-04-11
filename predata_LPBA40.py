import os
import sys
import gzip
import shutil
import nibabel as nib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

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


