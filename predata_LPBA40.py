import os
import sys
import gzip
import shutil
import nibabel as nib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

dataset_dir = '/LPBA40/native_space/'
croped_dir = '/LPBA40/train_data/'


def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def load_dataset(croped_dir,index, mode):

	croped_name = str(index)

	if mode == 0:
		f1 = os.path.join(croped_dir,croped_name+'_train_T1.nii.gz')
		fl = os.path.join(croped_dir,croped_name+'_train_label.nii.gz')
		img_T1 = nib.load(f1)
		inputs_T1 = img_T1.get_data()
		img_label = nib.load(fl)
		inputs_label = img_label.get_data()
	elif mode == 1:
		f1 = os.path.join(croped_dir,croped_name+'_test_T1.nii.gz')
		fl = os.path.join(croped_dir,croped_name+'_test_label.nii.gz')
		img_T1 = nib.load(f1)
		inputs_T1 = img_T1.get_data()
		img_label = nib.load(fl)
		inputs_label = img_label.get_data()
	else:
		f1 = os.path.join(croped_dir,croped_name+'_test_T1.nii.gz')
		img_T1 = nib.load(f1)
		inputs_T1 = img_T1.get_data()
		inputs_label = None
	
	return [inputs_T1,inputs_label]
	
	
def crop_data(dataset_dir, croped_dir, subject_id, total_t, mode, patch_size):
	
	subject_name = 'S%d' % subject_id
	if subject_id<10:
		T1_name = 'S0%d' % subject_id
	else:
		T1_name = 'S%d' % subject_id

	if mode == 0:
		t_name = 'train'
	elif mode == 1:
		t_name = 'test'
	
	#get T1 images
	f1 = os.path.join(dataset_dir, subject_name+'/'+T1_name+'.native.mri.hdr')
	f1gz = os.path.join(dataset_dir, subject_name+'/'+T1_name+'.native.mri.img.gz')
	f1img = os.path.join(dataset_dir, subject_name+'/'+T1_name+'.native.mri.img')

	img_T1 = nib.load(f1)
	affine = img_T1.affine
	with gzip.open(f1gz, 'rb') as f_in:
		with open(f1img, 'wb') as f_out:
        		shutil.copyfileobj(f_in, f_out)
	inputs_T1 = img_T1.get_data()

	patch_ids = prepare_validation(inputs_T1, )
	print('##################################################')
	print('croping subject' + str(subject_id) +' image...')
	
	if mode == 0 or mode == 1:

		#get labels for training data
		fl = os.path.join(dataset_dir, subject_name+'/tissue/'+T1_name+'.native.tissue.hdr')
		flgz = os.path.join(dataset_dir, subject_name+'/tissue/'+T1_name+'.native.tissue.img.gz')
		flimg = os.path.join(dataset_dir, subject_name+'/tissue/'+T1_name+'.native.tissue.img')
		with gzip.open(flgz, 'rb') as f_in:
			with open(flimg, 'wb') as f_out:
	        		shutil.copyfileobj(f_in, f_out)
		img_label = nib.load(fl)
		inputs_label = img_label.get_data()
	
		for i in range(len(patch_ids)):

			(d, h, w) = patch_ids[i]
			croped_name = str(total_t+i+1)
			croped_T1 = inputs_T1[d:d+patch_size, h:h+patch_size, w:w+patch_size, :]
			croped_label = inputs_label[d:d+patch_size, h:h+patch_size, w:w+patch_size, :]
			save_T1 = nib.Nifti1Image(croped_T1, affine)
			save_label = nib.Nifti1Image(croped_label, affine)
			save_T1.to_filename(os.path.join(croped_dir,croped_name+'_'+t_name+'_T1.nii.gz'))
			save_label.to_filename(os.path.join(croped_dir,croped_name+'_'+t_name+'_label.nii.gz'))
		print('croping finished.Got ' + str(i+1) + ' patches from subject '+ str(subject_id) +'...')
		print('Total patches: ' + croped_name)

	else: 
		for i in range(len(patch_ids)):
			(d, h, w) = patch_ids[i]
			croped_name = str(total_t+i+1)
			croped_T1 = inputs_T1[d:d+patch_size, h:h+patch_size, w:w+patch_size, :]
			save_T1 = nib.Nifti1Image(croped_T1, affine)
			save_T1.to_filename(save_T1, os.path.join(croped_dir,croped_name+'_predict_T1.nii.gz'))

		#******************havev't finished yet,for prediction data***************#
		
	return len(patch_ids)

def convert_one_hot(labels):
	
	A,B,C,D,E = labels.shape
	one_hot = np.zeros([A,3*B,C,D,E])
 
	for a in range(A):
		for c in range(C):
			for d in range(D):
				for e in range(E):
					if labels[a,0,c,d,e] == 1:
						one_hot[a,0,c,d,e] = 1
					elif labels[a,0,c,d,e] == 2:
						one_hot[a,1,c,d,e] = 1
					elif labels[a,0,c,d,e] == 3:
						one_hot[a,2,c,d,e] = 1
	return one_hot


def prepare_validation(img, patch_size = 32, overlap = 4):

	patch_ids = []

	D, H, W, _ = img.shape
	
	drange = list(range(0, D, (patch_size-overlap)))
	hrange = list(range(0, H, (patch_size-overlap)))
	wrange = list(range(0, W, (patch_size-overlap)))

	if (D-drange[len(drange)-1]) < patch_size-1:
		drange[len(drange)-1] = D-patch_size
	if (H-hrange[len(hrange)-1]) < patch_size-1:
		hrange[len(hrange)-1] = H-patch_size
	if (W-wrange[len(wrange)-1]) < patch_size-1:
		wrange[len(wrange)-1] = W-patch_size
	'''
	d = index // (len(hrange) * len(wrange))
	h = (index % (len(hrange) * len(wrange))) // len(wrange)
	w = (index % (len(hrange) * len(wrange))) % len(wrange)
	'''
	for d in drange:
		for h in hrange:
			for w in wrange:
				patch_ids.append((d, h, w))

	return patch_ids

