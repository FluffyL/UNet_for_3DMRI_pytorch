import SimpleITK as sitk
import nibabel as nib
import numpy as np
import os
import glob

class Config(object):
	shrink = 1
	new_spacing = (1.0,1.0,1.0)
	nbr_bins = 256
	data_path = '/home/lly/Desktop/intern_Xidian/WM&GM_segmentation/code/UNet_Pytorch/LPBA40/native_space'
	preprocessed_path = '/home/lly/Desktop/intern_Xidian/WM&GM_segmentation/code/UNet_Pytorch/LPBA40/processed'

def img_show(img):
	for i in range(img.shape[0]):
		io.imshow(img[i,:,:],cmap='gray')
		print(i)
		io.show()

def calculate_origin_offset(new_spacing, old_spacing):
	return np.subtract(new_spacing, old_spacing)/2


def sitk_resample_to_spacing(image, new_spacing, interpolator=sitk.sitkLinear, default_value=0.):
	zoom_factor = np.divide(image.GetSpacing(), new_spacing)
	new_size = np.asarray(np.ceil(np.round(np.multiply(zoom_factor, image.GetSize()), decimals=5)), dtype=np.int16)
	offset = calculate_origin_offset(new_spacing, image.GetSpacing())
	reference_image = sitk_new_blank_image(size=new_size, spacing=new_spacing, direction=image.GetDirection(),origin=image.GetOrigin() + offset, default_value=default_value)
	
	return sitk_resample_to_image(image, reference_image, interpolator=interpolator, default_value=default_value)


def sitk_resample_to_image(image, reference_image, default_value=0., interpolator=sitk.sitkLinear, transform=None,output_pixel_type=None):
	if transform is None:
		transform = sitk.Transform()
		transform.SetIdentity()
	if output_pixel_type is None:
		output_pixel_type = image.GetPixelID()
	resample_filter = sitk.ResampleImageFilter()
	resample_filter.SetInterpolator(interpolator)
	resample_filter.SetTransform(transform)
	resample_filter.SetOutputPixelType(output_pixel_type)
	resample_filter.SetDefaultPixelValue(default_value)
	resample_filter.SetReferenceImage(reference_image)
	return resample_filter.Execute(image)


def sitk_new_blank_image(size, spacing, direction, origin, default_value=0.):
	image = sitk.GetImageFromArray(np.ones(size, dtype=np.float).T * default_value)
	image.SetSpacing(spacing)
	image.SetDirection(direction)
	image.SetOrigin(origin)
	return image

def biasFieldCorrection(img,shrink):
	numberFittingLevels = 4
	inputImage = img
	maskImage = sitk.OtsuThreshold(inputImage,0,1,200)
	inputImage = sitk.Shrink(inputImage,[int(shrink)]* inputImage.GetDimension())
	maskImage = sitk.Shrink(maskImage, [int(shrink)] * inputImage.GetDimension() )
	inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
	corrector = sitk.N4BiasFieldCorrectionImageFilter();
	output = corrector.Execute(inputImage, maskImage)

	return output

def histeq(img,nbr_bins):
	img = sitk.GetArrayFromImage(img)
	img_max = np.max(img)
	img[img < img_max*0.05] = 0
	imhist, bins = np.histogram(img.flatten(), nbr_bins, normed = True)
	cdf = imhist.cumsum() # cumulative distribution function
	cdf = 255 * cdf /cdf[-1] 
	result = np.interp(img.flatten(),bins[:-1],cdf)
    
	return sitk.GetImageFromArray(result.reshape(img.shape))

if __name__=='__main__':
	config = Config()

	sto = os.listdir(config.data_path)
	for folder in sto:
		print('######################################################')
		print('processing subject ' + folder + ' ...')
		path = os.path.join(config.data_path, folder, folder + '.native.mri.img')
		f = nib.load(path)
		affine = f.affine
		data = f.get_data()
		save_data = nib.Nifti1Image(data,affine)
		nii_path = os.path.join(config.data_path, folder, folder + '.nii.gz')
		save_data.to_filename(nii_path)
		data = sitk.ReadImage(nii_path,sitk.sitkFloat32)
		print('Doing bias field correction...')
		correct = biasFieldCorrection(data,config.shrink)
		print('Doing histogram equalization...')
		histeqed = histeq(data,config.nbr_bins)
		print('Doing resample...')
		resample = sitk_resample_to_spacing(histeqed, config.new_spacing, interpolator=sitk.sitkLinear)
		
		if not os.path.exists(config.preprocessed_path):
        		os.makedirs(config.preprocessed_path)
		sitk.WriteImage(resample,os.path.join(config.preprocessed_path,folder + '.processed.nii.gz'))
		
