import nibabel as nb
import skimage
from nibabel.viewers import OrthoSlicer3D

img = nb.load('D:/copy_model/mask_nii/D3206741-gt.nii.gz') #读取nii格式文件
img_affine = img.affine
data = img.get_fdata()

header = img.header
print(header)
print(data.shape)
print(data.shape[2])

### Get original space
width, height, channel = img.dataobj.shape
ori_space = [width,height,channel]


def resample(data,ori_space, header, spacing):
    ### Calculate new space
    new_width = round(ori_space[0] * header['pixdim'][1] / spacing[0])
    new_height = round(ori_space[1] * header['pixdim'][2] / spacing[1])
    new_channel = round(ori_space[2] * header['pixdim'][3] / spacing[2])
    new_space = [new_width, new_height, new_channel]

    data_resampled = skimage.transform.resize(data,new_space,order=1, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=True, anti_aliasing_sigma=None)
    return data_resampled

# Resample to have 1.0 spacing in all axes
spacing = [1.0, 1.0, 1.0]
data_resampled = resample(data,ori_space, header, spacing)
print(data_resampled.shape)


