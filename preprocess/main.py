import os
import numpy as np
import scipy.ndimage as ndimage
import SimpleITK as sitk

def to255(img):     # Normalize to 0-255
    min_val = img.min()
    max_val = img.max()
    img = (img - min_val) / (max_val - min_val + 1e-5)  # Image normalization
    img = img * 255  # Scale to 255
    return img

def resample_image(image, target_size=(256, 256, 20), is_label=False):
    """
    Resample the image to target size using SimpleITK
    :param image: Input SimpleITK image
    :param target_size: Desired output size (x,y,z)
    :param is_label: If True, use nearest neighbor interpolation
    :return: Resampled image
    """
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    
    new_spacing = [
        original_spacing[0] * (original_size[0] / target_size[0]),
        original_spacing[1] * (original_size[1] / target_size[1]),
        original_spacing[2] * (original_size[2] / target_size[2])
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform())
    
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
    
    return resampler.Execute(image)

def preprocess_nii(filepath, target_size=(256, 256, 20)):
    # Read the image
    image = sitk.ReadImage(filepath)
    
    # Resample to target size
    image = resample_image(image, target_size)
    
    # Convert to numpy array for additional processing if needed
    img_array = sitk.GetArrayFromImage(image)
    
    # Normalize if needed (optional)
    img_array = to255(img_array)
    
    # Convert back to SimpleITK image
    processed_image = sitk.GetImageFromArray(img_array)
    processed_image.CopyInformation(image)  # Copy metadata
    
    return processed_image

rootpath = "/data3/wangchangmiao/jinhui/DATA/ICH_prognosis/origin_image"
Resultpath = '/data3/wangchangmiao/jinhui/DATA/ICH_prognosis/processed_image'

# Create output directory if it doesn't exist
os.makedirs(Resultpath, exist_ok=True)

# Process each file
for patient in os.listdir(rootpath):
    print(f"Processing {patient}...")
    try:
        image = preprocess_nii(os.path.join(rootpath, patient))
        save_path = os.path.join(Resultpath, patient)
        sitk.WriteImage(image, save_path)
    except Exception as e:
        print(f"Error processing {patient}: {str(e)}")