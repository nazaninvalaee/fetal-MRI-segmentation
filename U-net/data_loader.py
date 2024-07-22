import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import logging
from tqdm import tqdm
import config

logging.basicConfig(level=logging.INFO)

def get_subject_paths(dataset_path):
    subject_paths = [os.path.join(dataset_path, subject) for subject in os.listdir(dataset_path) if subject.startswith('sub-')]
    return subject_paths

def load_mri_image(image_path):
    try:
        image = nib.load(image_path).get_fdata()
        return image
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return None

def load_feta_dataset(dataset_path):
    subject_paths = get_subject_paths(dataset_path)
    images = []
    masks = []
    
    logging.info("Loading dataset...")
    for subject_path in tqdm(subject_paths, desc="Subjects"):
        subject_id = os.path.basename(subject_path)
        subject_number = int(subject_id.split('-')[1])

        if subject_number <= 40:
            t2w_path = os.path.join(subject_path, f'{subject_id}_rec-mial_T2w.nii.gz')
            dseg_path = os.path.join(subject_path, f'{subject_id}_rec-mial_dseg.nii.gz')
        else:
            t2w_path = os.path.join(subject_path, f'{subject_id}_rec-irtk_T2w.nii.gz')
            dseg_path = os.path.join(subject_path, f'{subject_id}_rec-irtk_dseg.nii.gz')
        
        if os.path.exists(t2w_path) and os.path.exists(dseg_path):
            mri_image = load_mri_image(t2w_path)
            segmentation_mask = load_mri_image(dseg_path)
            
            if mri_image is not None and segmentation_mask is not None:
                mri_image = np.expand_dims(mri_image, axis=-1)  # Expand dims to add channel dimension
                images.append(mri_image)
                masks.append(segmentation_mask)
            else:
                logging.warning(f"Failed to load data for {subject_id}. Skipping.")
        else:
            logging.warning(f"Missing files for {subject_id}")
    
    return np.array(images), np.array(masks)

def preprocess_data(images, masks, num_classes):
    images = images / np.max(images)
    logging.info(f"Shapes before preprocessing: images: {images.shape}, masks: {masks.shape}")
    unique_values = np.unique(masks)
    logging.info(f"Unique values in masks before one-hot encoding: {unique_values}")
    
    masks = np.clip(masks, 0, num_classes - 1)
    
    one_hot_masks = np.zeros((*masks.shape, num_classes), dtype=np.uint8)
    for i in tqdm(range(masks.shape[0]), desc="One-hot encoding masks"):
        one_hot_masks[i] = to_categorical(masks[i], num_classes=num_classes)
    
    logging.info(f"Shapes after preprocessing: images: {images.shape}, masks: {one_hot_masks.shape}")
    
    return images, one_hot_masks

def get_data_splits():
    logging.info("Loading and preprocessing data...")
    images, masks = load_feta_dataset(config.DATA_PATH)
    images, masks = preprocess_data(images, masks, config.NUM_CLASSES)
    logging.info("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val
