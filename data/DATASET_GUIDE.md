# Dataset Setup Guide

This document provides instructions for obtaining and preparing datasets for the snake venom classifier.

## Dataset Requirements

The classifier requires a dataset of snake images labeled as:
- **Venomous**: Snakes that are venomous
- **Non-venomous**: Snakes that are not venomous

## Recommended Datasets

### 1. Kaggle Datasets

Several snake datasets are available on Kaggle:

**Option A: Search for "snake species" or "snake classification"**
1. Visit [Kaggle Datasets](https://www.kaggle.com/datasets)
2. Search for "snake species" or "snake classification"
3. Download a suitable dataset
4. Extract and organize according to the structure below

**Option B: Use iNaturalist or other sources**
1. Download snake images from [iNaturalist](https://www.inaturalist.org/)
2. Filter for venomous and non-venomous species
3. Organize according to the structure below

### 2. Creating Your Own Dataset

You can also create a custom dataset by:
1. Collecting snake images from various sources
2. Ensuring you have permission to use the images
3. Manually labeling them as venomous or non-venomous
4. Organizing them in the required structure

## Required Directory Structure

Organize your dataset in the following structure:

```
data/raw/
├── venomous/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   └── ...
└── non_venomous/
    ├── image1.jpg
    ├── image2.jpg
    ├── image3.jpg
    └── ...
```

## Dataset Guidelines

### Image Quality
- **Resolution**: At least 128x128 pixels (higher is better)
- **Format**: JPG, PNG, or BMP
- **Quality**: Clear, well-lit images with visible snake features
- **Variety**: Include different angles, lighting conditions, and backgrounds

### Dataset Size
- **Minimum**: 50 images per class (100 total)
- **Recommended**: 200+ images per class (400+ total)
- **Ideal**: 1000+ images per class for better performance

### Label Accuracy
- Ensure images are correctly labeled
- Remove ambiguous or unclear images
- Verify venomous/non-venomous classification with reliable sources

## Example: Download from Kaggle

1. **Create a Kaggle account** at [kaggle.com](https://www.kaggle.com/)

2. **Install Kaggle CLI** (optional):
   ```bash
   pip install kaggle
   ```

3. **Set up API credentials**:
   - Go to Kaggle Account Settings
   - Create new API token
   - Download `kaggle.json`
   - Place in `~/.kaggle/` directory

4. **Download a dataset** (example):
   ```bash
   # Search for datasets
   kaggle datasets list -s "snake"
   
   # Download a specific dataset (replace with actual dataset name)
   kaggle datasets download -d <username>/<dataset-name>
   
   # Extract the dataset
   unzip <dataset-name>.zip -d data/raw/
   ```

5. **Organize the data** according to the required structure

## Data Augmentation (Optional)

For better model performance, you can augment your dataset with:
- Rotations (±15 degrees)
- Horizontal flips
- Brightness adjustments
- Contrast adjustments
- Small crops and zooms

This can be done during training or as a preprocessing step.

## Important Notes

- **Do NOT commit datasets** to the repository
- The `.gitignore` file already excludes `data/raw/*` and `data/processed/*`
- Keep your dataset in the local `data/` directory only
- Share dataset links instead of actual files if collaborating

## Troubleshooting

### Issue: No images found
- Check that images are in the correct subdirectories
- Verify image file extensions (.jpg, .jpeg, .png, .bmp)
- Ensure read permissions on the files

### Issue: Unbalanced dataset
- Try to have similar numbers of venomous and non-venomous images
- If imbalanced, consider data augmentation or weighted sampling

### Issue: Poor model performance
- Increase dataset size
- Improve image quality
- Verify label accuracy
- Try different feature extraction parameters

## Resources

- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [iNaturalist](https://www.inaturalist.org/)
- [Reptile Database](http://www.reptile-database.org/)
- [Snake Identification Guide](https://www.snakeidentification.com/)
