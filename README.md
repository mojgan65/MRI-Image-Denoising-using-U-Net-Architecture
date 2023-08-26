# MRI Image Denoising with U-Net

This repository provides an implementation of a denoising method for MRI images using the U-Net deep learning architecture. Given a set of noisy MRI images, the U-Net model is trained to output the denoised versions.

## Dependencies

- `matplotlib`
- `keras`
- `sklearn`
- `skimage`
- `cv2`
- `numpy`
- `tensorflow`
- `glob`
- `os`

To install the required packages, use:

```bash
pip install matplotlib keras scikit-learn scikit-image opencv-python-headless numpy tensorflow
```

## Dataset

Ensure the dataset is situated in the `Dataset_BUSI_with_GT` directory. If your dataset is located elsewhere, adjust the `image_paths` variable accordingly.

## Key Features

1. **Image Preprocessing**: MRI images are loaded, resized to 128x128, and normalized.
2. **Speckle Noise Addition**: Simulates the type of noise typically seen in MRI scans.
3. **U-Net Model**: A convolutional neural network designed specifically for image denoising and segmentation tasks.
4. **Training & Evaluation**: Train the model using noisy images and evaluate its performance on a test set.
5. **Visualization**: Compare original, noisy, and denoised MRI images side by side.


## Results

After executing the script:

- The first few original, noisy, and denoised MRI images are displayed.
- Training history for loss, PSNR, and SSIM across epochs is plotted.

## Contributing

Contributions, issues, and feature requests are welcome!


## Acknowledgements

- MRI dataset source (https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).
