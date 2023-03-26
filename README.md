# SRGAN_OCT
Using Super Resolution GAN to restore blurred/low resolution images of retina OCT scans and compare training results of original resolution and generated resolution images.

[Low Resolution (Left) vs High Resolution (Center) vs Generated Super Resolution (Right)](images/LRvsHRvsSR.jpg)

## Requirements
The following packages are required to run the code:

- Python (>= 3.6)
- TensorFlow (>= 2.0)
- NumPy
- Matplotlib
- OpenCV

## Setup
1. Clone the repository:
```
git clone https://github.com/stanlygomes/SRGAN_OCT.git
cd SRGAN_OCT
```

2. Download the Retina OCT dataset and extract it to the current directory (should put a folder in your master directory called 'Data'). The dataset can be obtained from [here](https://www.kaggle.com/datasets/paultimothymooney/kermany2018).

3. Run the 'SRGAN_TrainAndCompare.ipynb' notebook in Jupyter. The notebook contains code for loading and preprocessing dataset for SRGAN training (such as resizing the images for low and high resolution versions), building the SRGAN model from scratch, and training it on the dataset.

*Note: For the machine used for running the notebook, the directory for the repository location had to be stated for any directory access for some reason. This seems to only be an issue with my environment, so change the data directories for your own case based on the error debug statements.*

## Credits
The SRGAN model implementation is based on the paper "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" by Christian Ledig et al.

The Retina OCT dataset is obtained from [here](https://www.kaggle.com/datasets/paultimothymooney/kermany2018).
