# CTscan_MedicalImageAnalysis_LungTumorSegmentation

Tumor Segmentation in Lung Dataset: Methodology and Libraries

When handling raw CT scan data, we will need to pre-process them such that they can be conveniently used in neural network task. 
Below are the pre-processing steps to follow for CT images.

Pre-processing Steps:

Pre-processing Steps:

1. Normalization:
- CT images have a fixed intensity range from -1000 to 3071.
- Normalize by dividing by 3071 to standardize the pixel values to a range between 0 and 1.
2. Cropping for Focus:
- Crop out the lower abdomen by skipping the first 30 slices (or more) to focus on the lung tumors and reduce complexity.
3. 2D Slice-Level Processing:
- Process data at the slice level (2D) rather than at the subject level (3D) to reduce computational cost and store the preprocessed data as 2D slices.
4. Resizing:
- Resize the slices and masks to (256, 256) for consistency in input size.
- Use cv2.INTER_NEAREST interpolation for the masks during resizing to preserve the integrity of the segmentation labels.



After pre-processing is complete, follow following steps for the task of lung tumor segmentation from CT images:

Step 1: Import the Libraries
- PyTorch and TorchVision for model building, data manipulation, and training.
- PyTorch Lightning for simplifying the training loop and integrating various features like GPU support, logging, and checkpoints.
- NumPy and Matplotlib for numerical and visualization tasks.
- Albumentations for image augmentation, including affine transformations.
- Kaggle API for importing data from Kaggle.
- Scikit-learn for utility functions like metrics.

Step 2: Create Dataset and Import Data
  - 2(A): Import dataset from Kaggle:
    - Use the Kaggle API to download the lung tumor segmentation dataset (or other relevant datasets).
  - 2(B): Define path to training and validation data:
    - Define the paths where the training and validation image and mask data are stored (e.g., /data/train_images/, /data/val_images/).
  - 2(C): Define augmentation:
    - Apply affine augmentations (such as rotations, scaling, and translations) using Albumentations to further improve model generalization. These augmentations ensure better robustness and performance on diverse variations in the data.
  - 2(D): Create dataset:
    - Create a custom PyTorch dataset class that loads the images and corresponding masks, applies the necessary augmentations, and returns them as tensors.
  - 2(E): Pass training and validation data to dataset class:
    - Instantiate the dataset class for both training and validation data, ensuring proper data format and augmentation.


Step 3: Implement Oversampling to Handle Strong Class Imbalance
3(A): Create a list with only the class labels:
Create a list containing the class labels (e.g., 0 for background and 1 for the tumor).
3(B): Calculate weight of each class:
Calculate the frequency of each class in the dataset (e.g., how many pixels belong to the tumor class versus background).
3(C): Assign weight to each class and define new 'weighted list':
Based on the class frequency, assign a weight to each sample. For example, give higher weights to underrepresented classes (tumor regions).
3(D): Create a sampler:
Use WeightedRandomSampler from PyTorch to sample from the dataset based on the class weights, ensuring that each class is adequately represented in each mini-batch during training.
3(E): Creating Train and Validation DataLoaders with Custom Sampling:
Use the custom sampler to create DataLoader instances for the training and validation datasets, ensuring that the samples are balanced according to their class weights.


Step 4: Define Full Segmentation Model and Begin Training
4(A): Define Convolution block:
Define a DoubleConv block (a custom block containing two convolution layers followed by ReLU activations), which is used as the basic building block for the encoder and decoder in the UNet architecture.
4(B): Define UNET:
Define the UNet model, which consists of an encoder (down-convolution) and decoder (up-convolution) with skip connections between corresponding encoder and decoder layers.
4(C): Create a segmentation model:
Integrate the DoubleConv blocks into the full UNet architecture, with upsampling layers and a final 1x1 convolution to output a binary segmentation mask.
4(D): Instantiate the model:
Create an instance of the UNet model, specifying the number of input channels (1 for grayscale images) and output channels (1 for binary segmentation).
4(E): Create a checkpoint callback:
Define a callback that saves the model checkpoint with the best validation performance (using ModelCheckpoint from PyTorch Lightning).
4(F): Define the trainer:
Use PyTorch Lightning’s Trainer class to define the trainer, including specifying the optimizer (Adam), loss function (Binary Cross Entropy), learning rate, and other training parameters.
4(G): Train the model:
Train the model using the defined trainer, with the appropriate data loaders for training and validation, and the model checkpoint callback.


Step 5: Evaluation of the Training Results
5(A): Load the latest checkpoint:
Load the best model checkpoint (using load_from_checkpoint) to evaluate its performance on the validation dataset.
5(B): Evaluate the model on the validation dataset:
Evaluate the model by running inference on the validation set and comparing the predicted segmentation masks to the ground truth masks.
5(C): Define DiceScore and compute it:
Define a Dice Similarity Coefficient class to compute the Dice Score, which measures the overlap between the predicted and true segmentation masks. Compute the Dice score to evaluate the model’s segmentation accuracy.
5(D): Evaluate loss as a function of epoch:
Track and plot the training and validation loss across epochs to better understand model performance over time.


Step 6: Testing on Test Image
6(A): Test the model on test images:
Apply the trained model to test images for final evaluation, using the same preprocessing steps (normalization, resizing, etc.) to generate predictions and assess the final model performance on unseen data.
