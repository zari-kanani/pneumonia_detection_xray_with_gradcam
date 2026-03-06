# Pneumonia Detection from Chest X-Rays

This project extends the Kaggle notebook [Pneumonia Detection with ResNet & PyTorch](https://www.kaggle.com/code/denvermagtibay/pneumonia-detection-with-resnet-pytorch) by Denver Magtibay with several improvements to the training pipeline and the addition of model interpretability via Grad-CAM++.

## Modifications

- Backbone replaced from ResNet-18 to EfficientNet-B0
- Preprocessing refined: center cropping to remove border artifacts, grayscale-to-RGB conversion
- Validation set expanded from 16 images to a 15% split of the training data
- Layer-specific learning rates for backbone and classifier head
- Early stopping monitored on validation loss instead of accuracy
- Grad-CAM++ applied to visualize model attention on chest X-rays

## Results

88% accuracy on the test set with a pneumonia recall of 0.99 — comparable to the baseline. The main contribution is the Grad-CAM++ interpretability component, which confirms the model attends to the lung fields when making predictions.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — pediatric chest radiographs from Guangzhou Women and Children's Medical Center, labeled as NORMAL or PNEUMONIA.

## How to Run

Open the notebook in Google Colab and upload your `kaggle.json` credentials file when prompted. GPU runtime recommended.
