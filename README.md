This project is based on the method proposed in the following paper, which introduces an Attention Gate U-Net architecture for rainforest detection tasks:
https://www.sciencedirect.com/science/article/pii/S0303243422000113

In this coursework, we adapt the Attention Gate U-Net to address **vegetation detection in Chinese urban and rural regions**.

The project uses the **LoveDA dataset**, a publicly available land-cover dataset containing annotated satellite imagery from various cities in China:
https://zenodo.org/records/5706578

---

## Models Implemented

This repository contains three main training scripts for different variants of Attention Gate U-Net:

- **train_AttUnet_scratch.py**  
  Training an Attention Gate U-Net model **from scratch**.

- **train_AttUnet_transfer.py**  
  Applying **transfer learning** on the Attention Gate U-Net architecture as described in the original paper.

- **train_opti_AttUnet.py**  
  An **optimized variant** of Attention Gate U-Net, specifically improved for the LoveDA dataset and the Chinese vegetation detection task.

---

## Additional Scripts

- **reproduce_baseline.py**  
  Reproduces the baseline method described in the original paper.

- **preprocess-loveda.py**  
  Preprocesses the LoveDA dataset to make it compatible with the models used in this project.

- **train_unet.py**  
  A minimal U-Net implementation
