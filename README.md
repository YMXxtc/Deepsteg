# Deepsteg

This is the code implementation of paper “A New Deep Learning-Based Steganography: Integrating Residual Characteristics for Enhanced Image Security”. All source code prefer to running under Python3.7 with matlab interface.

## to run code with bash
bash main01.sh

## to run code respectively
### train the network
python deep_steg.py -g 0,1,2,3

### generate stego images and according costs
python stegan_cuda.py --netG Res/netG_epoch_deep_steg_72.pth -g 0 --config deep_steg

### STC embedding
matlab -nosplash -nodesktop -r "embedding /data/ymx/ymx/deep_steg/deep_steg_unet;exit" 
