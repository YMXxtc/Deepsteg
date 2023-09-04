#train the network
python deep_steg.py -g 0,1,2,3

#generate stego images and according costs
python stegan_cuda.py --netG Res/netG_epoch_deep_steg_72.pth -g 0 --config deep_steg

#STC embedding
matlab -nosplash -nodesktop -r "embedding /data/ymx/ymx/deep_steg/deep_steg;exit" 
