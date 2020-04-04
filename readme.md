# AMMAI HW1 APD FACE VERIFICATION

# Step 1. Obtain Dataset: [CASIA WebFace](https://drive.google.com/file/d/1wJC2aPA4AC0rI-tAL2BFs2M8vfcpX-w6/view?usp=sharing)

## unzip CASIA-WebFace clean version with following commands:

~~~~
unzip casia-maxpy-clean.zip
cd casia-maxpy-clean
zip -F CASIA-maxpy-clean.zip --out CASIA-maxpy-clean_fix.zip
unzip CASIA-maxpy-clean_fix.zip
~~~~

# Step 2. Align Face

## Obtain [face alignment tools](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)

~~~~
# Trainset Alignment
git clone https://github.com/ZhaoJ9014/face.evoLVe.PyTorch.git
cd face.evoLVe.PyTorch/align
python face_align.py -source_root '/media/iis/ssdx16/casia-maxpy-clean/CASIA-maxpy-clean' -dest_root '/media/iis/ssdx16/casia-maxpy-clean/CASIA-maxpy-clean-aligned' -crop_size 224

# Testset Alignment
python testset_converter.py # convert test data folder structure
cd face.evoLVe.PyTorch/align
python face_align.py -source_root '/media/iis/ssdx16/casia-maxpy-clean/A_' -dest_root '/media/iis/ssdx16/casia-maxpy-clean/A-aligned' -crop_size 224
python A2C.py # move cropped files to folder named "C"
~~~~

# Step 3. Train
bash run.sh
