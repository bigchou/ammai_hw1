# Inference
#python main.py --resume path/to/ckpt --arch resnet18 --inference --logfile SoftmaxTest.txt --loss softmax --dist cosine --plotroc --figname Softmax.png --outdir Softmax
#python main.py --resume path/to/ckpt --arch resnet18 --inference --logfile ASoftmaxTest.txt --loss sphereface --dist cosine --plotroc --figname ASoftmax.png --outdir ASoftmax
#python main.py --resume path/to/ckpt --arch resnet18 --inference --logfile TripletTest.txt --loss triplet --l2norm --dist l2 --plotroc --figname Triplet.png --outdir Triplet
# Training
python main.py --logfile SoftmaxTrain.txt --loss softmax --dist cosine --batch 256 --outdir Softmax
python main.py --logfile ASoftmaxTrain.txt --loss sphereface --dist cosine --batch 256 --outdir ASoftmax
python main.py --logfile TripletTrain.txt --loss triplet --l2norm --dist l2 --batch 128 --outdir Triplet
