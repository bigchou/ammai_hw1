python test.py --resume softmax_resnet18_withoutl2norm.pth --arch resnet18 --loss softmax --dist cosine --plotroc --figname C_224.png --outdir SoftmaxResNet18_cosdist --logfile C_224.txt --testdb C_224
python test.py --resume softmax_resnet18_withoutl2norm.pth --arch resnet18 --loss softmax --dist cosine --plotroc --figname C.png --outdir SoftmaxResNet18_cosdist --logfile C.txt --testdb C
python test.py --resume softmax_resnet18_withoutl2norm.pth --arch resnet18 --loss softmax --dist cosine --plotroc --figname lfw-aligned.png --outdir SoftmaxResNet18_cosdist --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume softmax_resnet18_withoutl2norm.pth --arch resnet18 --loss softmax --dist cosine --l2norm --plotroc --figname C_224.png --outdir SoftmaxResNet18_cosdist_l2norm --logfile C_224.txt --testdb C_224
python test.py --resume softmax_resnet18_withoutl2norm.pth --arch resnet18 --loss softmax --dist cosine --l2norm --plotroc --figname C.png --outdir SoftmaxResNet18_cosdist_l2norm --logfile C.txt --testdb C
python test.py --resume softmax_resnet18_withoutl2norm.pth --arch resnet18 --loss softmax --dist cosine --l2norm --plotroc --figname lfw-aligned.png --outdir SoftmaxResNet18_cosdist_l2norm --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume softmax_resnet18_withoutl2norm.pth --arch resnet18 --loss softmax --dist l2  --plotroc --figname C_224.png --outdir SoftmaxResNet18_l2dist --logfile C_224.txt --testdb C_224
python test.py --resume softmax_resnet18_withoutl2norm.pth --arch resnet18 --loss softmax --dist l2  --plotroc --figname C.png --outdir SoftmaxResNet18_l2dist --logfile C.txt --testdb C
python test.py --resume softmax_resnet18_withoutl2norm.pth --arch resnet18 --loss softmax --dist l2  --plotroc --figname lfw-aligned.png --outdir SoftmaxResNet18_l2dist --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume softmax_resnet18_withoutl2norm.pth --arch resnet18 --loss softmax --dist l2 --l2norm --plotroc --figname C_224.png --outdir SoftmaxResNet18_l2dist_l2norm --logfile C_224.txt --testdb C_224
python test.py --resume softmax_resnet18_withoutl2norm.pth --arch resnet18 --loss softmax --dist l2 --l2norm --plotroc --figname C.png --outdir SoftmaxResNet18_l2dist_l2norm --logfile C.txt --testdb C
python test.py --resume softmax_resnet18_withoutl2norm.pth --arch resnet18 --loss softmax --dist l2 --l2norm --plotroc --figname lfw-aligned.png --outdir SoftmaxResNet18_l2dist_l2norm --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume softmax_resnet34_withoutl2norm.pth --arch resnet34 --loss softmax --dist cosine --plotroc --figname C_224.png --outdir SoftmaxResNet34_cosdist --logfile C_224.txt --testdb C_224
python test.py --resume softmax_resnet34_withoutl2norm.pth --arch resnet34 --loss softmax --dist cosine --plotroc --figname C.png --outdir SoftmaxResNet34_cosdist --logfile C.txt --testdb C
python test.py --resume softmax_resnet34_withoutl2norm.pth --arch resnet34 --loss softmax --dist cosine --plotroc --figname lfw-aligned.png --outdir SoftmaxResNet34_cosdist --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume softmax_resnet34_withoutl2norm.pth --arch resnet34 --loss softmax --dist cosine --l2norm --plotroc --figname C_224.png --outdir SoftmaxResNet34_cosdist_l2norm --logfile C_224.txt --testdb C_224
python test.py --resume softmax_resnet34_withoutl2norm.pth --arch resnet34 --loss softmax --dist cosine --l2norm --plotroc --figname C.png --outdir SoftmaxResNet34_cosdist_l2norm --logfile C.txt --testdb C
python test.py --resume softmax_resnet34_withoutl2norm.pth --arch resnet34 --loss softmax --dist cosine --l2norm --plotroc --figname lfw-aligned.png --outdir SoftmaxResNet34_cosdist_l2norm --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume softmax_resnet34_withoutl2norm.pth --arch resnet34 --loss softmax --dist l2  --plotroc --figname C_224.png --outdir SoftmaxResNet34_l2dist --logfile C_224.txt --testdb C_224
python test.py --resume softmax_resnet34_withoutl2norm.pth --arch resnet34 --loss softmax --dist l2  --plotroc --figname C.png --outdir SoftmaxResNet34_l2dist --logfile C.txt --testdb C
python test.py --resume softmax_resnet34_withoutl2norm.pth --arch resnet34 --loss softmax --dist l2  --plotroc --figname lfw-aligned.png --outdir SoftmaxResNet34_l2dist --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume softmax_resnet34_withoutl2norm.pth --arch resnet34 --loss softmax --dist l2 --l2norm --plotroc --figname C_224.png --outdir SoftmaxResNet34_l2dist_l2norm --logfile C_224.txt --testdb C_224
python test.py --resume softmax_resnet34_withoutl2norm.pth --arch resnet34 --loss softmax --dist l2 --l2norm --plotroc --figname C.png --outdir SoftmaxResNet34_l2dist_l2norm --logfile C.txt --testdb C
python test.py --resume softmax_resnet34_withoutl2norm.pth --arch resnet34 --loss softmax --dist l2 --l2norm --plotroc --figname lfw-aligned.png --outdir SoftmaxResNet34_l2dist_l2norm --logfile lfw-aligned.txt --testdb lfw-aligned

# =========================================================================================

python test.py --resume asoftmax_resnet18.pth --arch resnet18 --loss sphereface --dist cosine --plotroc --figname C_224.png --outdir ASoftmaxResNet18_cosdist --logfile C_224.txt --testdb C_224
python test.py --resume asoftmax_resnet18.pth --arch resnet18 --loss sphereface --dist cosine --plotroc --figname C.png --outdir ASoftmaxResNet18_cosdist --logfile C.txt --testdb C
python test.py --resume asoftmax_resnet18.pth --arch resnet18 --loss sphereface --dist cosine --plotroc --figname lfw-aligned.png --outdir ASoftmaxResNet18_cosdist --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume asoftmax_resnet18.pth --arch resnet18 --loss sphereface --dist cosine --l2norm --plotroc --figname C_224.png --outdir ASoftmaxResNet18_cosdist_l2norm --logfile C_224.txt --testdb C_224
python test.py --resume asoftmax_resnet18.pth --arch resnet18 --loss sphereface --dist cosine --l2norm --plotroc --figname C.png --outdir ASoftmaxResNet18_cosdist_l2norm --logfile C.txt --testdb C
python test.py --resume asoftmax_resnet18.pth --arch resnet18 --loss sphereface --dist cosine --l2norm --plotroc --figname lfw-aligned.png --outdir ASoftmaxResNet18_cosdist_l2norm --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume asoftmax_resnet18.pth --arch resnet18 --loss sphereface --dist l2  --plotroc --figname C_224.png --outdir ASoftmaxResNet18_l2dist --logfile C_224.txt --testdb C_224
python test.py --resume asoftmax_resnet18.pth --arch resnet18 --loss sphereface --dist l2  --plotroc --figname C.png --outdir ASoftmaxResNet18_l2dist --logfile C.txt --testdb C
python test.py --resume asoftmax_resnet18.pth --arch resnet18 --loss sphereface --dist l2  --plotroc --figname lfw-aligned.png --outdir ASoftmaxResNet18_l2dist --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume asoftmax_resnet18.pth --arch resnet18 --loss sphereface --dist l2 --l2norm --plotroc --figname C_224.png --outdir ASoftmaxResNet18_l2dist_l2norm --logfile C_224.txt --testdb C_224
python test.py --resume asoftmax_resnet18.pth --arch resnet18 --loss sphereface --dist l2 --l2norm --plotroc --figname C.png --outdir ASoftmaxResNet18_l2dist_l2norm --logfile C.txt --testdb C
python test.py --resume asoftmax_resnet18.pth --arch resnet18 --loss sphereface --dist l2 --l2norm --plotroc --figname lfw-aligned.png --outdir ASoftmaxResNet18_l2dist_l2norm --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume asoftmax_resnet34.pth --arch resnet34 --loss sphereface --dist cosine --plotroc --figname C_224.png --outdir ASoftmaxResNet34_cosdist --logfile C_224.txt --testdb C_224
python test.py --resume asoftmax_resnet34.pth --arch resnet34 --loss sphereface --dist cosine --plotroc --figname C.png --outdir ASoftmaxResNet34_cosdist --logfile C.txt --testdb C
python test.py --resume asoftmax_resnet34.pth --arch resnet34 --loss sphereface --dist cosine --plotroc --figname lfw-aligned.png --outdir ASoftmaxResNet34_cosdist --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume asoftmax_resnet34.pth --arch resnet34 --loss sphereface --dist cosine --l2norm --plotroc --figname C_224.png --outdir ASoftmaxResNet34_cosdist_l2norm --logfile C_224.txt --testdb C_224
python test.py --resume asoftmax_resnet34.pth --arch resnet34 --loss sphereface --dist cosine --l2norm --plotroc --figname C.png --outdir ASoftmaxResNet34_cosdist_l2norm --logfile C.txt --testdb C
python test.py --resume asoftmax_resnet34.pth --arch resnet34 --loss sphereface --dist cosine --l2norm --plotroc --figname lfw-aligned.png --outdir ASoftmaxResNet34_cosdist_l2norm --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume asoftmax_resnet34.pth --arch resnet34 --loss sphereface --dist l2  --plotroc --figname C_224.png --outdir ASoftmaxResNet34_l2dist --logfile C_224.txt --testdb C_224
python test.py --resume asoftmax_resnet34.pth --arch resnet34 --loss sphereface --dist l2  --plotroc --figname C.png --outdir ASoftmaxResNet34_l2dist --logfile C.txt --testdb C
python test.py --resume asoftmax_resnet34.pth --arch resnet34 --loss sphereface --dist l2  --plotroc --figname lfw-aligned.png --outdir ASoftmaxResNet34_l2dist --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume asoftmax_resnet34.pth --arch resnet34 --loss sphereface --dist l2 --l2norm --plotroc --figname C_224.png --outdir ASoftmaxResNet34_l2dist_l2norm --logfile C_224.txt --testdb C_224
python test.py --resume asoftmax_resnet34.pth --arch resnet34 --loss sphereface --dist l2 --l2norm --plotroc --figname C.png --outdir ASoftmaxResNet34_l2dist_l2norm --logfile C.txt --testdb C
python test.py --resume asoftmax_resnet34.pth --arch resnet34 --loss sphereface --dist l2 --l2norm --plotroc --figname lfw-aligned.png --outdir ASoftmaxResNet34_l2dist_l2norm --logfile lfw-aligned.txt --testdb lfw-aligned

# =========================================================================================

python test.py --resume triplet_resnet18.pth --arch resnet18 --loss triplet --dist cosine --plotroc --figname C_224.png --outdir TripletResNet18_cosdist --logfile C_224.txt --testdb C_224
python test.py --resume triplet_resnet18.pth --arch resnet18 --loss triplet --dist cosine --plotroc --figname C.png --outdir TripletResNet18_cosdist --logfile C.txt --testdb C
python test.py --resume triplet_resnet18.pth --arch resnet18 --loss triplet --dist cosine --plotroc --figname lfw-aligned.png --outdir TripletResNet18_cosdist --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume triplet_resnet18.pth --arch resnet18 --loss triplet --dist cosine --l2norm --plotroc --figname C_224.png --outdir TripletResNet18_cosdist_l2norm --logfile C_224.txt --testdb C_224
python test.py --resume triplet_resnet18.pth --arch resnet18 --loss triplet --dist cosine --l2norm --plotroc --figname C.png --outdir TripletResNet18_cosdist_l2norm --logfile C.txt --testdb C
python test.py --resume triplet_resnet18.pth --arch resnet18 --loss triplet --dist cosine --l2norm --plotroc --figname lfw-aligned.png --outdir TripletResNet18_cosdist_l2norm --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume triplet_resnet18.pth --arch resnet18 --loss triplet --dist l2 --plotroc --figname C_224.png --outdir TripletResNet18_l2dist --logfile C_224.txt --testdb C_224
python test.py --resume triplet_resnet18.pth --arch resnet18 --loss triplet --dist l2 --plotroc --figname C.png --outdir TripletResNet18_l2dist --logfile C.txt --testdb C
python test.py --resume triplet_resnet18.pth --arch resnet18 --loss triplet --dist l2 --plotroc --figname lfw-aligned.png --outdir TripletResNet18_l2dist --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume triplet_resnet18.pth --arch resnet18 --loss triplet --dist l2 --l2norm --plotroc --figname C_224.png --outdir TripletResNet18_l2dist_l2norm --logfile C_224.txt --testdb C_224
python test.py --resume triplet_resnet18.pth --arch resnet18 --loss triplet --dist l2 --l2norm --plotroc --figname C.png --outdir TripletResNet18_l2dist_l2norm --logfile C.txt --testdb C
python test.py --resume triplet_resnet18.pth --arch resnet18 --loss triplet --dist l2 --l2norm --plotroc --figname lfw-aligned.png --outdir TripletResNet18_l2dist_l2norm --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume triplet_resnet34.pth --arch resnet34 --loss triplet --dist cosine --plotroc --figname C_224.png --outdir TripletResNet34_cosdist --logfile C_224.txt --testdb C_224
python test.py --resume triplet_resnet34.pth --arch resnet34 --loss triplet --dist cosine --plotroc --figname C.png --outdir TripletResNet34_cosdist --logfile C.txt --testdb C
python test.py --resume triplet_resnet34.pth --arch resnet34 --loss triplet --dist cosine --plotroc --figname lfw-aligned.png --outdir TripletResNet34_cosdist --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume triplet_resnet34.pth --arch resnet34 --loss triplet --dist cosine --l2norm --plotroc --figname C_224.png --outdir TripletResNet34_cosdist_l2norm --logfile C_224.txt --testdb C_224
python test.py --resume triplet_resnet34.pth --arch resnet34 --loss triplet --dist cosine --l2norm --plotroc --figname C.png --outdir TripletResNet34_cosdist_l2norm --logfile C.txt --testdb C
python test.py --resume triplet_resnet34.pth --arch resnet34 --loss triplet --dist cosine --l2norm --plotroc --figname lfw-aligned.png --outdir TripletResNet34_cosdist_l2norm --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume triplet_resnet34.pth --arch resnet34 --loss triplet --dist l2 --plotroc --figname C_224.png --outdir TripletResNet34_l2dist --logfile C_224.txt --testdb C_224
python test.py --resume triplet_resnet34.pth --arch resnet34 --loss triplet --dist l2 --plotroc --figname C.png --outdir TripletResNet34_l2dist --logfile C.txt --testdb C
python test.py --resume triplet_resnet34.pth --arch resnet34 --loss triplet --dist l2 --plotroc --figname lfw-aligned.png --outdir TripletResNet34_l2dist --logfile lfw-aligned.txt --testdb lfw-aligned

python test.py --resume triplet_resnet34.pth --arch resnet34 --loss triplet --dist l2 --l2norm --plotroc --figname C_224.png --outdir TripletResNet34_l2dist_l2norm --logfile C_224.txt --testdb C_224
python test.py --resume triplet_resnet34.pth --arch resnet34 --loss triplet --dist l2 --l2norm --plotroc --figname C.png --outdir TripletResNet34_l2dist_l2norm --logfile C.txt --testdb C
python test.py --resume triplet_resnet34.pth --arch resnet34 --loss triplet --dist l2 --l2norm --plotroc --figname lfw-aligned.png --outdir TripletResNet34_l2dist_l2norm --logfile lfw-aligned.txt --testdb lfw-aligned

# Merge Plot

python mergePlot.py --pkllist SoftmaxResNet18_cosdist/C_224.pkl,SoftmaxResNet18_cosdist_l2norm/C_224.pkl,SoftmaxResNet18_l2dist/C_224.pkl,SoftmaxResNet18_l2dist_l2norm/C_224.pkl --out SoftmaxResNet18_dist_norm.png
python mergePlot.py --pkllist ASoftmaxResNet18_cosdist/C_224.pkl,ASoftmaxResNet18_cosdist_l2norm/C_224.pkl,ASoftmaxResNet18_l2dist/C_224.pkl,ASoftmaxResNet18_l2dist_l2norm/C_224.pkl --out ASoftmaxResNet18_dist_norm.png
python mergePlot.py --pkllist TripletResNet18_cosdist/C_224.pkl,TripletResNet18_cosdist_l2norm/C_224.pkl,TripletResNet18_l2dist/C_224.pkl,TripletResNet18_l2dist_l2norm/C_224.pkl --out TripletResNet18_dist_norm.png

python mergePlot.py --pkllist SoftmaxResNet34_cosdist/C_224.pkl,SoftmaxResNet34_cosdist_l2norm/C_224.pkl,SoftmaxResNet34_l2dist/C_224.pkl,SoftmaxResNet34_l2dist_l2norm/C_224.pkl --out SoftmaxResNet34_dist_norm.png
python mergePlot.py --pkllist ASoftmaxResNet34_cosdist/C_224.pkl,ASoftmaxResNet34_cosdist_l2norm/C_224.pkl,ASoftmaxResNet34_l2dist/C_224.pkl,ASoftmaxResNet34_l2dist_l2norm/C_224.pkl --out ASoftmaxResNet34_dist_norm.png
python mergePlot.py --pkllist TripletResNet34_cosdist/C_224.pkl,TripletResNet34_cosdist_l2norm/C_224.pkl,TripletResNet34_l2dist/C_224.pkl,TripletResNet34_l2dist_l2norm/C_224.pkl --out TripletResNet34_dist_norm.png


python mergePlot.py --pkllist SoftmaxResNet18_cosdist/C_224.pkl,SoftmaxResNet18_cosdist_l2norm/C_224.pkl,ASoftmaxResNet18_cosdist/C_224.pkl,ASoftmaxResNet18_cosdist_l2norm/C_224.pkl,TripletResNet18_cosdist/C_224.pkl,TripletResNet18_cosdist_l2norm/C_224.pkl --out ResNet18_C_224_cosdist.png
python mergePlot.py --pkllist SoftmaxResNet18_cosdist/C.pkl,SoftmaxResNet18_cosdist_l2norm/C.pkl,ASoftmaxResNet18_cosdist/C.pkl,ASoftmaxResNet18_cosdist_l2norm/C.pkl,TripletResNet18_cosdist/C.pkl,TripletResNet18_cosdist_l2norm/C.pkl --out ResNet18_C_cosdist.png
python mergePlot.py --pkllist SoftmaxResNet18_cosdist/lfw-aligned.pkl,SoftmaxResNet18_cosdist_l2norm/lfw-aligned.pkl,ASoftmaxResNet18_cosdist/lfw-aligned.pkl,ASoftmaxResNet18_cosdist_l2norm/lfw-aligned.pkl,TripletResNet18_cosdist/lfw-aligned.pkl,TripletResNet18_cosdist_l2norm/lfw-aligned.pkl --out ResNet18_lfw_cosdist.png


python mergePlot.py --pkllist SoftmaxResNet18_l2dist/C_224.pkl,SoftmaxResNet18_l2dist_l2norm/C_224.pkl,ASoftmaxResNet18_l2dist/C_224.pkl,ASoftmaxResNet18_l2dist_l2norm/C_224.pkl,TripletResNet18_l2dist/C_224.pkl,TripletResNet18_l2dist_l2norm/C_224.pkl --out ResNet18_C_224_l2dist.png
python mergePlot.py --pkllist SoftmaxResNet18_l2dist/C.pkl,SoftmaxResNet18_l2dist_l2norm/C.pkl,ASoftmaxResNet18_l2dist/C.pkl,ASoftmaxResNet18_l2dist_l2norm/C.pkl,TripletResNet18_l2dist/C.pkl,TripletResNet18_l2dist_l2norm/C.pkl --out ResNet18_C_l2dist.png
python mergePlot.py --pkllist SoftmaxResNet18_l2dist/lfw-aligned.pkl,SoftmaxResNet18_l2dist_l2norm/lfw-aligned.pkl,ASoftmaxResNet18_l2dist/lfw-aligned.pkl,ASoftmaxResNet18_l2dist_l2norm/lfw-aligned.pkl,TripletResNet18_l2dist/lfw-aligned.pkl,TripletResNet18_l2dist_l2norm/lfw-aligned.pkl --out ResNet18_lfw_l2dist.png



python mergePlot.py --pkllist SoftmaxResNet34_cosdist/C_224.pkl,SoftmaxResNet34_cosdist_l2norm/C_224.pkl,ASoftmaxResNet34_cosdist/C_224.pkl,ASoftmaxResNet34_cosdist_l2norm/C_224.pkl,TripletResNet34_cosdist/C_224.pkl,TripletResNet34_cosdist_l2norm/C_224.pkl --out ResNet34_C_224_cosdist.png
python mergePlot.py --pkllist SoftmaxResNet34_cosdist/C.pkl,SoftmaxResNet34_cosdist_l2norm/C.pkl,ASoftmaxResNet34_cosdist/C.pkl,ASoftmaxResNet34_cosdist_l2norm/C.pkl,TripletResNet34_cosdist/C.pkl,TripletResNet34_cosdist_l2norm/C.pkl --out ResNet34_C_cosdist.png
python mergePlot.py --pkllist SoftmaxResNet34_cosdist/lfw-aligned.pkl,SoftmaxResNet34_cosdist_l2norm/lfw-aligned.pkl,ASoftmaxResNet34_cosdist/lfw-aligned.pkl,ASoftmaxResNet34_cosdist_l2norm/lfw-aligned.pkl,TripletResNet34_cosdist/lfw-aligned.pkl,TripletResNet34_cosdist_l2norm/lfw-aligned.pkl --out ResNet34_lfw_cosdist.png


python mergePlot.py --pkllist SoftmaxResNet34_l2dist/C_224.pkl,SoftmaxResNet34_l2dist_l2norm/C_224.pkl,ASoftmaxResNet34_l2dist/C_224.pkl,ASoftmaxResNet34_l2dist_l2norm/C_224.pkl,TripletResNet34_l2dist/C_224.pkl,TripletResNet34_l2dist_l2norm/C_224.pkl --out ResNet34_C_224_l2dist.png
python mergePlot.py --pkllist SoftmaxResNet34_l2dist/C.pkl,SoftmaxResNet34_l2dist_l2norm/C.pkl,ASoftmaxResNet34_l2dist/C.pkl,ASoftmaxResNet34_l2dist_l2norm/C.pkl,TripletResNet34_l2dist/C.pkl,TripletResNet34_l2dist_l2norm/C.pkl --out ResNet34_C_l2dist.png
python mergePlot.py --pkllist SoftmaxResNet34_l2dist/lfw-aligned.pkl,SoftmaxResNet34_l2dist_l2norm/lfw-aligned.pkl,ASoftmaxResNet34_l2dist/lfw-aligned.pkl,ASoftmaxResNet34_l2dist_l2norm/lfw-aligned.pkl,TripletResNet34_l2dist/lfw-aligned.pkl,TripletResNet34_l2dist_l2norm/lfw-aligned.pkl --out ResNet34_lfw_l2dist.png