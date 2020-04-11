import pickle, os, argparse
import matplotlib.pyplot as plt

"""
# Usage:

python mergePlot.py --pkllist SoftmaxResNet18_cosdist/C_224.pkl,SoftmaxResNet18_cosdist_l2norm/C_224.pkl,SoftmaxResNet18_l2dist/C_224.pkl,SoftmaxResNet18_l2dist_l2norm/C_224.pkl --out SoftmaxResNet18_dist_norm.png
python mergePlot.py --pkllist ASoftmaxResNet18_cosdist/C_224.pkl,ASoftmaxResNet18_cosdist_l2norm/C_224.pkl,ASoftmaxResNet18_l2dist/C_224.pkl,ASoftmaxResNet18_l2dist_l2norm/C_224.pkl --out ASoftmaxResNet18_dist_norm.png
python mergePlot.py --pkllist TripletResNet18_cosdist/C_224.pkl,TripletResNet18_cosdist_l2norm/C_224.pkl,TripletResNet18_l2dist/C_224.pkl,TripletResNet18_l2dist_l2norm/C_224.pkl --out TripletResNet18_dist_norm.png

python mergePlot.py --pkllist SoftmaxResNet34_cosdist/C_224.pkl,SoftmaxResNet34_cosdist_l2norm/C_224.pkl,SoftmaxResNet34_l2dist/C_224.pkl,SoftmaxResNet34_l2dist_l2norm/C_224.pkl --out SoftmaxResNet34_dist_norm.png
python mergePlot.py --pkllist ASoftmaxResNet34_cosdist/C_224.pkl,ASoftmaxResNet34_cosdist_l2norm/C_224.pkl,ASoftmaxResNet34_l2dist/C_224.pkl,ASoftmaxResNet34_l2dist_l2norm/C_224.pkl --out ASoftmaxResNet34_dist_norm.png
python mergePlot.py --pkllist TripletResNet34_cosdist/C_224.pkl,TripletResNet34_cosdist_l2norm/C_224.pkl,TripletResNet34_l2dist/C_224.pkl,TripletResNet34_l2dist_l2norm/C_224.pkl --out TripletResNet34_dist_norm.png


python mergePlot.py --pkllist SoftmaxResNet18_cosdist/C_224.pkl,SoftmaxResNet18_cosdist_l2norm/C_224.pkl,ASoftmaxResNet18_cosdist/C_224.pkl,ASoftmaxResNet18_cosdist_l2norm/C_224.pkl,TripletResNet18_cosdist/C_224.pkl,TripletResNet18_cosdist_l2norm/C_224.pkl --out C_224_cosdist.png
python mergePlot.py --pkllist SoftmaxResNet18_cosdist/C.pkl,SoftmaxResNet18_cosdist_l2norm/C.pkl,ASoftmaxResNet18_cosdist/C.pkl,ASoftmaxResNet18_cosdist_l2norm/C.pkl,TripletResNet18_cosdist/C.pkl,TripletResNet18_cosdist_l2norm/C.pkl --out C_cosdist.png
python mergePlot.py --pkllist SoftmaxResNet18_cosdist/lfw-aligned.pkl,SoftmaxResNet18_cosdist_l2norm/lfw-aligned.pkl,ASoftmaxResNet18_cosdist/lfw-aligned.pkl,ASoftmaxResNet18_cosdist_l2norm/lfw-aligned.pkl,TripletResNet18_cosdist/lfw-aligned.pkl,TripletResNet18_cosdist_l2norm/lfw-aligned.pkl --out lfw_cosdist.png


python mergePlot.py --pkllist SoftmaxResNet18_l2dist/C_224.pkl,SoftmaxResNet18_l2dist_l2norm/C_224.pkl,ASoftmaxResNet18_l2dist/C_224.pkl,ASoftmaxResNet18_l2dist_l2norm/C_224.pkl,TripletResNet18_l2dist/C_224.pkl,TripletResNet18_l2dist_l2norm/C_224.pkl --out C_224_l2dist.png
python mergePlot.py --pkllist SoftmaxResNet18_l2dist/C.pkl,SoftmaxResNet18_l2dist_l2norm/C.pkl,ASoftmaxResNet18_l2dist/C.pkl,ASoftmaxResNet18_l2dist_l2norm/C.pkl,TripletResNet18_l2dist/C.pkl,TripletResNet18_l2dist_l2norm/C.pkl --out C_l2dist.png
python mergePlot.py --pkllist SoftmaxResNet18_l2dist/lfw-aligned.pkl,SoftmaxResNet18_l2dist_l2norm/lfw-aligned.pkl,ASoftmaxResNet18_l2dist/lfw-aligned.pkl,ASoftmaxResNet18_l2dist_l2norm/lfw-aligned.pkl,TripletResNet18_l2dist/lfw-aligned.pkl,TripletResNet18_l2dist_l2norm/lfw-aligned.pkl --out lfw_l2dist.png

"""

parser = argparse.ArgumentParser(description='[AMMAI] Plot Merger')
parser.add_argument('--pkllist',default='a.pkl,b.pkl,c.pkl',type=str, help="pickle list. each pickle file is splited by comma")
parser.add_argument('--out',default='expname.png',type=str, help="output path")
args = parser.parse_args()
pathlist = args.pkllist.split(",")
fig = plt.figure()
code = ["red","blue","green","orange","black","pink","cyan","olive","brown","purple"]
for i, path in enumerate(pathlist):
    with open(path,"rb") as f:
        data = pickle.load(f)
    auc = data["auc"]
    fpr = data["fpr"]
    tpr = data["tpr"]
    label = "%s (auc = %.4f)"%(path.split("/")[0],auc)
    plt.plot(fpr, tpr, color=code[i], lw=2, label=label)
plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle="--", label="Random")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
fig.savefig(args.out, dpi=fig.dpi)


