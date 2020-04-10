import pickle, os, argparse
import matplotlib.pyplot as plt

"""
# Usage:
python mergePlot.py --pkllist SoftmaxResNet34/C.pkl,ASoftmaxResNet34/C.pkl,TripletResNet34/C.pkl --out C.png
python mergePlot.py --pkllist SoftmaxResNet34/C_224.pkl,ASoftmaxResNet34/C_224.pkl,TripletResNet34/C_224.pkl --out C_224.png
python mergePlot.py --pkllist SoftmaxResNet34/lfw-aligned.pkl,ASoftmaxResNet34/lfw-aligned.pkl,TripletResNet34/lfw-aligned.pkl --out lfw-aligned.png
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


