import torch, pdb, os, torchvision, argparse, warnings, json, PIL, time, sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
from metrics import TripletLoss
from torchvision import datasets
from roc import eval, plot_roc_lfw
from modified_resnet import resnet18, resnet34
from torch.nn.modules.distance import PairwiseDistance
from data import data_transforms, APDDataset, TripletFaceDataset, LFWDataset
from torch.nn.functional import cosine_similarity, pairwise_distance
"""
Data Requirements:
1. C (path to store official alignment APD dataset. Size: 250x250)
2. C_224 (path to store my alignment APD dataset. Size: 224x224)
3. lfw-aligned (path to store my alignment APD dataset. Size: 224x224)
3. positive_pairs.txt (evaluate APD)
4. negative_pairs.txt (evaluate APD)
6. LFW_pairs.txt (evaluate LFW)


Usage:
# Inference
python main.py --resume path/to/ckpt --arch resnet18 --inference --logfile SoftmaxTest.txt --loss softmax --dist cosine --plotroc --figname Softmax.png --outdir Softmax
python main.py --resume path/to/ckpt --arch resnet18 --inference --logfile TripletTest.txt --loss triplet --l2norm --dist l2 --plotroc --figname Triplet.png --outdir Triplet
python main.py --resume path/to/ckpt --arch resnet18 --inference --logfile ASoftmaxTest.txt --loss sphereface --dist cosine --plotroc --figname ASoftmax.png --outdir ASoftmax
"""
parser = argparse.ArgumentParser(description='[AMMAI] APD Face Verification [EVAL]')
# choose testing datset
parser.add_argument('--testdb',default='C',choices=["C","C_224","lfw-aligned"],type=str, help="testing dataset")
# path to save experiment information
parser.add_argument('--outdir',type=str,help="path to store output results")
parser.add_argument("--logfile", default="log.txt", type=str, help="path to store logging file (default: log.txt)")
# plotting control
parser.add_argument("--plotroc", action="store_true", help="enable plotting ROC curve (default: False)")
parser.add_argument('--figname', default="roc.png", type=str, help="path to output figure (roc.png)")
# model config
parser.add_argument('--arch',default='resnet18', type=str, choices=["resnet18","resnet34"], help="backbone (default: resnet18)")
parser.add_argument('--resume', default='', type=str, help="resume from the checkpoint")
parser.add_argument("--loss", default="softmax", type=str, choices=["softmax", "triplet", "sphereface"], help="cost function (default: softmax)")
parser.add_argument('--dist',default="cosine", type=str, choices=["l2","cosine"], help="distance measurement (default: cosine)")
parser.add_argument('--l2norm', action="store_true", help="apply l2-norm in the output of the model (default: False)")

if __name__ == "__main__":
    # Log arguments
    args = parser.parse_args()
    print(args)
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)
        print("create dir "+args.outdir)
    args.logfile = os.path.join(args.outdir,args.logfile)
    args.figname = os.path.join(args.outdir,args.figname)
    with open(args.logfile,"w",encoding="utf-8") as f:
        f.write("\n".join(sys.argv[1:]))
        f.write("\n===============================\n")
        json.dump(args.__dict__, f, indent=2)
        print("save experiment info into "+args.logfile)
    # Model
    if args.arch == "resnet18":
        model = resnet18(pretrained=False,loss_fn=args.loss,num_classes=256 if args.loss=="triplet" else 10575)
    else:
        model = resnet34(pretrained=False,loss_fn=args.loss,num_classes=256 if args.loss=="triplet" else 10575)
    if args.resume:
        print("ready to load "+args.resume)
        model = model.load_state_dict(torch.load(filename))
    # GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print("USE DEVICE: "+device)
    # Dataloader
    if args.testdb[0] == "C":
        valset = APDDataset(root_dir=args.testdb,transform=data_transforms["val"])
        valloader = DataLoader(valset, batch_size=args.batch, shuffle=False, num_workers=8, pin_memory=True)
    else:
        valset = LFWDataset(dir=args.testdb,transform=data_transforms["val"])
        valloader = DataLoader(valset, batch_size=args.batch, shuffle=False, num_workers=8, pin_memory=True)
    # Test
    print("START TESTING")
    records = []
    model.eval()
    with torch.no_grad():
        for idx, tmp in enumerate(valloader):
            #tmp = (name1, img1, name2, img2, issame)
            issame = tmp[-1].numpy().astype(bool)#bool
            img1 = tmp[1].to(device)
            img2 = tmp[3].to(device)
            # Extract feature
            feat1 = model(img1,featmod=True,l2norm=args.l2norm)#torch.Size([batch, dim, 1, 1])
            feat2 = model(img2,featmod=True,l2norm=args.l2norm)#torch.Size([batch, dim, 1, 1])
            feat1 = feat1.view(-1,feat1.shape[1])#torch.Size([batch, dim])
            feat2 = feat2.view(-1,feat2.shape[1])#torch.Size([batch, dim])
            if idx==0: print("Dim. of current feat. :",feat1.shape[-1])
            # Measure distance
            dists = pairwise_distance(feat1, feat2) if args.dist == "l2" else (1.0 - cosine_similarity(feat1, feat2, dim=1))

            # Record results
            for gt, dist in zip(issame, dists):
                records.append({"gt_issame":bool(gt),"dist":dist.item()})
    distances = np.array([rec["dist"] for rec in records])
    labels = np.array([rec["gt_issame"] for rec in records])
    true_positive_rate, false_positive_rate, roc_auc = eval(distances=distances,labels=labels)
    print("roc_auc:%f"%(roc_auc))
    if args.plotroc: plot_roc_lfw(false_positive_rate,true_positive_rate, figure_name=args.figname)
    with open(args.logfile,"a",encoding="utf-8") as f: f.write("roc_auc: %f\n"%(roc_auc))
