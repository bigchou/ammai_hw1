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
from data import data_transforms, APDDataset, TripletFaceDataset
from torch.nn.functional import cosine_similarity, pairwise_distance
#warnings.filterwarnings("ignore")#uncomment this line if you would like to suppress warnings

"""
Data Requirements:
1. summary.csv (summarize your training data, see summary.py) 
3. CASIA-maxpy-clean-aligned (path to store CASIAWebFace dataset)
4. C (path to store APD dataset for evaluation)
5. positive_pairs.txt (evaluate APD)
6. negative_pairs.txt (evaluate APD)

Usage:
# Inference
python main.py --resume path/to/ckpt --arch resnet18 --inference --logfile SoftmaxTest.txt --loss softmax --dist cosine --plotroc --figname Softmax.png --outdir Softmax
python main.py --resume path/to/ckpt --arch resnet18 --inference --logfile TripletTest.txt --loss triplet --l2norm --dist l2 --plotroc --figname Triplet.png --outdir Triplet
python main.py --resume path/to/ckpt --arch resnet18 --inference --logfile ASoftmaxTest.txt --loss sphereface --dist cosine --plotroc --figname ASoftmax.png --outdir ASoftmax

# Training
python main.py --logfile SoftmaxTrain.txt --loss softmax --dist cosine --batch 256 --outdir Softmax
python main.py --logfile TripletTrain.txt --loss triplet --l2norm --dist l2 --batch 128 --outdir Triplet
python main.py --logfile ASoftmaxTrain.txt --loss sphereface --dist cosine --batch 256 --outdir ASoftmax
"""
# define supporting functions
def save_model(model,filename):
    state = model.state_dict()
    for key in state:
        state[key] = state[key].clone().cpu()
    torch.save(state,filename)

def load_model(model,filename):
    model.load_state_dict(torch.load(filename))
    return model

# Arguments
parser = argparse.ArgumentParser(description='[AMMAI] APD Face Verification')
parser.add_argument('--outdir',type=str,help="path to store output results")
parser.add_argument('--resume', default='', type=str, help="resume from the checkpoint")
parser.add_argument("--inference", action="store_true", help="enable inference mode (default: False)")
parser.add_argument("--epoch", default="50", type=int, help="#training epochs (default: 50)")
parser.add_argument("--batch", default="256", type=int, help="batch size (default: 256)")
parser.add_argument("--trainpath", default="CASIA-maxpy-clean-aligned", type=str, help="path to the training image folder")
parser.add_argument("--apdpath",default="C", type=str, help="path to the APD image folder")
parser.add_argument("--logfile", default="log.txt", type=str, help="path to store logging file (default: log.txt)")
parser.add_argument("--loss", default="softmax", type=str, choices=["softmax", "triplet", "sphereface"], help="cost function (default: softmax)")
parser.add_argument("--lr", default="0.1", type=float, help="learning rate (default: 0.1)")
parser.add_argument("--plotroc", action="store_true", help="enable plotting ROC curve (default: False)")
parser.add_argument('--figname', default="roc.png", type=str, help="path to output figure (roc.png)")
parser.add_argument('--l2norm', action="store_true", help="apply l2-norm in the output of the model (default: False)")
parser.add_argument('--dist',default="cosine", type=str, choices=["l2","cosine"], help="distance measurement (default: cosine)")
parser.add_argument('--arch',default='resnet18', type=str, choices=["resnet18","resnet34"], help="backbone (default: resnet18)")
# TripletLoss-specific arguments
parser.add_argument('--summary_csv',default='summary.csv', type=str, help="csv file used to summarize your training data")
parser.add_argument('--training_triplets_path',default='training_triplets_450000.npy', type=str, help="npy file containing preprocessed triplet information")
parser.add_argument('--num_triplets_train', default=450000, type=int, help="#triplets for training (default:450000)")
parser.add_argument('--triplet_margin', default=0.5, type=float, help="triplet margin (default:0.5)")


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
    
    # Dataloader
    if args.loss == "triplet":
        trainset = TripletFaceDataset(
            root_dir=args.trainpath,
            csv_name=args.summary_csv,
            num_triplets=args.num_triplets_train,
            training_triplets_path=args.training_triplets_path,
            transform=data_transforms['train']
        )
        trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)
        num_triplet = len(trainset)
        out = "[TRAINING] num_triplet: %d"%(num_triplet)
        print(out)
        with open(args.logfile,"a",encoding="utf-8") as f: f.write("\n"+out+"\n")
    else:
        trainset = datasets.ImageFolder(root=args.trainpath, transform=data_transforms['train'])
        trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)
        num_class = len(trainset.classes)
        out = "[TRAINING] num_class: %d"%(num_class)
        print(out)
        with open(args.logfile,"a",encoding="utf-8") as f: f.write("\n"+out+"\n")
    valset = APDDataset(root_dir=args.apdpath,transform=data_transforms["val"])
    valloader = DataLoader(valset, batch_size=args.batch, shuffle=False, num_workers=8, pin_memory=True)

    # Model
    if args.arch == "resnet18":
        model = resnet18(pretrained=False,loss_fn=args.loss,num_classes=256 if args.loss=="triplet" else num_class)
    else:
        model = resnet34(pretrained=False,loss_fn=args.loss,num_classes=256 if args.loss=="triplet" else num_class)

    # Resume
    if args.resume:
        print("ready to load "+args.resume)
        model = load_model(model,args.resume)

    # GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print("USE DEVICE: "+device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # distance measure move to device
    if args.loss == "triplet":
        l2dist = PairwiseDistance(2).to(device)

    # Main
    print("===== START =====")
    for epoch in range(args.epoch):
        # ===== Train ============================================================
        if not args.inference:
            model.train()
            if args.loss == "triplet":
                triplet_loss_sum = 0
                num_valid_training_triplets = 0
                epoch_time_start = time.time()
                for batch_idx, (imgs) in enumerate(trainloader):
                    anc_img, pos_img, neg_img = imgs['anc_img'].to(device), imgs['pos_img'].to(device), imgs['neg_img'].to(device)
                    anc_emb, pos_emb, neg_emb = model(anc_img,l2norm=args.l2norm), model(pos_img,l2norm=args.l2norm), model(neg_img,l2norm=args.l2norm)# get embedding
                    # Pick hard negatives only for training
                    pos_dist, neg_dist = l2dist.forward(anc_emb, pos_emb), l2dist.forward(anc_emb, neg_emb)
                    all = (neg_dist - pos_dist < args.triplet_margin).cpu().numpy().flatten()
                    hard_triplets = np.where(all == 1)#difficult to distinguish
                    if len(hard_triplets[0]) == 0: continue
                    anc_hard_emb, pos_hard_emb, neg_hard_emb = anc_emb[hard_triplets].to(device), pos_emb[hard_triplets].to(device), neg_emb[hard_triplets].to(device)
                    # Calculate triplet loss
                    triplet_loss = TripletLoss(margin=args.triplet_margin).forward(anchor=anc_hard_emb,positive=pos_hard_emb,negative=neg_hard_emb).to(device)
                    # Calculating loss
                    triplet_loss_sum += triplet_loss.item()
                    num_valid_training_triplets += len(anc_hard_emb)
                    # Backward pass
                    optimizer.zero_grad()
                    triplet_loss.backward()
                    optimizer.step()
                # Model only trains on hard negative triplets
                avg_triplet_loss = 0 if (num_valid_training_triplets == 0) else triplet_loss_sum / num_valid_training_triplets
                epoch_time_end = time.time()
                # Print training statistics and add to log
                out = 'Epoch {}:\tAverage Triplet Loss: {:.4f}\tEpoch Time: {:.3f} hours\tNumber of valid training triplets in epoch: {}'.format(
                    epoch,
                    avg_triplet_loss,
                    (epoch_time_end - epoch_time_start)/3600,
                    num_valid_training_triplets
                )
                print(out)
                with open(args.logfile, 'a', encoding='utf-8') as f: f.write(out+"\n")
            else:
                for batch_idx, (data, target) in enumerate(trainloader):
                    data = data.to(device)
                    target = target.to(device).long()
                    optimizer.zero_grad()
                    output = model(data,target,l2norm=args.l2norm)
                    loss = criterion(output,target)
                    loss.backward()
                    optimizer.step()
                    if batch_idx%300 == 0:
                        msg = "EPOCH %d, ITER %d, LOSS: %.4f"%(epoch,batch_idx,loss.item())
                        print(msg)
                        with open(args.logfile,"a",encoding="utf-8") as f: f.write(msg+"\n")
        save_model(model,filename="%s/%d.pth"%(args.outdir,epoch))# Save
        if args.loss != "triplet": scheduler.step()# Update scheduler

        # ===== Eval AUC ===========================================
        records = []
        model.eval()
        with torch.no_grad():
            for idx, (name1, img1, name2, img2, issame) in enumerate(valloader):
                name1 = valset.name2label[name1.numpy()]#str
                name2 = valset.name2label[name2.numpy()]#str
                issame = issame.numpy().astype(bool)#bool
                img1 = img1.to(device)
                img2 = img2.to(device)
                # Extract feature
                feat1 = model(img1,featmod=True,l2norm=args.l2norm)#torch.Size([batch, dim, 1, 1])
                feat2 = model(img2,featmod=True,l2norm=args.l2norm)#torch.Size([batch, dim, 1, 1])
                feat1 = feat1.view(-1,feat1.shape[1])#torch.Size([batch, dim])
                feat2 = feat2.view(-1,feat2.shape[1])#torch.Size([batch, dim])
                if idx==0: print("Dim. of current feat. :",feat1.shape[-1])
                # Measure distance
                dists = pairwise_distance(feat1, feat2) if args.dist == "l2" else (1.0 - cosine_similarity(feat1, feat2, dim=1))

                # Record results
                for n1, f1, n2, f2, gt, dist in zip(name1, feat1, name2, feat2, issame, dists):
                    #feat1 = f1.cpu().numpy().tolist()
                    #feat2 = f2.cpu().numpy().tolist()
                    records.append({"name1":str(n1),"name2":str(n2),"gt_issame":bool(gt),"dist":dist.item()})
        distances = np.array([rec["dist"] for rec in records])
        labels = np.array([rec["gt_issame"] for rec in records])
        true_positive_rate, false_positive_rate, roc_auc = eval(distances=distances,labels=labels)
        print("roc_auc:%f"%(roc_auc))
        if args.plotroc: plot_roc_lfw(false_positive_rate,true_positive_rate, figure_name=args.figname)
        with open(args.logfile,"a",encoding="utf-8") as f: f.write("roc_auc: %f\n"%(roc_auc))
        if args.inference: break
        # ===========================================================


        
       


        
