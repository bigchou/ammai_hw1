import os, torch
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.datasets as datasets


def add_extension(path):
    # Support .jpg and .png only
    if os.path.exists(path + '.jpg'):
        return path + '.jpg'
    elif os.path.exists(path + '.png'):
        return path + '.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

# Preprocessing
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class APDDataset(Dataset):
    def __init__(self, root_dir, ext=".jpg", pospairfile="positive_pairs.txt", negpairfile="negative_pairs.txt", transform=None, log="missfile.txt"):
        name2label = {}# convert name string to integer label
        data = []# [ [name1, path1, name2, path2, issame], ...,]
        miss = set()# record missing files
        k = 0
        for idx, pairfile in enumerate([pospairfile, negpairfile]):
            with open(pairfile,"r",encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    tmp = line.split("\t")
                    # Parsing
                    if len(tmp) == 4:
                        name1, id1, name2, id2 = line.split("\t")
                    else:
                        name1, id1, id2 = line.split("\t")
                    path1 = os.path.join(root_dir,name1+"_"+id1+ext)
                    if len(tmp) == 4:
                        path2 = os.path.join(root_dir,name2+"_"+id2+ext)
                        tmp = [name1, path1, name2, path2]
                    else:
                        path2 = os.path.join(root_dir,name1+"_"+id2+ext)
                        tmp = [name1, path1, name1, path2]
                    # Skip missing files
                    if not os.path.exists(tmp[1]):
                        if tmp[1] not in miss:
                            miss.add(tmp[1])
                        continue
                    if not os.path.exists(tmp[3]):
                        if tmp[3] not in miss:
                            miss.add(tmp[3])
                        continue
                    # Mapping
                    if tmp[0] not in name2label:
                        name2label[tmp[0]] = k
                        k+=1
                    tmp[0] = name2label[tmp[0]]
                    if tmp[2] not in name2label:
                        name2label[tmp[2]] = k
                        k+=1
                    tmp[2] = name2label[tmp[2]]
                    tmp.append(False if idx else True)
                    data.append(tmp)
        print("#pairs: %d"%(len(data)))
        with open(log,"w",encoding="utf-8") as f:
            f.write("#pairs: %d\n"%(len(data)))
            f.write("#missing files: %d\n===============\n"%(len(miss)))
            for i in miss:
                f.write(i+"\n")
        # ==================================================================
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        self.name2label = np.array(list(name2label.keys()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        content = self.data[idx]
        name1 = torch.from_numpy(np.array(content[0],dtype=np.uint8))
        sample1 = Image.open(content[1]).convert('RGB')
        name2 = torch.from_numpy(np.array(content[2],dtype=np.uint8))
        sample2 = Image.open(content[3]).convert('RGB')
        issame = torch.from_numpy(np.array(content[-1],dtype=np.uint8))
        if self.transform:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)
        return name1, sample1, name2, sample2, issame



class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, csv_name, num_triplets, training_triplets_path=None, transform = None):
        self.root_dir          = root_dir
        self.df = pd.read_csv(csv_name, dtype={'id': object, 'name': object, 'class': int})
        self.num_triplets      = num_triplets
        self.transform         = transform
        if training_triplets_path:
            if os.path.exists(training_triplets_path):
                self.training_triplets = np.load(training_triplets_path)
            else:
                self.training_triplets = self.generate_triplets(self.df, self.num_triplets)
        else:
            self.training_triplets = self.generate_triplets(self.df, self.num_triplets)

    @staticmethod
    def generate_triplets(df, num_triplets):
        def make_dictionary_for_face_class(df):
            face_classes = dict()
            for idx, label in enumerate(df['class']):
                if label not in face_classes:
                    face_classes[label] = []
                face_classes[label].append(df.iloc[idx, 0])
            return face_classes#{'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
        triplets    = []
        classes     = df['class'].unique()
        face_classes = make_dictionary_for_face_class(df)
        print("\nGenerating {} triplets...".format(num_triplets))
        for _ in range(num_triplets):
            '''
                - randomly choose anchor, positive and negative images for triplet loss
                - anchor and positive images in pos_class
                - negative image in neg_class
                - at least, two images needed for anchor and positive images in pos_class
                - negative image should have different class as anchor and positive images by definition
            '''
            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)
            
            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size = 2, replace = False)
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_classes[pos_class]))
            ineg = np.random.randint(0, len(face_classes[neg_class]))
            
            triplets.append([
                    face_classes[pos_class][ianc],
                    face_classes[pos_class][ipos],
                    face_classes[neg_class][ineg],
                    pos_class,
                    neg_class,
                    pos_name,
                    neg_name
                ]
            )
        # Save the triplets as a *.npy to not have to redo this process every training execution from scratch
        print("Saving training triplets list in current directory ...")
        np.save('training_triplets_{}.npy'.format(num_triplets), triplets)
        print("Training triplets' list Saved!\n")
        return triplets

    def __len__(self):
        return len(self.training_triplets)

    def __getitem__(self, idx):
        
        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]
        
        anc_img   = add_extension(os.path.join(self.root_dir, str(pos_name), str(anc_id)))
        pos_img   = add_extension(os.path.join(self.root_dir, str(pos_name), str(pos_id)))
        neg_img   = add_extension(os.path.join(self.root_dir, str(neg_name), str(neg_id)))
        
        # Modified to open as PIL image in the first place
        anc_img   = Image.open(anc_img)
        pos_img   = Image.open(pos_img)
        neg_img   = Image.open(neg_img)

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))
        
        sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img, 'pos_class': pos_class, 'neg_class': neg_class}

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])
        return sample

class LFWDataset(datasets.ImageFolder):
    def __init__(self, dir, pairs_path="LFW_pairs.txt", transform=None):
        super(LFWDataset, self).__init__(dir, transform)
        self.pairs_path = pairs_path
        self.validation_images = self.get_lfw_paths(dir)
        print("#pairs: %d"%(len(self.validation_images)))

    def read_lfw_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs)

    def get_lfw_paths(self, lfw_dir):
        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []
        for pair in pairs:
            if len(pair) == 3:
                path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
                issame = True
            elif len(pair) == 4:
                path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                path_list.append((path0, path1, issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)
        return path_list


    def __getitem__(self, index):
        """
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        """

        def transform(img_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """

            img = self.loader(img_path)
            return self.transform(img)

        (path_1, path_2, issame) = self.validation_images[index]
        img1, img2 = transform(path_1), transform(path_2)
        return img1, img2, issame

    def __len__(self):
        return len(self.validation_images)
