

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
trainset = datasets.ImageFolder(root="CASIA-maxpy-clean-aligned", transform=None)
trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
print(len(trainloader))
