from torch.utils.data import DataLoader
from dataset import CrackDataset
import torchvision.transforms as T


transform = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
# Paths to your folders and COCO JSON files
train_images = "dataset/train"
train_json   = "dataset/train/_annotations.coco.json"

valid_images = "dataset/valid"
valid_json   = "dataset/valid/_annotations.coco.json"

test_images  = "dataset/test"
test_json    = "dataset/test/_annotations.coco.json"

# Create datasets
train_dataset = CrackDataset(train_images, train_json)
valid_dataset = CrackDataset(valid_images, valid_json)
test_dataset  = CrackDataset(test_images, test_json)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)