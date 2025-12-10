import os
import json
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.optim as optim

# Mengubah polygon COCO menjadi mask biner
def polygons_to_mask(polygons, height, width):
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if len(poly) < 6:
            continue
        xy = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
        draw.polygon(xy, fill=1)
    return np.array(mask, dtype=np.uint8)

# Kelas Dataset COCO untuk segmentasi
class CocoSegDataset(Dataset):
    def __init__(self, images_dir, coco_json_path, classes, transform=None, img_size=(256,256)):
        self.images_dir = images_dir
        self.coco = json.load(open(coco_json_path, 'r'))
        self.transform = transform
        self.img_size = img_size

        self.classes = classes
        self.num_classes = len(classes)

        self.imgs = {img['id']: img for img in self.coco['images']}
        self.ids = list(self.imgs.keys())

        # Group annotations by image_id
        self.anns = {}
        for ann in self.coco['annotations']:
            self.anns.setdefault(ann['image_id'], []).append(ann)

        # Map COCO category id → class index
        catid2name = {c['id']: c['name'] for c in self.coco['categories']}
        self.catname2idx = {name: i for i,name in enumerate(classes)}
        self.catid2idx = {cid:self.catname2idx[name] for cid,name in catid2name.items()}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        meta = self.imgs[img_id]

        img_path = os.path.join(self.images_dir, meta['file_name'])
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        # Multi-channel masks
        masks = np.zeros((self.num_classes, orig_h, orig_w), dtype=np.uint8)
        presence = np.zeros(self.num_classes, dtype=np.float32)

        anns = self.anns.get(img_id, [])
        for ann in anns:
            cid = ann['category_id']
            cls_idx = self.catid2idx[cid]

            seg = ann['segmentation']
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

            for polygon in seg:
                m = polygons_to_mask([polygon], orig_h, orig_w)
                mask = np.maximum(mask, m)

            masks[cls_idx] = np.maximum(masks[cls_idx], mask)
            presence[cls_idx] = 1

        # Resize image + masks
        img = img.resize(self.img_size, Image.BILINEAR)

        masks_resized = np.zeros((self.num_classes, self.img_size[1], self.img_size[0]), dtype=np.uint8)
        for c in range(self.num_classes):
            masks_resized[c] = np.array(Image.fromarray(masks[c]).resize(self.img_size, Image.NEAREST))

        # Transform
        if self.transform:
            img = self.transform(img)
        else:
            img = TF.to_tensor(img)

        masks_tensor = torch.from_numpy(masks_resized).float()
        presence_tensor = torch.from_numpy(presence).float()

        return img, masks_tensor, presence_tensor

# Model U-Net Multi-Task
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

# Kelas U-Net Multi-Task
class UNetMultiTask(nn.Module):
    def __init__(self, in_channels=3, base_c=32, num_classes=11):
        super().__init__()

        self.enc1 = conv_block(in_channels, base_c)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = conv_block(base_c, base_c*2)
        self.enc3 = conv_block(base_c*2, base_c*4)
        self.enc4 = conv_block(base_c*4, base_c*8)

        self.bottleneck = conv_block(base_c*8, base_c*16)

        self.up4 = nn.ConvTranspose2d(base_c*16, base_c*8, 2, 2)
        self.dec4 = conv_block(base_c*16, base_c*8)
        self.up3 = nn.ConvTranspose2d(base_c*8, base_c*4, 2, 2)
        self.dec3 = conv_block(base_c*8, base_c*4)
        self.up2 = nn.ConvTranspose2d(base_c*4, base_c*2, 2, 2)
        self.dec2 = conv_block(base_c*4, base_c*2)
        self.up1 = nn.ConvTranspose2d(base_c*2, base_c, 2, 2)
        self.dec1 = conv_block(base_c*2, base_c)

        self.seg_head = nn.Conv2d(base_c, num_classes, 1)

        # Classification head
        self.cls_pool = nn.AdaptiveAvgPool2d((1,1))
        self.cls_fc = nn.Linear(base_c*16, num_classes)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        bott = self.bottleneck(p4)

        # classification branch
        cls_vec = self.cls_pool(bott).view(bott.size(0), -1)
        cls_logits = self.cls_fc(cls_vec)

        # decoding U-Net
        d4 = self.up4(bott)
        d4 = self.dec4(torch.cat([d4, e4], 1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], 1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], 1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], 1))

        seg_logits = self.seg_head(d1)

        return seg_logits, cls_logits

def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum((2,3))
    union = pred.sum((2,3)) + target.sum((2,3))
    dice = (2*intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# Fungsi untuk melatih model
def train_model(train_loader, val_loader, model, device, epochs=20, lr=1e-3, lambda_cls=1.0):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    best_val = 999

    for ep in range(1, epochs+1):
        model.train()
        loss_total = 0

        for imgs, masks, pres in tqdm(train_loader, desc=f"Epoch {ep}"):
            imgs, masks, pres = imgs.to(device), masks.to(device), pres.to(device)

            seg_logits, cls_logits = model(imgs)

            loss_seg = bce(seg_logits, masks) + dice_loss(seg_logits, masks)
            loss_cls = bce(cls_logits, pres)

            loss = loss_seg + lambda_cls * loss_cls

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_total += loss.item()

        print(f"Train loss: {loss_total/len(train_loader):.4f}")

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks, pres in val_loader:
                imgs, masks, pres = imgs.to(device), masks.to(device), pres.to(device)
                seg_logits, cls_logits = model(imgs)

                loss = (
                    bce(seg_logits, masks)
                    + dice_loss(seg_logits, masks)
                    + lambda_cls*bce(cls_logits, pres)
                )
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Val loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "unet_multitask_best.pth")
            print("✓ Saved best model")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    IMAGES_DIR = os.path.normpath(os.path.join(script_dir, "..", "..", "train_buah"))
    COCO_JSON = os.path.normpath(os.path.join(script_dir, "..", "..", "train_buah", "_annotations.coco.json"))

    CLASSES = [
        "Fruits","Apple","Banana","Grapes","Kiwi",
        "Mango","Orange","Pineapple","Strawberry","Sugarapple","Watermelon"
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    dataset = CocoSegDataset(IMAGES_DIR, COCO_JSON, classes=CLASSES, transform=transform)
    n = len(dataset)
    train_size = int(0.9*n)
    val_size = n - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    model = UNetMultiTask(num_classes=len(CLASSES))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(train_loader, val_loader, model, device, epochs=80, lr=1e-3)
