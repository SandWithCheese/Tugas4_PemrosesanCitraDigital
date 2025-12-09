import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import torchvision.transforms.functional as TF

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

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

        # classification head
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

        # decoder
        d4 = self.up4(bott)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        seg_logits = self.seg_head(d1)

        return seg_logits, cls_logits


def infer(image_path, model_path, class_names, img_size=(256,256)):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model = UNetMultiTask(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # load image
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    img_input = transform(img).unsqueeze(0).to(device)

    # run model
    with torch.no_grad():
        seg_logits, cls_logits = model(img_input)

    seg_pred = torch.sigmoid(seg_logits)[0].cpu().numpy()     # (C, H, W)
    cls_pred = torch.sigmoid(cls_logits)[0].cpu().numpy()     # (C,)

    detected = [class_names[i] for i, p in enumerate(cls_pred) if p > 0.5]

    for name in detected:
        print(f"- {name}")

    seg_mask = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

    colors = [
        (255, 0, 0), (0,255,0), (0,0,255),
        (255,255,0), (255,0,255), (0,255,255),
        (128,0,0), (0,128,0), (0,0,128),
        (128,128,0), (128,0,128)
    ]

    for c in range(seg_pred.shape[0]):
        mask = (seg_pred[c] > 0.5).astype(np.uint8)
        seg_mask[mask == 1] = colors[c]

    seg_mask = cv2.resize(seg_mask, (orig_w, orig_h))

    img_np = np.array(img)
    overlay = cv2.addWeighted(img_np, 0.7, seg_mask, 0.3, 0)

    out_path = "result_overlay.png"
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"\nMask overlay disimpan ke: {out_path}")
    return detected, out_path
