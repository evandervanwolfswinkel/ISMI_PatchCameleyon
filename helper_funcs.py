import torchvision.models as models
import torch.nn as nn
import torch
import torchvision.transforms as T
import pretrainedmodels
from Net import Net1


def get_backbone(backbone_name, pretrained=False, freeze=True):
    if backbone_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)

        model.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1, bias=True),
            nn.Sigmoid()
        )
        if freeze:
            model.requires_grad_(False)
            model.fc.requires_grad_(True)
            

    elif backbone_name == "efficientnet_b0":
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=pretrained)
        model.classifier.fc = nn.Sequential(
            nn.Linear(in_features=1280, out_features=1, bias=True),
            nn.Sigmoid()
        )
        if freeze:
            model.requires_grad_(False)
            model.classifier.fc.requires_grad_(True)

    elif backbone_name == "densenet121":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1, bias=True),
            nn.Sigmoid()
        )
        if freeze:
            model.requires_grad_(False)
            model.classifier.requires_grad_(True)

    elif backbone_name == "convnext":
        model = models.convnext_base(pretrained=True).requires_grad_(False)
        model.classifier[2] = nn.Sequential(nn.Linear(1024, 1), nn.Sigmoid())
        #model = model.to(device)
        if freeze:
                model.requires_grad_(False)
                model.classifier[2].requires_grad_(True)

    elif backbone_name == "net":
        model = Net1()
        #model = model.to(device)
        # if freeze:
        #         model.requires_grad_(False)
        #         model.classifier[2].requires_grad_(True)
    
    return model

def get_model(backbone_name, checkpoint_path, device):
    model = get_backbone(backbone_name=backbone_name, pretrained=False, freeze=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device))["state_dict"])
    return model

def get_augmentations(all):
    if all:
        transforms = T.Compose(
            [
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), # this transform is a normalization recommended when using all pretrained pytorch models
                # T.RandomApply(transforms=[T.Grayscale()], p=0.5), # turning this on causes an error
                T.RandomApply(transforms=[T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)], p=0.5),
                # T.RandomApply(transforms=[T.RandomCrop(size=(64, 64))], p=0.5),
                T.RandomInvert(p=0.5),
                # T.RandomRotation(degrees=180),
                # T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomHorizontalFlip(p=0.5)
            ]
        )
    else:
        transforms = T.Compose(
            [ 
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) # this transform is a normalization recommended when using all pretrained pytorch models
            ] 
        )
    return transforms

def get_augmentations_test(all):
    if all:
        transforms = T.Compose(
            [
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), # this transform is a normalization recommended when using all pretrained pytorch models
                # T.RandomApply(transforms=[T.Grayscale()], p=0.5), # turning this on causes an error
                T.RandomApply(transforms=[T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)], p=0.5),
                # T.RandomApply(transforms=[T.RandomCrop(size=(64, 64))], p=0.5),
                # T.RandomInvert(p=0.5),
                # T.RandomRotation(degrees=180),
                T.RandomVerticalFlip(p=0.5),
                # T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5),
                T.RandomHorizontalFlip(p=0.5)
            ]
        )
    else:
        transforms = T.Compose(
            [ 
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) # this transform is a normalization recommended when using all pretrained pytorch models
            ] 
        )
    return transforms
