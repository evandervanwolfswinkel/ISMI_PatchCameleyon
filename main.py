import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from hdf5 import CustomDataset
import sys
import shutil
import time
import helper_funcs

torch.set_num_threads(4)
BEST_VAL_LOSS = 999999999999999999999
torch.manual_seed(42)
np.random.seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_checkpoint(state, is_best, last_checkpoint, best_checkpoint):
    torch.save(state, last_checkpoint)
    if is_best:
        shutil.copyfile(last_checkpoint, best_checkpoint)

def train(train_loader: DataLoader, val_loader: DataLoader, last_checkpoint:str, best_checkpoint:str, epochs) -> None:
    for epoch in range(epochs):
        for batch, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            model.train()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                val_loss = val(val_loader, last_checkpoint, best_checkpoint, epoch)
                print("[Epoch {}, Batch {}], loss {:.3f}, val loss {:.3f}".format(epoch, batch, loss.cpu().item(), val_loss))

def val(val_loader: DataLoader, last_checkpoint:str, best_checkpoint:str, epoch):
    global BEST_VAL_LOSS
    with torch.no_grad():
        for batch, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)
            model.eval()
            y_pred = model(x)
            loss = loss_fn(y_pred, y).cpu()
            is_best = loss.item() < BEST_VAL_LOSS
            BEST_VAL_LOSS = min(loss.item(), BEST_VAL_LOSS)
            
            save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()}, is_best, last_checkpoint, best_checkpoint)
            
    return loss.item()

if __name__ == '__main__':
    data_dir = sys.argv[1]
    slurm_id = sys.argv[2]
    submissions_folder = sys.argv[3]
    checkpoints_folder = sys.argv[4]
    backbone_name = sys.argv[5]
    extra_augmentations = bool(sys.argv[6])
    freeze = bool(sys.argv[7])
    pretrained = bool(sys.argv[8])

    epochs = int(sys.argv[9])
    run_name = sys.argv[10]

    best_checkpoint = checkpoints_folder + slurm_id + "_best_" + run_name + ".pth.tar"
    last_checkpoint = checkpoints_folder + slurm_id + "_last_" + run_name + ".pth.tar"

    model = helper_funcs.get_backbone(backbone_name=backbone_name, pretrained=pretrained, freeze=freeze) # Get model
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()

    import torchvision.transforms as T
    hoi = T.Compose(
        [ 
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) # this transform is a normalization recommended when using a model pretrained on imagenet
        ] 
    )

    if extra_augmentations:
        train_dataset = CustomDataset("{}/camelyonpatch_level_2_split_train_x.h5".format(data_dir),
                                "{}/camelyonpatch_level_2_split_train_y.h5".format(data_dir), transform=helper_funcs.get_augmentations(extra_augmentations))
    else:
        train_dataset = CustomDataset("{}/camelyonpatch_level_2_split_train_x.h5".format(data_dir),
                                "{}/camelyonpatch_level_2_split_train_y.h5".format(data_dir), transform=hoi) #transform=helper_funcs.get_augmentations(extra_augmentations)
    val_dataset = CustomDataset("{}/camelyonpatch_level_2_split_valid_x.h5".format(data_dir),
                                "{}/camelyonpatch_level_2_split_valid_y.h5".format(data_dir), transform=helper_funcs.get_augmentations(False))
    test_dataset = CustomDataset("{}/camelyonpatch_level_2_split_test_x.h5".format(data_dir),
                                 "{}/camelyonpatch_level_2_split_test_y.h5".format(data_dir), transform=helper_funcs.get_augmentations(False))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    t0 = time.time()

    train(train_loader, val_loader, last_checkpoint, best_checkpoint, epochs)
    print("{} seconds elapsed after training {} epochs.".format(time.time() - t0, epochs))
