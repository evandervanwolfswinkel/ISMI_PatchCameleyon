import torch
from torch.utils.data import DataLoader
import numpy as np
from hdf5 import CustomDataset
import sys
from tqdm import tqdm
import helper_funcs


#torch.manual_seed(42)
np.random.seed(42)

def test(test_loader, submission_path, model):
    model.eval()
    model.to(device)
    acc = []
    i = 0
    with open(submission_path, "w") as f:
        f.write("case,prediction")
    f.close()
    with torch.no_grad():
        test_progress = tqdm(test_loader)
        for (x, y) in test_progress:
            x = x.to(device)
            model.eval()
            y_pred = model(x)
            for item in [["{},{}".format(str(idx + i), pred.cpu().item()),
                          torch.sum(torch.argmax(y_pred, dim=1) == y).item() / y_pred.shape[0]] for idx, pred in
                         enumerate(y_pred)]:
                aa = open(submission_path, "a")
                aa.write("\n")
                aa.write(item[0])
                acc.append(item[1])
                #test_progress.set_description("Test, test accuracy: {:.2f}".format(item[1]))
            aa.close()

            del x, y_pred, item
            i += 128
        print("Avg. Accuracy: {}".format(sum(acc)/len(acc)))

def get_loader(data_dir, transform=True):
    if transform:
        test_dataset = CustomDataset("{}/camelyonpatch_level_2_split_test_x.h5".format(data_dir),
                                "{}/camelyonpatch_level_2_split_test_y.h5".format(data_dir), transform= helper_funcs.get_augmentations_test(transform))
    else:
        test_dataset = CustomDataset("{}/camelyonpatch_level_2_split_test_x.h5".format(data_dir),
                                "{}/camelyonpatch_level_2_split_test_y.h5".format(data_dir), transform= helper_funcs.get_augmentations_test(False))

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    return test_loader

if __name__ == "__main__":
    data_dir = sys.argv[1]
    checkpoint_path = sys.argv[2]
    submission_path = sys.argv[3]
    backbone_name = sys.argv[4]
    TTA = sys.argv[5]

    test_loader = get_loader(data_dir, TTA)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model = helper_funcs.get_model(backbone_name=backbone_name, checkpoint_path=checkpoint_path, device=device)
    test(test_loader, submission_path, model)
