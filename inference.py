import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

from models import *
from data import prepare_test_loader

def inference(model, data_loader, device):
    preds = []
    model.eval()
    pbar = tqdm(data_loader, total=len(data_loader), position=0, leave=True)
    for images in pbar:
        images = images.to(device)
        preds.extend(model(images).detach().cpu().numpy())
    return preds

def load_model_inference(path, df, model, test_loader, model_name, device):
    preds = []
    df = df.copy()
    for i in range(5):
        file_name = f"{model_name}_{i}.pth"
        print(f"{file_name} start")
        model.load_state_dict(torch.load(f"{path}/{file_name}", map_location=device))
        with torch.no_grad():
            preds += [inference(model, test_loader, device)]
    preds = np.mean(preds, axis=0)
    outcomes = pd.concat([df['ImageID'], pd.DataFrame(preds)], axis=1)

    return outcomes

def main():

    df = pd.read_csv('./info.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader = prepare_test_loader(df)


    PATH_NFNET = "./trained_models/nfnet/eca_nfnet_l0"
    MODEL_NFNET = MaskNFNet(model_arch='eca_nfnet_l0', n_classes=18)
    PATH_EFFNET = "./trained_models/efficient/tf_efficientnet_b3_ns"
    MODEL_EFFNET = MaskEfficientNet(model_arch='tf_efficientnet_b3_ns', n_classes=18)
    PATH_ViT = "./trained_models/ViT/vit_base_patch16_384"
    MODEL_ViT = MaskViT(model_arch='vit_base_patch16_384', n_classes=18)

    nfnet_outcomes = load_model_inference(PATH_NFNET, df, MODEL_NFNET, 'eca_nfnet_l0', device)
    effnet_outcomes = load_model_inference(PATH_EFFNET, df, MODEL_EFFNET, 'tf_efficientnet_b3_ns', device)
    vit_outcomes = load_model_inference(PATH_ViT, df, MODEL_ViT, 'vit_base_patch16_384', device)

    final_preds = (effnet_outcomes_b3.drop('ImageID', axis=1)*0.6 + nfnet_outcomes.drop('ImageID', axis=1)*0.3 +  vitnet_outcomes.drop('ImageID', axis=1)*0.1).to_numpy()
    final_preds = softmax(final_preds).argmax(1)

    submit = pd.DataFrame({'ImageID': df['ImageID'].values, 'ans': final_preds})
    submit.to_csv('./submission.csv', index=False)

if __name__ == "__main__":
    main()
    
