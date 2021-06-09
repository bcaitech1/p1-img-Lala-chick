# BoostCamp P-stage 1 Image Classification
## Install Requirements
`pip install -r requirements.txt`  

## Data Description
- train images: 2700
- test images: 1800 
- image size: 384 x 512
- classes: 18

## Performence
- Public LB: 0.7923
- Private LB: 0.7736

## Train
`python train.py --model [model_name]`  

### models
- tf_efficientnet_b3_ns  
- eca_nfnet_l0  
- vit_base_patch16_384  

### train augmentations
- with distortions(EfficientNet-b3)
```
A.Compose(
        [
            A.CenterCrop(cfg["img_size"], cfg["img_size"], p=1.0),
            A.HueSaturationValue(),
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.4),
                    A.GridDistortion(p=0.2),
                    A.IAAPiecewiseAffine(p=0.4),
                ],
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.3), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            A.Normalize(
                mean=[0.56019358, 0.52410121, 0.501457],
                std=[0.23318603, 0.24300033, 0.24567522],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ]
    )
```
- w/o distortions(ECA NFNet-l0, VisionTransformer-base)
```
A.Compose(
        [
            A.CenterCrop(cfg["img_size"], cfg["img_size"], p=1.0),
            A.HueSaturationValue(),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.3), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            A.Normalize(
                mean=[0.56019358, 0.52410121, 0.501457],
                std=[0.23318603, 0.24300033, 0.24567522],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ]
    )
```

- Custom CutMix(All models)
기존의 CutMix의 경우 bbox를 랜덤하게 선택하기 때문에 마스크/성별/나이를 모두 고려하는 bbox가 나오기 쉽지 않음.  
주어진 데이터셋의 특성상 이미지의 중앙에 얼굴이 위치하고 있기 때문에 2개의 이미지의 절반을 잘라 합치는 방식을 사용.  
이 때, 데이터셋의 분포에 따르면 60대 이상의 이미지가 압도적으로 적었기 때문에 60대 이미지의 양을 보완하고자 합치는 이미지 중 1개 이상은 60대의 이미지가 사용되도록 함.  
```
def custom_cutmix(data, target, old_data, old_target):
    indices = torch.randperm(data.size(0))
    shuffled_old_target = old_target[indices]

    size = data.size()
    W = size[2]

    new_data = data.clone()
    new_data[:, :, :, : W // 2] = old_data[indices, :, :, W // 2 :]

    Iam = 0.5
    targets = (target, shuffled_old_target, Iam)

    return new_data, targets

```
```
for step, (imgs, image_labels) in pbar:
    imgs = imgs.to(device).float()
    image_labels = image_labels.to(device).long()
    mix_decision = np.random.rand()
    if mix_decision < cfg["mix_prob"]:
        if old_step < old_length - 1:
            old_imgs, old_labels = next(old_iter)
            old_imgs = old_imgs.to(device).float()
            old_labels = old_labels.to(device).long()
            old_step += 1
        else:
            old_step = 1
            old_iter = iter(old_train_loader)
            old_imgs, old_labels = next(old_iter)
            old_imgs = old_imgs.to(device).float()
            old_labels = old_labels.to(device).long()
        imgs, image_labels = custom_cutmix(imgs, image_labels, old_imgs, old_labels)
```

## Inference
`python inference.py`

### Ensemble
학습된 EfficientNet, NFNet, VisionTransformer에 대해 각각 K-fold 앙상블을 진행한 뒤, 6:3:1의 비율로 앙상블