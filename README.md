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
- w/o distortions(ECA NFnet-l0, VisionTransformer-base)
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

## Inference
`python inference.py`