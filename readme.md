# Brain Tumor Retrieval SNN Pytorch
The `Retrieval of brain MRI with tumor using contrastive loss based similarity on GoogLeNet encodings` reproduce using pytorch.
Refer to [Brain_Tumor_Retrieval matlab version](https://figshare.com/articles/software/Brain_Tumor_Retrieval/12911102)

- Do my best to reproduce the original paper code

## Download image data and Convert mat to jpeg
1. Download image data (MRI) from https://doi.org/10.6084/m9.figshare.1512427.v5
2. Put the entire data in a folder and run the code `preprocess.m`
> The .jpg image file will generate in `imageData` folder

## Stage 1, Train for classification
```bash
python train_classifier.py -d ./imageData --batch-size 20 --epoch 15
```
- After training, you should see the pth(GoogLeNet-Classification-fold{number}) files in folder

### Test
```bash
python test_classifier.py -d ./imageData -w GoogLeNet-Classification-fold{number}.pth --fold {fold_number}
```

## Stage 2, Train for similarity measurement
```bash
python train_cbir.py --stage1 GoogLeNet-Classification-fold{number}.pth -d ./imageData --batch-size 15 -i 600
```
- After training, you should see the pth(GoogleNet-CBIR-fold{number}) files in folder

### Test
```bash
python evaluate_siamese.py -d ./imageData -w GoogleNet-CBIR-fold{number}.pth GoogLeNet-Classification-fold{number}.pth --stage1 --fold {fold_number}
```