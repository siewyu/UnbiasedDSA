---
# Template is taken from Hugging Face: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards

language:
- en
library_name: torch
license: mit
datasets:
- images/
- data/labels.csv
- data/meta.csv
- data/combined_data.csv
tags:
- regression
- healthcare
- hemoglobin
- lip-images
- pytorch
metrics:
- mae
  
model_name: UnbiasedDSA-Hemoglobin-Predictor
model_type: MobileNetV3-Small (pre-trained on ImageNet)
pipeline_tag: image-regression
inference: true
  
---

## Model Details

### Model Description

Through deep learning regression, the model aims to estimate non-invasive hemoglobin from smartphone lip images 

- **Developed by:** Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q.V., & Adam, H. (2019). [https://doi.org/10.48550/arXiv.1905.02244]
- **Model type:** Model_use_case.image_classification
- **License:** From[ qualcomm]([url](https://huggingface.co/qualcomm/MobileNet-v3-Small)),  the licenses are [original implementation of MobileNet-v3-Small]([url](https://github.com/pytorch/vision/blob/main/LICENSE)) and [compiled assets for on-device deployment]([url](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf))

### Model Sources 

- **Repository:** https://huggingface.co/qualcomm/MobileNet-v3-Small

## Uses

### Direct Use

A machine learning pipeline for estimating hemoglobin levels from lip images using computer vision and deep learning.

## Bias, Risks, and Limitations

Potential Biases
- Skin tone bias - Hemoglobin estimation accuracy typically degrades on darker skin tones due to reduced light penetration and lower signal-to-noise ratio. The dataset used appears to have limited representation of Fitzpatrick V-VI tones.
- Device bias - HEIC images have complete metadata while JPEGs often have stripped EXIF data, potentially creating device-specific performance disparities.
- Gender/age bias - unable to access due to missing demographic labels

### Recommendations

The model includes fairness checks across inferred skin tone proxies, gender, and device types.
Performance consistency is monitored to ensure no demographic group is disproportionately
affected. Future work includes bias mitigation using domain adaptation and adversarial
regularization.

## How to Get Started with the Model [Duplicated from README.md]

Use the code below to generate predictions on images.

Generate predictions on images:
```bash
python inference.py --images ../data/images --weights ../weights/best_model.pt --meta ../data/meta.csv --out ../data/predictions.csv --img_size 192
```

Output format (`predictions.csv`):
```csv
image_id,predicted_hgb
HgB_10.7gdl_Individual01,10.8
HgB_12.0gdl_Individual02,11.9
...
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

**Step 1:** Place all lip images in data folder

**Step 2:** By default, `extractdata.py` reads from `../data/images`. Update the path only if your images are stored elsewhere.
```python
IMAGE_FOLDER = r"..\data\images"
```

**Step 3:** Run data extraction:
(This has been extracted already under the data folder but use extractdata.py for other datasets)
```bash
cd code
python extractdata.py
```

This generates:
- `data/labels.csv` - HgB values extracted from filenames
- `data/meta.csv` - Camera metadata (device, ISO, exposure, etc.)
- `data/combined_data.csv` - Complete dataset

### Training Procedure

Please refer to README.md file for the full details on our entire process. 

```bash
cd code
python train.py --images_dir ../data/images --labels_csv ../data/labels.csv --meta_csv ../data/meta.csv --output_dir ../data --batch_size 2 --epochs 40 --img_size 192 --use_metadata --backbone mobilenet_v3_small
```

**Training options:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 2)
- `--lr`: Learning rate (default: 1e-4)
- `--images_dir`: Path to images (default: ../data/images)
- `--labels_csv`: Path to labels (default: ../data/labels.csv)
- `--output_dir`: Output directory for model (default: ../weights)

**Expected output:**
- Progress bars showing training/validation
- Epoch-by-epoch MAE metrics
- Best model saved to `weights/best_model.pt`

## Evaluation

Mean Absolute Error (MAE):   1.8941 g/dL
Root Mean Squared Error:     2.5412 g/dL
R² Score:                    0.3665
Median Absolute Error:       1.3670 g/dL
Std of Absolute Error:       1.7222 g/dL
Min / Max Abs Error:         0.1041 / 6.3843 g/dL

## Compute Efficiency

The model is designed to run efficiently on CPU-only systems. Training on 31 samples for 40
epochs takes under 15 minutes on a standard Intel i7 processor. Inference runs under 1 second per
image.


{{ model_card_contact | default("[More Information Needed]", true)}}


