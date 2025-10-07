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

<!-- Provide a longer summary of what this model is. -->

{{ model_description | default("Through deep learning regression, the model aims to estimate non-invasive hemoglobin from smartphone lip images  ", true) }}

- **Developed by:** {{ developers | default("[More Information Needed]", true)}}
- **Model type:** {{ model_type | default("Model_use_case.image_classification", true)}}
- **License:** {{ license | default("From[ qualcomm]([url](https://huggingface.co/qualcomm/MobileNet-v3-Small)),  the licenses are [original implementation of MobileNet-v3-Small]([url](https://github.com/pytorch/vision/blob/main/LICENSE)) and [compiled assets for on-device deployment]([url](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf))", true)}}
- **Finetuned from model [optional]:** {{ base_model | default("[More Information Needed]", true)}}

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** {{ repo | default("[More Information Needed]", true)}}
- **Paper [optional]:** {{ paper | default("[More Information Needed]", true)}}
- **Demo [optional]:** {{ demo | default("[More Information Needed]", true)}}

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

{{ direct_use | default("[More Information Needed]", true)}}

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

{{ out_of_scope_use | default("[More Information Needed]", true)}}

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

{{ bias_risks_limitations | default("[More Information Needed]", true)}}

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

{{ bias_recommendations | default("Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.", true)}}

## How to Get Started with the Model

Use the code below to get started with the model.

{{ get_started_code | default("[More Information Needed]", true)}}

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

{{ training_data | default("[More Information Needed]", true)}}

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

{{ preprocessing | default("[More Information Needed]", true)}}


#### Training Hyperparameters

- **Training regime:** {{ training_regime | default("[More Information Needed]", true)}} <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

{{ speeds_sizes_times | default("[More Information Needed]", true)}}

## Evaluation

Mean Absolute Error (MAE):   1.8941 g/dL
Root Mean Squared Error:     2.5412 g/dL
R² Score:                    0.3665
Median Absolute Error:       1.3670 g/dL
Std of Absolute Error:       1.7222 g/dL
Min / Max Abs Error:         0.1041 / 6.3843 g/dL

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

{{ model_examination | default("[More Information Needed]", true)}}

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** {{ hardware_type | default("[More Information Needed]", true)}}
- **Hours used:** {{ hours_used | default("[More Information Needed]", true)}}
- **Cloud Provider:** {{ cloud_provider | default("[More Information Needed]", true)}}
- **Compute Region:** {{ cloud_region | default("[More Information Needed]", true)}}
- **Carbon Emitted:** {{ co2_emitted | default("[More Information Needed]", true)}}

**BibTeX:**

{{ citation_bibtex | default("[More Information Needed]", true)}}

**APA:**

{{ citation_apa | default("[More Information Needed]", true)}}

## Model Card Contact

{{ model_card_contact | default("[More Information Needed]", true)}}
