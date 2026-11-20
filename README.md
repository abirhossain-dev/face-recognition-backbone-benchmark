# Face Recognition Backbone Benchmark (work in progress)

An ongoing study comparing pretrained CNN backbones (MobileNetV2, DenseNet, ResNet50, EfficientNetB0, NASNetMobile) for face recognition with an ArcFace-style angular-margin head on the Labeled Faces in the Wild (LFW) dataset.

> **Status: work in progress.** This started as MSc coursework (MSc IT with Data Analytics, University of the West of Scotland). I am publishing it openly, including its known weaknesses, while I rebuild the evaluation properly. I am not claiming that any backbone is better than another yet.

## What is in this repository

| Path | Contents |
|---|---|
| `notebooks/densenet169_transfer_learning.ipynb` | DenseNet169 (ImageNet weights, frozen base) with augmentation, early stopping and LR decay on a small 3-class set |
| `notebooks/nasnetmobile_lfw.ipynb` | NASNetMobile on LFW: dataset download, preprocessing and training |

## Method (summary)

- **Data:** LFW (`jessicali9530/lfw-dataset` on Kaggle).
- **Backbones:** ImageNet-pretrained, frozen, used as feature extractors.
- **Head:** ArcFace-style additive angular margin loss.
- **Training:** augmentation, early stopping, ReduceLROnPlateau, checkpointing.

## Known issues (being fixed)

I audited my own code before publishing. These issues mean the earlier cross-model numbers (including a "1.000 test accuracy for every model" chart) **should not be treated as valid results**, so I have left them (and the per-model curves from the same pipeline) out until the evaluation is fixed:

1. **Evaluation uses ground-truth labels.** The ArcFace model takes the true label as an input, so test predictions are made with label information in the forward pass. Correct inference should use only the embedding and cosine similarity to the class weights.
2. **One backbone is mislabelled.** In the five-model comparison, the "MobileNetV2" run instantiates `NASNetMobile`, so MobileNetV2 was never actually benchmarked.
3. **Single preprocessing function.** `mobilenet_v2.preprocess_input` is applied to every backbone, but DenseNet, ResNet50 and EfficientNet each expect their own preprocessing.
4. **Splits are small and not identity-disjoint**, and the DenseNet169 test set contains only 23 images, so the accuracy estimate is very noisy.
5. Identity filter threshold and comments disagree (`min_faces_per_person=7` vs "20").

## Planned work

- [ ] Fix the five issues above and re-run all five backbones with the correct model and preprocessing.
- [ ] Evaluate with the standard LFW verification protocol (6,000 pairs, 10-fold) and report ROC/AUC and TAR@FAR.
- [ ] Report parameters, model size and inference latency per backbone.
- [ ] Add a pinned `requirements.txt` and a runnable Colab link.

## Run it

1. Open a notebook in Google Colab or Jupyter.
2. Install TensorFlow/Keras, scikit-learn and matplotlib.
3. For the NASNet notebook, supply your own Kaggle API credentials (never commit them).

## Author

MD Abir Hossain, London, UK. MSc IT with Data Analytics (University of the West of Scotland).
