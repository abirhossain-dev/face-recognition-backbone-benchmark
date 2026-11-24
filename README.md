# Face Recognition Backbone Benchmark (leakage-free protocol)

A controlled comparison of pretrained CNN backbones for face verification with an ArcFace-style angular-margin head. Started as MSc coursework (MSc IT with Data Analytics, University of the West of Scotland); rebuilt after I found a flaw in my own original evaluation (see "Audit" below).

## Protocol
- **Training:** ImageNet-pretrained backbone fine-tuned end to end, ArcFace head (scale 30, margin 0.5), on a CelebA subset: 2,500 identities, 61,205 images, 128x128, horizontal flip, 8 epochs, AdamW + OneCycle, one seed.
- **Evaluation:** the standard LFW verification pairs (6,000 pairs, 10 folds), cosine similarity, threshold chosen on 9 folds and tested on the 10th. No labels are used at inference. AUC has a pair-level bootstrap 95% CI.
- **Code:** `code/train_arcface_colab.py` (run it on Google Colab with a T4 GPU); `code/train_arcface_colab_v2.py` loads a prebuilt CelebA subset instead of downloading CelebA.

## Results
| Backbone | Params (M) | Accuracy (10-fold) | AUC (95% CI) | TAR@FAR 0.1 | TAR@FAR 0.01 |
|---|---|---|---|---|---|
| MobileNetV2 | 2.88 | 0.771 ± 0.014 | 0.849 (0.839-0.859) | 0.616 | 0.349 |
| EfficientNet-B0 | 4.66 | 0.780 ± 0.024 | 0.856 (0.846-0.866) | 0.629 | 0.338 |
| DenseNet121 | 7.48 | 0.808 ± 0.019 | 0.893 (0.885-0.902) | 0.711 | 0.404 |
| ResNet50 | 24.56 | 0.735 ± 0.018 | 0.807 (0.796-0.817) | 0.530 | 0.191 |

Frozen ImageNet features with no face training scored about 0.69-0.70 accuracy (AUC 0.74-0.77) on the same pairs, so fine-tuning on faces matters.
Ranking under this budget: DenseNet121 > EfficientNet-B0 ~ MobileNetV2 > ResNet50. ResNet50's AUC interval does not overlap MobileNetV2's, so it is clearly lowest here; I did not tune it, so this says nothing about ResNet50 in general (it is the largest model and may need a longer schedule or a different learning rate).

**Hardware note:** MobileNetV2, EfficientNet-B0 and DenseNet121 were trained on a Colab T4. ResNet50 was trained afterwards on a Mac (Apple GPU via PyTorch MPS) with the same data, hyperparameters and evaluation, because the Colab run was blocked by a Google Drive download quota (`code/train_arcface_colab_v2.py` can load a prebuilt subset). Training times are therefore not comparable across rows, so I have omitted them from the table.

## Audit of my original evaluation
My MSc dissertation reported 1.000 accuracy for every model. That was wrong: the ArcFace layer took the true label as an input at test time, so predictions used label information, and the run labelled "MobileNetV2" actually instantiated NASNetMobile. I kept the notebooks in `notebooks/` (outputs cleared) for transparency and replaced the evaluation with the protocol above.

## Limitations
- CelebA and LFW may share identities (CelebA identities are anonymised, so overlap cannot be removed by name).
- One seed; pair-level (not identity-level) bootstrap; only 2,500 training identities and 8 epochs, far below standard face-recognition training sets; no comparison with face-specific pretrained models.
- Results are a controlled small-budget comparison, not a state-of-the-art claim.

## Next steps
More seeds, identity-level intervals, larger training data, more benchmarks (CFP-FP, AgeDB, IJB), robustness to blur and low resolution, latency on common hardware.

## Author
MD Abir Hossain, London, UK. MSc IT with Data Analytics (University of the West of Scotland). MIT licence.
