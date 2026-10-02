# Color-Invariant Saree Design Recognition

A PyTorch image-retrieval system for recognizing the same saree design across different colors.

Given a saree image, the system produces a compact embedding and retrieves visually related designs from a reference gallery. It also supports pairwise verification: whether two saree images are likely to contain the same design.

The main challenge is **color invariance**: the model should focus on design structure, motifs, borders, and texture rather than treating color as the identity of the saree.

---

## Approach Note

We train a ResNet-50 embedding model with GeM pooling and a 256-D normalized output using supervised contrastive learning. Each training image receives two independently color-augmented views, encouraging the same design to remain close despite color changes. Images are grouped using perceptual hashing before splitting to reduce near-duplicate leakage. Retrieval uses cosine similarity, while verification uses a validation-calibrated similarity threshold.

---

## Datasets

Two datasets were used:

- **DeepLure Saree Corpus**
- **Indian Saree Patterns**

The datasets were cleaned and combined into a single manifest.

The source folder/category labels were retained as metadata but were **not treated as design identities**, because the assignment requires recognition of individual saree designs rather than broad categories such as Banarasi, Bandhani, Ikat, or Pichwai.

---

## Dataset Preparation

The final combined dataset contained **1,628 usable images** after removing 5 exact duplicates.

Preprocessing included:

- Corrupt/unreadable images removed
- Images with a shorter side below 128 pixels removed
- Exact byte-identical duplicates removed using MD5
- EXIF orientation applied
- Images converted to RGB
- Transparent PNG backgrounds converted to black
- Original folder labels retained only as metadata

### Near-Duplicate Grouping

To reduce train/test leakage, images were grouped using a 64-bit grayscale difference hash (dHash).

Images within a Hamming distance of **2** were grouped together, while very low-texture images were handled separately.

Entire groups were assigned to only one split.

Final split:

| Split | Images |
|---|---:|
| Train | 1,137 |
| Validation | 164 |
| Test | 327 |

The split was approximately 70/10/20 at the group level using a fixed seed.

---

## Model Architecture

The final model is intentionally simple:

```text
RGB Saree Image
      |
      v
ResNet-50
(ImageNet pretrained)
      |
      v
GeM Pooling
      |
      v
256-D Projection
      |
      v
L2 Normalization
      |
      v
256-D Saree Design Embedding
Components
Backbone: ResNet-50
Initialization: ImageNet pretrained weights
Pooling: Generalized Mean (GeM)
Embedding dimension: 256
Embedding normalization: L2
Loss: Supervised Contrastive Loss
Optimizer: AdamW
Scheduler: OneCycleLR
Epochs: 10
Batch size: 96
Backbone learning rate: 1e-4
Projection-head learning rate: 1e-3
Color-Invariance Strategy

Color invariance is the central part of the system.

During training, each image produces two independently augmented views. The augmentation pipeline includes random color transformations so that the same design can appear with substantially different colors.

The supervised contrastive objective encourages embeddings of the same design to remain close while separating embeddings belonging to different designs.

This makes the learned representation less dependent on raw RGB appearance and more dependent on structural characteristics such as:

Motifs
Borders
Repeated patterns
Texture
Geometric arrangement

No explicit color-removal preprocessing is required.

Evaluation Protocol

The test set contains 327 images.

The evaluation is performed using an image-embedding retrieval setup.

For identification:

A query image is embedded.
Gallery images are embedded.
Cosine similarity is calculated between the query and gallery embeddings.
Gallery images are ranked by similarity.
Top-1, Top-5, and Top-10 retrieval accuracy are reported.

For verification:

Positive pairs represent the same design.
Negative pairs represent different designs.
ROC-AUC, EER, and TPR at 1% FPR are reported.

The test set is kept separate from training and validation.

Synthetic Color-Invariance Evaluation

Because the datasets do not provide enough naturally labeled pairs of the same design in different colors, a controlled synthetic recoloring evaluation was used.

For each test image, a recolored version is generated using a different color mapping while preserving the underlying spatial design structure.

This directly tests whether the embedding remains stable when the appearance changes primarily because of color.

This synthetic evaluation is the main quantitative evidence for color robustness.

Results
Identification

On the held-out test set with synthetic recoloring:

Metric	Result
Top-1	42.20%
Top-5	94.19%
Top-10	98.17%

The model reaches very high Top-5 and Top-10 retrieval while Top-1 remains more challenging because multiple sarees can contain visually similar motifs.

Verification

Synthetic recoloring versus color-matched negative pairs:

Metric	Result
ROC-AUC	0.9944
EER	3.06%
TPR @ 1% FPR	94.50%

Synthetic recoloring versus random negative pairs:

Metric	Result
ROC-AUC	0.9945
EER	3.21%
TPR @ 1% FPR	94.50%

These results show that the learned embedding remains highly similar for the same design after the controlled color transformation.

Natural-Image Identification

On naturally captured test images where the design identity is available, but color variation is not explicitly labeled:

Metric	Result
Top-1	97.81%
Top-5	100%
Top-10	100%

This demonstrates strong general retrieval performance on the available natural test data.

However, this result should not be interpreted as a direct measurement of real-world color invariance, because the dataset does not contain enough naturally labeled same-design/different-color pairs.

Color-Augmentation Ablation

To measure the effect of color augmentation, the same architecture was trained without color augmentation.

With Color Augmentation
Metric	Result
Test Top-1	42.20%
Test Top-5	94.19%
Synthetic recolor ROC-AUC	0.9944
Without Color Augmentation
Metric	Result
Test Top-1	25.69%
Test Top-5	71.87%
Synthetic recolor ROC-AUC	0.9353

The controlled ablation shows that color augmentation materially improves retrieval and robustness to synthetic recoloring.

ImageNet Baseline

A direct ImageNet-pretrained ResNet-50 embedding baseline was also evaluated without the task-specific contrastive training.

Recorded results:

Metric	ImageNet Baseline
Top-1	28.13%
Top-5	70.64%
Top-10	83.79%
Synthetic recolor ROC-AUC	0.8898

The task-specific 256-D embedding therefore provides a stronger representation for this saree retrieval task than using the raw ImageNet representation directly.

Efficiency
Property	Value
Backbone	ResNet-50
Parameters	24.03M
Embedding size	256-D
Input	224 × 224
FLOPs	~8.18 GFLOPs/image
Batch-1 latency	~5.69 ms
Batch-32 latency	~84.89 ms
Device	NVIDIA T4
Precision	FP32

Latency was measured during inference on the GPU.

The compact 256-D embedding makes gallery search relatively inexpensive compared with storing and comparing full backbone features.

External Real-Image Sanity Check

A small external test was performed using 10 manually collected same-design/different-color pairs that were not part of the training or test datasets.

Because the sample size is very small, this is treated only as a sanity check rather than a benchmark.

Results:

Metric	Result
Number of same-design pairs	10
Top-1 identification	50%
Top-5 identification	70%
Top-10 identification	100%
Same-design verification TPR	100%
Color-matched negative ROC-AUC	0.92
Cross-design negative ROC-AUC	0.784

The 95% confidence intervals are wide because only 10 positive pairs were available.

Therefore, these results are useful as an external sanity check but should not be treated as a statistically reliable estimate of real-world performance.

Additional Real-Image Sanity Check

A separate manual check was performed on six additional unseen saree images.

All 15 possible image pairs were manually checked, and the fixed verification threshold produced:

15/15 manually confirmed pairwise decisions correct.

This is only an additional qualitative sanity check and is not included as a formal benchmark because the images were not part of a predefined labeled evaluation set.

Results & Demo
## Results & Demo

Screenshot 2026-10-02 020857.png


![Result](./Screenshot%202026-10-02%20025030.png)

![Result](./Screenshot%202026-10-02%20025117.png)

![Result](./Screenshot%202026-10-02%20025130.png)

![Result](./Screenshot%202026-10-02%20025138.png)

![Result](./Screenshot%202026-10-02%20025145.png)

![Result](./Screenshot%202026-10-02%20025156.png)

![Result](./Screenshot%202026-10-02%20025213.png)

![Result](./Screenshot%202026-10-02%20025222.png)

The trained model supports three main operations:

Build a Gallery
python infer.py build \
  --checkpoint runs/resnet50/best.pt \
  --manifest work_strict/manifest.csv \
  --split test \
  --out gallery_test.pt
Query the Gallery
python infer.py query \
  --checkpoint runs/resnet50/best.pt \
  --gallery gallery_test.pt \
  --image path/to/query.jpg \
  --topk 5
Verify Two Images

First calibrate the verification threshold:

python infer.py calibrate \
  --checkpoint runs/resnet50/best.pt \
  --out threshold.json

Then verify two images:

python infer.py verify \
  --checkpoint runs/resnet50/best.pt \
  --a image_a.jpg \
  --b image_b.jpg \
  --threshold-file threshold.json

The validation calibration produced an EER-based threshold of approximately:

0.4526

This threshold was calibrated using synthetic-recolor positives and color-matched negatives from the validation split.

It should therefore be considered a starting threshold rather than a production threshold for a much larger real-world dataset.

Project Structure
Ml project/
│
├── common.py
├── prepare_data.py
├── train.py
├── evaluate.py
├── eval_pairs.py
├── infer.py
├── requirements.txt
├── README.md
│
├── work_strict/
│   ├── manifest.csv
│   ├── report.json
│   └── group_check.png
│
├── runs/
│   └── resnet50/
│       ├── best.pt
│       └── history.json
│
├── eval_resnet50/
│   └── results.json
│
├── eval_real_pairs/
│   ├── results.json
│   └── failures.csv
│
└── real_pairs/
    └── real_pairs.csv
Reproducibility

The dataset preparation, training, evaluation, and inference pipelines are implemented as Python scripts.

Example training command:

python train.py \
  --manifest work_strict/manifest.csv \
  --out runs/resnet50 \
  --backbone resnet50 \
  --dim 256 \
  --epochs 10 \
  --bs 96 \
  --workers 2

Example evaluation command:

python evaluate.py \
  --manifest work_strict/manifest.csv \
  --checkpoint runs/resnet50/best.pt \
  --backbone resnet50 \
  --out eval_resnet50 \
  --workers 2
Pretrained and External Resources

The model uses the standard ImageNet-pretrained ResNet-50 weights provided through torchvision.

No paid API or proprietary inference service is required.

Main dependencies include:

PyTorch
torchvision
NumPy
Pandas
Pillow
scikit-learn
tqdm
Limitations

The main limitation is the availability of naturally labeled same-design/different-color examples.

The datasets provide many saree images, but they do not provide enough explicit identity labels showing the same design photographed or manufactured in multiple colors.

Therefore:

Synthetic recoloring is used for the main controlled color-invariance evaluation.
The external real-image experiment contains only 10 same-design/different-color pairs.
Real-world results should be validated on a larger dataset containing explicit design identities and multiple colorways.
The reported verification threshold is calibrated on synthetic validation pairs.
The test set is relatively small and comes from a single data split and training seed.

These limitations mean the system demonstrates color robustness, but further real-world data would be required before claiming production-level color invariance.

Final Summary

This project implements a practical color-robust saree design retrieval system using a ResNet-50 backbone, GeM pooling, supervised contrastive learning, and a compact 256-D embedding.

The key design choice is independent color augmentation during contrastive training, which explicitly encourages the model to recognize design structure independently of color.

The resulting system supports:

Color-robust image retrieval
Top-K gallery search
Pairwise design verification
Synthetic color-invariance evaluation
External real-image sanity testing
Lightweight 256-D embeddings
GPU inference suitable for a practical deployment pipeline

The results should be interpreted together with the dataset limitations, particularly the limited availability of naturally labeled same-design/different-color pairs.
