# WSLCD
The implementation of Beyond Pixel-level Annotation: Learning to Detect Changes With Image-Level Supervision. It includes the part of generating pixel-level pseudo labels using image-level labels.

<img src="https://github.com/mfzhao1998/WSLCD/blob/main/model.png" width="75%">

Flowchart of the proposed framework. It mainly includes three parts: double-branch Siamese network structure, MLER, and PCL. The double-branch network structure is used to extract features and generate CAMs and embeddings. In particular, the feature extractor is also a Siamese network and integrates the multiscale context information in CNN. MLER not only constrains the consistency of CAMs from different views but also makes them learn from each other based on saliency regions. PCL performs contrastive learning based on CAMs and embeddings so that the same class has a similar feature representation. Furthermore, we not only build intraview contrast, but also interview contrast.
## Illustration of code
1. Training weakly supervised models：train.py
2. Export pixel-level labels：infer.py


