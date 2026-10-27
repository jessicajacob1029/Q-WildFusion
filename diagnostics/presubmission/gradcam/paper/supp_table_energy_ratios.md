Supplementary Table. CAM energy inside GT boxes relative to chance. Ratio = mean fraction of CAM energy inside GT boxes / mean fraction of image area covered by GT boxes (Chance); 1.0 = chance. All ratios are computed on maps upsampled to 768x960 with the same full-resolution GT box mask; Sum = the P3+P4+P5 maps summed after upsampling, so each image's Sum fraction is the energy-weighted average of its three per-level fractions. Within a row, P3/P4/P5 and Sum use the same images and the same Chance. Val set, 1740 images; Grad-CAM excludes images with no target prediction (conf>=0.25, IoU>=0.5 to a GT box), hence n < 1740 and Chance 0.0340 for the RGB-only model. A per-level Grad-CAM map that is entirely zero after ReLU counts as fraction 0 (counts in the CSV column 'empty P3/P4/P5'). Single-modality baselines: 'RGB features' / 'Thermal features' = their backbone (layers 4/6/10) in the same position as the fusion models' branch features; they have no Fused row.

| Method | Model | Layer | P3 | P4 | P5 | Sum | n images | Chance |
|---|---|---|---|---|---|---|---|---|
| Grad-CAM | Full model | RGB features | 1.09 | 1.23 | 0.41 | 1.03 | 1734 | 0.0333 |
| Grad-CAM | Full model | Thermal features | 2.53 | 1.30 | 1.96 | 1.26 | 1734 | 0.0333 |
| Grad-CAM | Full model | Fused | 1.97 | 1.31 | 1.83 | 1.27 | 1734 | 0.0333 |
| Grad-CAM | Full model | Neck output | 2.61 | 4.87 | 2.39 | 2.65 | 1734 | 0.0333 |
| Grad-CAM | Fusion baseline (CBAM + add) | RGB features | 1.12 | 1.19 | 0.96 | 1.03 | 1732 | 0.0333 |
| Grad-CAM | Fusion baseline (CBAM + add) | Thermal features | 1.83 | 1.66 | 1.78 | 1.38 | 1732 | 0.0333 |
| Grad-CAM | Fusion baseline (CBAM + add) | Fused | 1.81 | 1.67 | 1.90 | 1.35 | 1732 | 0.0333 |
| Grad-CAM | Fusion baseline (CBAM + add) | Neck output | 2.39 | 5.57 | 4.37 | 3.58 | 1732 | 0.0333 |
| Grad-CAM | Thermal-only YOLO11n | Thermal features | 2.43 | 1.79 | 1.89 | 1.68 | 1734 | 0.0333 |
| Grad-CAM | Thermal-only YOLO11n | Neck output | 1.90 | 7.30 | 3.58 | 3.24 | 1734 | 0.0333 |
| Grad-CAM | RGB-only YOLO11n | RGB features | 1.42 | 1.11 | 1.27 | 1.07 | 1588 | 0.0340 |
| Grad-CAM | RGB-only YOLO11n | Neck output | 1.43 | 6.69 | 2.16 | 2.83 | 1588 | 0.0340 |
| EigenCAM | Full model | RGB features | 1.01 | 1.14 | 1.34 | 1.13 | 1740 | 0.0333 |
| EigenCAM | Full model | Thermal features | 1.89 | 1.35 | 1.68 | 1.55 | 1740 | 0.0333 |
| EigenCAM | Full model | Fused | 2.20 | 1.42 | 1.82 | 1.69 | 1740 | 0.0333 |
| EigenCAM | Full model | Neck output | 1.48 | 1.68 | 1.92 | 1.79 | 1740 | 0.0333 |
| EigenCAM | Fusion baseline (CBAM + add) | RGB features | 0.99 | 0.85 | 0.90 | 0.89 | 1740 | 0.0333 |
| EigenCAM | Fusion baseline (CBAM + add) | Thermal features | 1.66 | 1.59 | 0.90 | 1.43 | 1740 | 0.0333 |
| EigenCAM | Fusion baseline (CBAM + add) | Fused | 1.43 | 1.17 | 1.01 | 1.21 | 1740 | 0.0333 |
| EigenCAM | Fusion baseline (CBAM + add) | Neck output | 1.31 | 1.73 | 1.98 | 1.81 | 1740 | 0.0333 |
| EigenCAM | Thermal-only YOLO11n | Thermal features | 1.99 | 1.39 | 1.29 | 1.49 | 1740 | 0.0333 |
| EigenCAM | Thermal-only YOLO11n | Neck output | 1.21 | 1.90 | 2.15 | 1.92 | 1740 | 0.0333 |
| EigenCAM | RGB-only YOLO11n | RGB features | 0.99 | 1.01 | 0.42 | 0.73 | 1740 | 0.0333 |
| EigenCAM | RGB-only YOLO11n | Neck output | 1.09 | 1.44 | 1.88 | 1.66 | 1740 | 0.0333 |
