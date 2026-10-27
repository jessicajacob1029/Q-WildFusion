Q-WildFusion

Reliability-guided quantum fusion of RGB and thermal infrared images for night-time pedestrian detection.

Q-WildFusion is a dual-stream RGB + IR detector built on a custom-extended YOLO11n. It fuses the two modalities with a Reliability-Guided Variational Quantum Fusion (RGVQF) block. Cross-modal attention and a per-modality reliability estimate decide how much each stream contributes at every scale.

Architecture
Component	What it does	Code
Dual-stream backbone	Separate YOLO11n feature extractors for RGB and IR, with CBAM attention	phase3/code/model/rgbt_backbone_cbam.py, phase3/code/model/cbam.py
CMAAM	Cross-Modal Attention Aggregation Module: lets each modality attend to the other before fusion	phase3/code/model/attention/cmaam.py
REM	Reliability Estimation Module: scores how trustworthy each modality is for the current input	phase3/code/model/reliability/rem.py
RGVQF	Reliability-Guided Variational Quantum Fusion: a variational quantum circuit (entangled and non-entangled variants) that fuses features, weighted by REM	phase3/code/model/fusion/rgvqf.py, phase3/code/quantum_fusion.py, phase3/code/quantum_fusion_no_entangle.py
MQFP	Multi-scale quantum fusion across the detector's feature pyramid levels	phase3/code/model/rgbt_backbone_cbam.py
Composite loss	Detection loss plus a reliability term (L_rel, from REM's modality weights) and a quantum fusion term (L_q)	phase3/code/losses/composite.py
Classical baselines	Add, concat and quantum-inspired fusion for comparison	phase3/code/model/fusion/

Quantum circuits run on PennyLane through its PyTorch interface (lightning.qubit, adjoint differentiation).

Repository layout
phase1/                  VOC to YOLO label conversion
phase2/code/             Early fusion prototypes
phase3/code/             Main model, trainer, fusion modules, evaluation and Grad-CAM scripts
  model/                 Backbone, CBAM, CMAAM, REM, RGVQF and fusion variants
  losses/                Composite loss
  rgbt_trainer.py        Paired RGB+IR trainer and validator for Ultralytics
  train_ablation_study.py  Ablation grid entry point
diagnostics/             Validation checks, multi-seed runs, pre-submission reports
  presubmission/         Final run driver, Grad-CAM figures, result tables
*.py (root)              Inference, demo apps, result collection and evaluation helpers
*.txt (root)             Dataset split and file-matching lists
Dataset

Experiments use LLVIP: aligned visible and infrared image pairs captured at night, with one class (person).

The dataset is not included in this repo. Download it and arrange it as YOLO-format paired folders:

phase3/dataset_aligned/
  images/rgb/{train,val}
  images/ir/{train,val}
  labels/{train,val}
  data.yaml

Update the absolute paths in data.yaml and in the script defaults to match your machine.

Setup
bash
python -m venv .venv
source .venv/bin/activate
pip install torch ultralytics opencv-python numpy pennylane pennylane-lightning
# optional, for the demo apps
pip install fastapi uvicorn python-multipart streamlit

Training was done on an Apple M1 Pro using the mps device. CUDA and CPU also work through --device.

Training

Run the full ablation grid:

bash
python phase3/code/train_ablation_study.py --data phase3/dataset_aligned/data.yaml --epochs 50 --device mps

Run selected configurations:

bash
python phase3/code/train_ablation_study.py --only complete cbam_rgvqf cbam_add
Key	Configuration
no_cbam_add	Dual stream, additive fusion, no attention
cbam_add	CBAM + additive fusion
cbam_vqc	CBAM + plain VQC fusion
cbam_rgvqf	CBAM + RGVQF
cbam_rgvqf_no_entangle	CBAM + RGVQF without entangling gates
cbam_rgvqf_rem	CBAM + RGVQF + REM
cmaam_add	CMAAM + additive fusion
cmaam_rgvqf_rem	CMAAM + RGVQF + REM
cmaam_rgvqf_rem_single_scale	Same as above, single scale (MQFP ablation)
complete	Full model: CMAAM + REM + RGVQF + MQFP + composite loss

Multi-seed final runs use diagnostics/presubmission/run_final.py --configs <keys> --seeds 0 1 2 ....

Evaluation and visualisation
phase3/code/eval_filtered_map*.py, eval_area_ap.py, eval_medium_large.py: mAP overall and by object size
diagnostics/presubmission/gradcam/: paired RGB/IR Grad-CAM figures
collect_results.py, extract_three_models_results.py, plot_results.py: result tables and plots
Demo
bash
streamlit run app.py          # upload an image and view detections
uvicorn api:app --reload      # POST an image to /detect

Both apps expect trained weights. Update the weight path at the top of each file.

Limitations
LLVIP contains only night scenes, so day, fog and rain performance is not evaluated.
LLVIP has a single class, so results are reported for person only.
Citation

A paper describing this work is in preparation. Citation details will be added on publication.
