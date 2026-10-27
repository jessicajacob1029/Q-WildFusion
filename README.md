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


Dataset

Experiments use LLVIP: aligned visible and infrared image pairs captured at night, with one class (person).

The dataset is not included in this repo. Download it and arrange it as YOLO-format paired folders:

phase3/dataset_aligned/
  images/rgb/{train,val}
  images/ir/{train,val}
  labels/{train,val}
  data.yaml

Update the absolute paths in data.yaml and in the script defaults to match your machine.
