# Step 5 pre-grid smoke test
ultralytics 8.3.246, device=mps
Live data config: `phase3/dataset_aligned/data.yaml` (confirmed has `ir:` key)

## cbam_rgvqf_proposed (six signals)
- model type: `RGBTDetectionModel` (is RGBTDetectionModel: True) -- **PASS**
- CBAM/CMAAM module count: 7 -- **PASS**
- RGVQFFusion instance count: 3 -- **PASS**
- RGVQFFusion.forward() calls: 1413 (first call shapes: {'rgb_in': (4, 128, 96, 120), 'ir_in': (4, 128, 96, 120), 'out': (4, 128, 96, 120)}) -- **PASS**
- QNode (`_rgvqf_circuit`) calls: 1413 -- **PASS**
- forward-input structure (first 3 calls): [{'kind': 'single_tensor', 'shape': (1, 3, 256, 256), 'via_dict': False}, {'kind': 'single_tensor', 'shape': (1, 3, 32, 32), 'via_dict': False}, {'kind': 'tuple/list', 'len': 2, 'shapes': [(4, 3, 768, 960), (4, 3, 768, 960)], 'via_dict': True}] -- **PASS**
- loss finite over 35 steps: True -- **PASS**

**cbam_rgvqf_proposed verdict: PASS**

## cbam_add (the config that triggered the alarm; reduced check)
- forward-input structure (first 3 calls): [{'kind': 'single_tensor', 'shape': (1, 3, 256, 256), 'via_dict': False}, {'kind': 'single_tensor', 'shape': (1, 3, 32, 32), 'via_dict': False}, {'kind': 'tuple/list', 'len': 2, 'shapes': [(4, 3, 768, 960), (4, 3, 768, 960)], 'via_dict': True}] -- **PASS**
- CBAM/CMAAM module count: 7 -- **PASS**
- (RGVQFFusion instances / QNode calls: N/A for fusion_mode="add", not checked)

**cbam_add verdict: PASS**

## Overall verdict: PASS -- grid is safe to launch
