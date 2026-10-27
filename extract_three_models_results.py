import os
import shutil
import random
import cv2
import numpy as np
import torch
import zipfile
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

base_dir = "/Users/test/Q-WildFusion"
models = {
    "before_cbam": "runs/detect/ablation_pilot/abl_no_cbam_add/weights/best.pt",
    "after_cbam": "runs/detect/ablation_pilot/abl_cbam_no_vqc/weights/best.pt",
    "after_cbam_and_vqc": "runs/detect/ablation_pilot/abl_cbam_vqc_proposed/weights/best.pt"
}

images_rgb_dir = os.path.join(base_dir, "phase3/dataset_aligned/images/rgb/val")
images_ir_dir = os.path.join(base_dir, "phase3/dataset_aligned/images/ir/val")
output_dir = os.path.join(base_dir, "comparative_results_extracted")

class YOLOConfidenceTarget:
    def __call__(self, model_output):
        scores = None
        if isinstance(model_output, tuple):
            out = model_output[1]
            if isinstance(out, dict):
                scores = out.get("scores")
        if scores is None and torch.is_tensor(model_output):
            if model_output.ndim == 3:
                scores = model_output[..., 4]
        if scores is None:
            raise ValueError("Could not extract YOLO scores from model output")
        return scores.max()

def run_predictions():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # gather files
    all_files = [f for f in os.listdir(images_rgb_dir) if f.endswith('.jpg')]
    sampled_files = random.sample(all_files, min(len(all_files), 60))
    
    for model_name, model_path in models.items():
        print(f"\\n--- Processing Model: {model_name} ---")
        full_model_path = os.path.join(base_dir, model_path)
        yolo_rgb = YOLO(full_model_path)
        model_gradcam = yolo_rgb.model
        model_gradcam.eval()
        
        # In case the model has enough layers, target [10] as in previous script
        try:
            target_layers = [model_gradcam.model[10]]
            cam = GradCAM(model=model_gradcam, target_layers=target_layers)
            use_gradcam = True
        except:
            use_gradcam = False
            print("GradCAM initialization failed, skipping GradCAM for this model")

        # Create output folders
        model_out = os.path.join(output_dir, model_name)
        rgb_out = os.path.join(model_out, "rgb")
        ir_out = os.path.join(model_out, "ir")
        gc_out = os.path.join(model_out, "gradcam")
        os.makedirs(rgb_out, exist_ok=True)
        os.makedirs(ir_out, exist_ok=True)
        os.makedirs(gc_out, exist_ok=True)
        
        for idx, filename in enumerate(sampled_files):
            rgb_path = os.path.join(images_rgb_dir, filename)
            ir_path = os.path.join(images_ir_dir, filename)
            
            # Predict RGB
            results_rgb = yolo_rgb(rgb_path, conf=0.25, verbose=False)
            if len(results_rgb) > 0:
                results_rgb[0].names = {0: 'human'}
                res_rgb_img = cv2.cvtColor(results_rgb[0].plot(labels=True, conf=True), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(rgb_out, filename), res_rgb_img)
            
            # Predict IR
            if os.path.exists(ir_path):
                results_ir = yolo_rgb(ir_path, conf=0.25, verbose=False)
                if len(results_ir) > 0:
                    results_ir[0].names = {0: 'human'}
                    res_ir_img = cv2.cvtColor(results_ir[0].plot(labels=True, conf=True), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(ir_out, filename), res_ir_img)
            
            # GradCAM (RGB)
            if use_gradcam:
                img = cv2.imread(rgb_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
                    input_tensor.requires_grad_(True)
                    
                    try:
                        with torch.enable_grad():
                            grayscale_cam = cam(input_tensor=input_tensor, targets=[YOLOConfidenceTarget()])[0]
                            
                        cam_image = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
                        cam_image = (cam_image * 255).astype(np.uint8) if cam_image.dtype == np.float32 else cam_image
                        
                        cv2.putText(cam_image, "human", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.imwrite(os.path.join(gc_out, filename), cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
                    except Exception as e:
                        print(f"GradCAM failed for {filename}: {e}")
            
            if (idx + 1) % 15 == 0:
                print(f"Processed {idx+1}/{len(sampled_files)} images for {model_name}")
                
    # ZIP
    print("Zipping results...")
    zip_path = os.path.join(base_dir, "comparative_results.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)
    print(f"Done! Saved to {zip_path}")

if __name__ == "__main__":
    run_predictions()
