import os
import shutil
import random
import zipfile

def collect_files():
    base_dir = "/Users/test/Q-WildFusion"
    gradcam_dir = os.path.join(base_dir, "runs/detect/qualitative_results_gradcam")
    rgb_dir = os.path.join(base_dir, "phase3/dataset/images/rgb/val")
    ir_dir = os.path.join(base_dir, "phase3/dataset/images/ir/val")
    output_dir = os.path.join(base_dir, "extracted_results")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    for category in ["gradcam", "rgb", "ir"]:
        os.makedirs(os.path.join(output_dir, category))
        
    def sample_and_copy(src_dir, dst_dir, n=60):
        if not os.path.exists(src_dir):
            print(f"Warning: {src_dir} does not exist.")
            return
        files = [f for f in os.listdir(src_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        sampled = random.sample(files, min(len(files), n))
        for i, f in enumerate(sampled):
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))
            
    print("Collecting GradCAM...")
    sample_and_copy(gradcam_dir, os.path.join(output_dir, "gradcam"), n=60)
    
    print("Collecting RGB...")
    sample_and_copy(rgb_dir, os.path.join(output_dir, "rgb"), n=60)
    
    print("Collecting IR...")
    sample_and_copy(ir_dir, os.path.join(output_dir, "ir"), n=60)
    
    print("Zipping results...")
    zip_path = os.path.join(base_dir, "all_results_extracted.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)
                
    print(f"Done! Results zipped at {zip_path}")
    print(f"\nThe output directory '{output_dir}' also contains the structured folders.")

if __name__ == "__main__":
    collect_files()
