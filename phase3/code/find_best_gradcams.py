import json
import os
import shutil

def find_best_gradcam():
    preds_file = "/Users/test/Q-WildFusion/runs/detect/val24/predictions.json"
    gradcam_dir = "/Users/test/Q-WildFusion/runs/detect/qualitative_results_gradcam"
    target_dir = "/Users/test/.gemini/antigravity/brain/be143aae-0c0f-43db-b94f-0541b1c3a599"
    
    with open(preds_file, 'r') as f:
        preds = json.load(f)
        
    # We want images that have exactly 1 or 2 highly confident, moderately large predictions
    # to perfectly showcase the attention mechanism scaling around a human.
    
    img_scores = {}
    for p in preds:
        img_id = p["image_id"]
        w, h = p["bbox"][2], p["bbox"][3]
        area = w * h
        score = p["score"]
        
        # Prioritize high confidence & decent area (not just a speck)
        if area > 4000:  
            # Ensure the grad-cam file actually exists for this prediction
            if os.path.exists(os.path.join(gradcam_dir, p["file_name"])):
                if img_id not in img_scores:
                    img_scores[img_id] = []
                img_scores[img_id].append(score)
            
    # Score each image by the maximum confidence prediction it has
    best_images = []
    for img_id, scores in img_scores.items():
        if max(scores) > 0.85 and len(scores) <= 3:  # high confidence, not too cluttered
            best_images.append((img_id, max(scores)))
            
    # Sort by score descending
    best_images.sort(key=lambda x: x[1], reverse=True)
    
    # Pick top 3
    top_3 = best_images[:3]
    print("Found top 3 Grad-CAM candidates:", top_3)
    
    for idx, (img_id, score) in enumerate(top_3):
        # Image filename is zero padded or just standard? 
        # In predictions it's the raw ID if we use image_id, or we can look at "file_name"
        # Since I didn't save file_name in my dict, let's grab it by matching back
        file_name = next(p["file_name"] for p in preds if p["image_id"] == img_id)
        
        src = os.path.join(gradcam_dir, file_name)
        dst = os.path.join(target_dir, f"gradcam_{idx+1}.jpg")
        
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied {file_name} to {dst}")
        else:
            print(f"Missing {src}")

if __name__ == "__main__":
    find_best_gradcam()
