import os
import json
import cv2
import numpy as np

def generate_ir_grid():
    ir_dir = "/Users/test/Q-WildFusion/phase3/dataset_aligned/images/ir/val"
    preds_file = "/Users/test/Q-WildFusion/runs/detect/val24/predictions.json"
    output_path = "/Users/test/.gemini/antigravity/brain/be143aae-0c0f-43db-b94f-0541b1c3a599/4_thermal_layer.jpg"
    
    # Get first 16 images (assuming batch size 16, validation loader is un-shuffled)
    images = sorted(os.listdir(ir_dir))[:16]
    
    with open(preds_file, 'r') as f:
        predictions = json.load(f)
        
    grid_images = []
    
    for img_name in images:
        img_id = int(img_name.split('.')[0])
        img_path = os.path.join(ir_dir, img_name)
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (960, 960))
        h_img, w_img, _ = img.shape
        
        # Draw predictions
        for p in predictions:
            if p["image_id"] == img_id:
                if p["score"] > 0.25: # Confidence threshold
                    bbox = p["bbox"]
                    x1, y1, w, h = [int(v) for v in bbox]
                    # Since predicted on 960x960 optionally or original
                    # scale properly if needed. Let's assume predictions are absolute coordinates mapped to original 1280 or 960.
                    # Usually predictions.json is in original image coordinates
                    cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 0, 255), 3) # Red boxes for thermal
                    cv2.putText(img, f'{p["score"]:.2f}', (x1, max(10, y1-10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                                
        grid_images.append(img)
        
    # Build 4x4 grid
    rows = []
    for i in range(0, 16, 4):
        row = np.hstack(grid_images[i:i+4])
        rows.append(row)
        
    grid = np.vstack(rows)
    
    # Resize grid down so it's not massive
    grid = cv2.resize(grid, (1920, 1920))
    cv2.imwrite(output_path, grid)
    print("Saved thermal grid to", output_path)

if __name__ == "__main__":
    generate_ir_grid()
