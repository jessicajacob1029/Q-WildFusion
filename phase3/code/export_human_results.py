import json
import os
import pandas as pd

def export_human_labeled_results(preds_json_path, output_csv_path):
    """
    Export COCO predictions with explicit 'human' labels for all detections
    """

    # Load predictions
    with open(preds_json_path, 'r') as f:
        predictions = json.load(f)

    # Convert to readable format
    results = []
    for pred in predictions:
        # Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]
        x1, y1, w, h = pred['bbox']
        x2 = x1 + w
        y2 = y1 + h

        results.append({
            'image_id': pred['image_id'],
            'label': 'human',  # Explicit human label
            'confidence': pred['score'],
            'bbox_x1': round(x1, 2),
            'bbox_y1': round(y1, 2),
            'bbox_x2': round(x2, 2),
            'bbox_y2': round(y2, 2),
            'bbox_width': round(w, 2),
            'bbox_height': round(h, 2)
        })

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)

    print(f"✅ Exported {len(results)} human detections to {output_csv_path}")
    print(f"📊 Sample results:")
    print(df.head(10))

    return df

if __name__ == "__main__":
    preds_json = "phase3/eval/preds_val_coco.json"
    output_csv = "phase3/eval/final_human_detections.csv"

    if os.path.exists(preds_json):
        export_human_labeled_results(preds_json, output_csv)
    else:
        print(f"❌ Predictions file not found: {preds_json}")