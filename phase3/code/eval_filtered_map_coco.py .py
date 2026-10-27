import argparse
import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

print("USING UPDATED eval_filtered_map_coco.py")

def evaluate_coco(gt_json, pred_json, areas):
    assert os.path.exists(gt_json), f"GT file not found: {gt_json}"
    assert os.path.exists(pred_json), f"Pred file not found: {pred_json}"

    print("Loading ground truth...")
    cocoGt = COCO(gt_json)

    print("Loading predictions...")
    cocoDt = cocoGt.loadRes(pred_json)

    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")

    # ---- Area filtering ----
    area_map = {
        "small": [0**2, 32**2],
        "medium": [32**2, 96**2],
        "large": [96**2, 1e10],
    }

    area_ranges = [area_map[a] for a in areas]
    cocoEval.params.areaRng = area_ranges
    cocoEval.params.areaRngLbl = areas

    # ---- Evaluate AP@50 only ----
    cocoEval.params.iouThrs = [0.5]

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    ap50 = cocoEval.stats[1]  # AP@50
    print(f"\n📌 Final AP@50 ({'+'.join(areas)}): {ap50:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--area", nargs="+", default=["medium", "large"])
    args = parser.parse_args()

    evaluate_coco(args.gt, args.pred, args.area)
