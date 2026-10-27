import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--area", default="all",
                        choices=["all", "medium", "large", "medium+large", "non-small"])
    args = parser.parse_args()

    assert os.path.exists(args.gt), f"GT file not found: {args.gt}"
    assert os.path.exists(args.pred), f"Pred file not found: {args.pred}"

    print("Loading ground truth...")
    cocoGt = COCO(args.gt)

    print("Loading predictions...")
    cocoDt = cocoGt.loadRes(args.pred)

    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")

    # Area filtering
    if args.area == "medium":
        cocoEval.params.areaRng = [[32**2, 96**2]]
        cocoEval.params.areaRngLbl = ["medium"]
    elif args.area == "large":
        cocoEval.params.areaRng = [[96**2, 1e5**2]]
        cocoEval.params.areaRngLbl = ["large"]
    elif args.area == "medium+large":
        cocoEval.params.areaRng = [[32**2, 1e5**2]]
        cocoEval.params.areaRngLbl = ["medium+large"]
    elif args.area == "non-small":
        cocoEval.params.areaRng = [[32**2, 1e5**2]]
        cocoEval.params.areaRngLbl = ["non-small"]

    print("\nRunning COCO evaluation...")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # Safe metric extraction
    ap50 = cocoEval.stats[1]   # AP@50
    ar50 = cocoEval.stats[6]   # AR@50 (maxDets=100)

    # Save CSV
    output_csv = f"phase3/eval/metrics_{args.area}.csv"
    os.makedirs("phase3/eval", exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Area", "AP@50", "AR@50"])
        writer.writerow([args.area, round(ap50, 4), round(ar50, 4)])

    print("\n📌 Final Metric")
    print(f"AP@50 ({args.area}) = {ap50:.4f}")
    print(f"AR@50 ({args.area}) = {ar50:.4f}")
    print(f"✅ Saved to {output_csv}")

if __name__ == "__main__":
    main()
