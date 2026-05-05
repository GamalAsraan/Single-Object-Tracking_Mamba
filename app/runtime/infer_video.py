import os
import sys
import time
import json
import argparse
import csv
import cv2
import numpy as np

# Ensure we can import from app/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if "/app" not in sys.path:
    sys.path.append("/app")

from runtime.trackingmamba_runtime import TrackingMambaRuntime


def parse_args():
    parser = argparse.ArgumentParser(description="Single Video Inference for TrackingMamba")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--init-bbox", type=str, required=True, help="Initial bbox as 'x,y,w,h'")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="/app/experiments/trackingmamba/trackingmamba.yaml")
    parser.add_argument("--output", type=str, default="/outputs/boxes.csv")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-format", type=str, choices=["csv", "jsonl"], default="csv")
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse initial bbox
    try:
        init_bbox = [float(x) for x in args.init_bbox.split(",")]
        assert len(init_bbox) == 4
    except Exception as e:
        raise ValueError(f"Invalid --init-bbox format. Must be 'x,y,w,h'. Got {args.init_bbox}")

    print("Loading TrackingMambaRuntime...")
    tracker = TrackingMambaRuntime(
        checkpoint=args.checkpoint,
        config=args.config,
        device=args.device,
        fp16=args.fp16
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    results = []
    latencies = []
    
    frame_idx = 0
    while True:
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break
            
        ret, frame_bgr = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        t_start = time.perf_counter()
        if frame_idx == 0:
            bbox = tracker.initialize(frame_rgb, init_bbox)
        else:
            bbox = tracker.track(frame_rgb)
        t_end = time.perf_counter()
        
        latency_ms = (t_end - t_start) * 1000
        
        # skip warmup for latency stats
        if frame_idx >= args.warmup:
            latencies.append(latency_ms)
            
        results.append({
            "frame_idx": frame_idx,
            "x": bbox[0],
            "y": bbox[1],
            "w": bbox[2],
            "h": bbox[3],
            "latency_ms": latency_ms
        })
        
        if frame_idx > 0 and frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}...")
            
        frame_idx += 1

    cap.release()
    
    # Save output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if args.output_format == "csv":
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "x", "y", "w", "h", "latency_ms"])
            for r in results:
                writer.writerow([r["frame_idx"], r["x"], r["y"], r["w"], r["h"], r["latency_ms"]])
    else:
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
                
    # Stats
    if len(latencies) > 0:
        latencies = np.array(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        avg_fps = 1000.0 / np.mean(latencies)
    else:
        p50 = p95 = avg_fps = 0.0

    print(f"\nInference complete!")
    print(f"Frames processed: {frame_idx}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"P50 Latency: {p50:.2f} ms")
    print(f"P95 Latency: {p95:.2f} ms")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
