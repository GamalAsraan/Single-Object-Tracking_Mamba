import os
import sys
import time
import argparse
import csv
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if "/app" not in sys.path:
    sys.path.append("/app")

from runtime.trackingmamba_runtime import TrackingMambaRuntime


def parse_args():
    parser = argparse.ArgumentParser(description="Latency Benchmark for TrackingMamba")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--init-bbox", type=str, required=True, help="Initial bbox as 'x,y,w,h'")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="/app/experiments/trackingmamba/trackingmamba.yaml")
    parser.add_argument("--output", type=str, default="/outputs/latency.csv", help="Where to save the benchmark CSV")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--frames", type=int, default=300, help="Max frames to benchmark")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup frames to exclude from stats")
    return parser.parse_args()


def main():
    args = parse_args()
    
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
    
    print(f"Benchmarking up to {args.frames} frames (skipping first {args.warmup} for stats)...")
    
    frame_idx = 0
    while frame_idx < args.frames:
        # Time decoding
        t_dec_start = time.perf_counter()
        ret, frame_bgr = cap.read()
        if not ret:
            print(f"Video ended early at frame {frame_idx}.")
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t_dec_end = time.perf_counter()
        
        decode_ms = (t_dec_end - t_dec_start) * 1000.0
        
        # Time tracking
        t_trk_start = time.perf_counter()
        if frame_idx == 0:
            bbox = tracker.initialize(frame_rgb, init_bbox)
        else:
            bbox = tracker.track(frame_rgb)
        
        # Note: torch.cuda.synchronize() ensures accurate GPU timing if async execution happens
        if args.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize()
            
        t_trk_end = time.perf_counter()
        
        track_ms = (t_trk_end - t_trk_start) * 1000.0
        total_ms = decode_ms + track_ms
        
        if frame_idx >= args.warmup:
            results.append({
                "frame_idx": frame_idx,
                "decode_ms": decode_ms,
                "track_ms": track_ms,
                "total_ms": total_ms
            })
            
        if frame_idx > 0 and frame_idx % 50 == 0:
            print(f"Benchmarked frame {frame_idx}/{args.frames}")
            
        frame_idx += 1

    cap.release()
    
    if len(results) == 0:
        print("Not enough frames processed for benchmarking stats.")
        return

    # Calculate statistics
    track_times = np.array([r["track_ms"] for r in results])
    decode_times = np.array([r["decode_ms"] for r in results])
    total_times = np.array([r["total_ms"] for r in results])
    
    p50_total = np.percentile(total_times, 50)
    p90_total = np.percentile(total_times, 90)
    p95_total = np.percentile(total_times, 95)
    min_total = np.min(total_times)
    max_total = np.max(total_times)
    avg_total = np.mean(total_times)
    
    p50_track = np.percentile(track_times, 50)
    p95_track = np.percentile(track_times, 95)
    avg_track = np.mean(track_times)
    
    avg_fps = 1000.0 / avg_total if avg_total > 0 else 0
    track_fps = 1000.0 / avg_track if avg_track > 0 else 0

    print("\n" + "="*50)
    print("LATENCY BENCHMARK RESULTS")
    print("="*50)
    print(f"Frames Benchmarked:  {len(results)} (excluding {args.warmup} warmup frames)")
    print(f"Goal Check:          Is Total Latency <= 30ms/frame?")
    print("-" * 50)
    print(f"Average FPS:         {avg_fps:.2f} FPS (including decode)")
    print(f"Track-Only FPS:      {track_fps:.2f} FPS")
    print("-" * 50)
    print(f"Avg Decode time:     {np.mean(decode_times):.2f} ms")
    print(f"Avg Track time:      {avg_track:.2f} ms")
    print("-" * 50)
    print("Total Latency (Decode + Track):")
    print(f"  Average:           {avg_total:.2f} ms")
    print(f"  Min:               {min_total:.2f} ms")
    print(f"  P50 (Median):      {p50_total:.2f} ms")
    print(f"  P90:               {p90_total:.2f} ms")
    print(f"  P95:               {p95_total:.2f} ms")
    print(f"  Max:               {max_total:.2f} ms")
    print("="*50)

    # Save output
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "decode_ms", "track_ms", "total_ms"])
            for r in results:
                writer.writerow([r["frame_idx"], f"{r['decode_ms']:.4f}", f"{r['track_ms']:.4f}", f"{r['total_ms']:.4f}"])
        print(f"Latency details saved to: {args.output}")


if __name__ == "__main__":
    main()
