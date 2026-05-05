import os
import sys
import json
import argparse
import shutil
from queue import Queue
from threading import Thread
from types import SimpleNamespace

import cv2
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser("TrackingMamba AIC/MTC video/frame inference")

    parser.add_argument("--proj", type=str, default="/kaggle/working/TrackingMamba",
                        help="TrackingMamba repo root")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="TrackingMamba checkpoint path")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config path. Default: <proj>/experiments/trackingmamba/trackingmamba.yaml")

    parser.add_argument("--source", type=str, default="frames",
                        choices=["frames", "video", "convert"],
                        help=(
                            "frames  = use existing JPEG frames; "
                            "video   = read original mp4 directly; "
                            "convert = convert mp4 to JPEG frames then infer"
                        ))

    parser.add_argument("--jpeg-root", type=str,
                        default="/kaggle/input/datasets/gamalasran/mtc-aic4-jpegs/mtc-aic4-jpegs",
                        help="Root containing dataset*/seq/000000.jpg and metadata/contestant_manifest.json")

    parser.add_argument("--video-root", type=str,
                        default="/kaggle/input/datasets/gamalasran/aic-mtc-4",
                        help="Root containing original videos and metadata/sample_submission.csv")

    parser.add_argument("--manifest", type=str, default=None,
                        help="Manifest json. Default: jpeg_root/metadata/contestant_manifest.json")

    parser.add_argument("--sample-sub", type=str, default=None,
                        help="sample_submission.csv. Default: video_root/metadata/sample_submission.csv")

    parser.add_argument("--split", type=str, default="public_lb",
                        help="Manifest split to infer, usually public_lb")

    parser.add_argument("--frames-cache", type=str,
                        default="/kaggle/working/aic_frames_cache",
                        help="Writable cache dir for converted video frames")

    parser.add_argument("--output", type=str, default="/kaggle/working/submission.csv",
                        help="Output CSV path")

    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device, e.g. cuda:0")
    parser.add_argument("--prefetch", type=int, default=8,
                        help="Number of frames to prefetch")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="JPEG quality when --source convert")
    parser.add_argument("--keep-frames", action="store_true",
                        help="Keep converted frames after inference")
    parser.add_argument("--max-seqs", type=int, default=None,
                        help="Debug: limit number of sequences")
    parser.add_argument("--no-filelink", action="store_true",
                        help="Do not display IPython FileLink")

    return parser.parse_args()


def setup_repo(proj):
    if proj not in sys.path:
        sys.path.insert(0, proj)

    # Reload patched model file if running inside notebook / repeated process.
    import importlib
    import lib.models.trackingmamba.models_mamba
    importlib.reload(lib.models.trackingmamba.models_mamba)
    import lib.models.trackingmamba
    importlib.reload(lib.models.trackingmamba)


def load_manifest(args):
    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = os.path.join(args.jpeg_root, "metadata/contestant_manifest.json")

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if args.split not in manifest:
        raise KeyError(f"Split '{args.split}' not found in manifest. Keys: {list(manifest.keys())}")

    return manifest, manifest_path


def read_init_bbox(info, jpeg_root, video_root):
    """
    Public test annotations are one-line init bbox.
    Try video_root first, then jpeg_root.
    """
    rel = info["annotation_path"]

    candidates = [
        os.path.join(video_root, rel),
        os.path.join(jpeg_root, rel),
    ]

    for p in candidates:
        if os.path.isfile(p):
            with open(p, "r") as f:
                line = f.readline().strip()
            return list(map(float, line.split(",")))

    raise FileNotFoundError(f"Annotation not found. Tried: {candidates}")


def build_tracker(args):
    from lib.config.trackingmamba.config import cfg, update_config_from_file
    from lib.test.tracker.trackingmamba import TrackingMamba as TrackerClass

    cfg_file = args.config
    if cfg_file is None:
        cfg_file = os.path.join(args.proj, "experiments/trackingmamba/trackingmamba.yaml")

    if not os.path.isfile(cfg_file):
        raise FileNotFoundError(f"Config not found: {cfg_file}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    update_config_from_file(cfg_file)

    params = SimpleNamespace(
        cfg=cfg,
        checkpoint=args.checkpoint,
        template_factor=cfg.TEST.TEMPLATE_FACTOR,
        template_size=cfg.TEST.TEMPLATE_SIZE,
        search_factor=cfg.TEST.SEARCH_FACTOR,
        search_size=cfg.TEST.SEARCH_SIZE,
        save_all_boxes=False,
        debug=0,
    )

    torch.cuda.set_device(args.device)
    torch.cuda.empty_cache()

    tracker = TrackerClass(params, dataset_name="aerial")
    tracker.network.eval()

    return tracker, cfg_file


def make_fast_reader():
    """
    Returns function path -> RGB np.ndarray.
    jpeg4py is usually faster for JPEG frame folders.
    """
    try:
        import jpeg4py as jpeg

        def fast_read_rgb(path):
            return jpeg.JPEG(path).decode()

        print("✅ Using jpeg4py for JPEG decode")
        return fast_read_rgb

    except Exception as e:
        print(f"⚠️ jpeg4py unavailable ({e}), using cv2")

        def fast_read_rgb(path):
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return fast_read_rgb


def extract_video_to_frames(video_path, out_dir, expected_n=None, jpeg_quality=95):
    """
    Convert video to:
        out_dir/000000.jpg
        out_dir/000001.jpg
        ...
    Skips conversion if enough frames already exist.
    """
    os.makedirs(out_dir, exist_ok=True)

    if expected_n is not None:
        last_frame = os.path.join(out_dir, f"{expected_n - 1:06d}.jpg")
        if os.path.isfile(last_frame):
            return expected_n

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        out_path = os.path.join(out_dir, f"{idx:06d}.jpg")
        cv2.imwrite(out_path, frame_bgr, encode_params)
        idx += 1

    cap.release()

    if expected_n is not None and idx != expected_n:
        print(f"⚠️ Video frame count mismatch for {video_path}: decoded={idx}, expected={expected_n}")

    return idx


def frame_producer_from_folder(seq_dir, n_frames, q, read_rgb):
    for idx in range(n_frames):
        path = os.path.join(seq_dir, f"{idx:06d}.jpg")
        try:
            img = read_rgb(path)
        except Exception as e:
            print(f"⚠️ read error frame={idx} path={path}: {e}")
            img = None
        q.put((idx, img))
    q.put((None, None))


def frame_producer_from_video(video_path, q):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        q.put((idx, frame_rgb))
        idx += 1

    cap.release()
    q.put((None, None))


def track_sequence_from_queue(tracker, q, n_frames, init_bbox):
    """
    Track sequentially from a producer queue.
    """
    bboxes = [None] * n_frames

    with torch.inference_mode():
        while True:
            idx, image = q.get()

            if idx is None:
                break

            if idx >= n_frames:
                # Extra frames in video. Ignore.
                continue

            if image is None:
                if idx > 0 and bboxes[idx - 1] is not None:
                    bboxes[idx] = bboxes[idx - 1]
                else:
                    bboxes[idx] = init_bbox
                continue

            if idx == 0:
                tracker.initialize(image, {"init_bbox": init_bbox})
                bboxes[idx] = init_bbox
            else:
                out = tracker.track(image)
                bboxes[idx] = out["target_bbox"]

    # If video ended early, fill missing boxes with last known state.
    last = init_bbox
    for i in range(n_frames):
        if bboxes[i] is None:
            bboxes[i] = last
        else:
            last = bboxes[i]

    return bboxes


def track_sequence_frames(tracker, seq_dir, n_frames, init_bbox, read_rgb, prefetch=8):
    q = Queue(maxsize=prefetch)
    t = Thread(target=frame_producer_from_folder, args=(seq_dir, n_frames, q, read_rgb), daemon=True)
    t.start()
    bboxes = track_sequence_from_queue(tracker, q, n_frames, init_bbox)
    t.join()
    return bboxes


def track_sequence_video(tracker, video_path, n_frames, init_bbox, prefetch=8):
    q = Queue(maxsize=prefetch)
    t = Thread(target=frame_producer_from_video, args=(video_path, q), daemon=True)
    t.start()
    bboxes = track_sequence_from_queue(tracker, q, n_frames, init_bbox)
    t.join()
    return bboxes


def get_seq_paths(args, seq_key, info):
    frames_dir = os.path.join(args.jpeg_root, info["dataset"], info["seq_name"])
    video_path = os.path.join(args.video_root, info["video_path"])
    cache_dir = os.path.join(args.frames_cache, info["dataset"], info["seq_name"])
    return frames_dir, video_path, cache_dir


def main():
    args = parse_args()

    print("=" * 80)
    print("TrackingMamba AIC/MTC inference")
    print("=" * 80)
    print(f"source      : {args.source}")
    print(f"checkpoint  : {args.checkpoint}")
    print(f"output      : {args.output}")
    print(f"device      : {args.device}")
    print("=" * 80)

    setup_repo(args.proj)

    tracker, cfg_file = build_tracker(args)
    print(f"✅ Tracker loaded")
    print(f"✅ Config: {cfg_file}")

    manifest, manifest_path = load_manifest(args)
    print(f"✅ Manifest: {manifest_path}")

    sample_sub = args.sample_sub
    if sample_sub is None:
        sample_sub = os.path.join(args.video_root, "metadata/sample_submission.csv")

    if not os.path.isfile(sample_sub):
        raise FileNotFoundError(f"sample_submission.csv not found: {sample_sub}")

    seq_items = list(manifest[args.split].items())
    if args.max_seqs is not None:
        seq_items = seq_items[:args.max_seqs]

    total_frames = sum(info["n_frames"] for _, info in seq_items)
    print(f"✅ Sequences: {len(seq_items)}")
    print(f"✅ Frames   : {total_frames:,}")

    read_rgb = make_fast_reader()

    ids = []
    coords = []

    with tqdm(total=total_frames, desc="Frames", unit="f", dynamic_ncols=True) as pbar:
        for seq_key, info in seq_items:
            n_frames = int(info["n_frames"])
            init_bbox = read_init_bbox(info, args.jpeg_root, args.video_root)

            frames_dir, video_path, cache_dir = get_seq_paths(args, seq_key, info)

            if args.source == "frames":
                if not os.path.isdir(frames_dir):
                    raise FileNotFoundError(f"Frames dir not found: {frames_dir}")
                bboxes = track_sequence_frames(
                    tracker=tracker,
                    seq_dir=frames_dir,
                    n_frames=n_frames,
                    init_bbox=init_bbox,
                    read_rgb=read_rgb,
                    prefetch=args.prefetch,
                )

            elif args.source == "video":
                if not os.path.isfile(video_path):
                    raise FileNotFoundError(f"Video not found: {video_path}")
                bboxes = track_sequence_video(
                    tracker=tracker,
                    video_path=video_path,
                    n_frames=n_frames,
                    init_bbox=init_bbox,
                    prefetch=args.prefetch,
                )

            elif args.source == "convert":
                if not os.path.isfile(video_path):
                    raise FileNotFoundError(f"Video not found: {video_path}")

                extract_video_to_frames(
                    video_path=video_path,
                    out_dir=cache_dir,
                    expected_n=n_frames,
                    jpeg_quality=args.jpeg_quality,
                )

                bboxes = track_sequence_frames(
                    tracker=tracker,
                    seq_dir=cache_dir,
                    n_frames=n_frames,
                    init_bbox=init_bbox,
                    read_rgb=read_rgb,
                    prefetch=args.prefetch,
                )

                if not args.keep_frames:
                    shutil.rmtree(cache_dir, ignore_errors=True)

            else:
                raise ValueError(args.source)

            for frame_idx, bbox in enumerate(bboxes):
                ids.append(f"{seq_key}_{frame_idx}")
                coords.append(bbox)

            pbar.update(n_frames)
            pbar.set_postfix(seq=seq_key[:35], n=n_frames)

    coords_arr = np.round(np.array(coords, dtype=np.float32)).astype(np.int32)

    sub_ours = pd.DataFrame({
        "id": ids,
        "x": coords_arr[:, 0],
        "y": coords_arr[:, 1],
        "w": coords_arr[:, 2],
        "h": coords_arr[:, 3],
    })

    print(f"\nGenerated rows: {len(sub_ours)}")

    template = pd.read_csv(sample_sub)
    print(f"Template rows : {len(template)}")

    sub_final = template[["id"]].merge(sub_ours, on="id", how="left")

    n_missing = sub_final[["x", "y", "w", "h"]].isna().any(axis=1).sum()
    if n_missing > 0:
        print(f"⚠️ Missing rows: {n_missing}")
        print(sub_final[sub_final["x"].isna()].head(10))
    else:
        print(f"✅ All {len(sub_final)} rows matched")

    for col in ["x", "y", "w", "h"]:
        sub_final[col] = sub_final[col].fillna(-1).astype("int64")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sub_final.to_csv(args.output, index=False)

    print(f"\n✅ Saved: {args.output}")
    print(sub_final.head(10))
    print(sub_final.tail(5))

    if not args.no_filelink:
        try:
            from IPython.display import display, FileLink
            display(FileLink(args.output))
        except Exception:
            pass


if __name__ == "__main__":
    main()