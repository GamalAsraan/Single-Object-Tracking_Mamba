import os
import sys

# Ensure we can import from app/ directory when run locally, and add /app for Docker
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if "/app" not in sys.path:
    sys.path.append("/app")

import torch
import numpy as np

from lib.config.trackingmamba.config import cfg, update_config_from_file
from lib.models.trackingmamba import build_trackingmamba
from lib.test.tracker.data_utils import Preprocessor
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.test.utils.hann import hann2d
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class TrackingMambaRuntime:
    def __init__(self, checkpoint: str, config: str, device: str = "cuda:0", fp16: bool = False):
        self.device = device
        self.fp16 = fp16
        
        update_config_from_file(config)
        self.cfg = cfg
        
        self.template_factor = self.cfg.TEST.TEMPLATE_FACTOR
        self.template_size = self.cfg.TEST.TEMPLATE_SIZE
        self.search_factor = self.cfg.TEST.SEARCH_FACTOR
        self.search_size = self.cfg.TEST.SEARCH_SIZE
        
        self.network = build_trackingmamba(self.cfg, training=False)
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.network.load_state_dict(ckpt['net'], strict=True)
        
        self.network.to(self.device)
        self.network.eval()
        
        if self.fp16:
            try:
                self.network.half()
            except Exception as e:
                print(f"Warning: fp16 failed, falling back to fp32. Error: {e}")
                self.fp16 = False
                self.network.float()
                
        self.preprocessor = Preprocessor()
        
        # Initialize Hann window
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).to(self.device)
        
        self.reset()

    def reset(self) -> None:
        self.state = None
        self.z_dict1 = None
        self.box_mask_z = None

    def initialize(self, frame_rgb: np.ndarray, init_bbox: list[float]) -> list[float]:
        self.reset()
        
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            frame_rgb, init_bbox, self.template_factor, output_sz=self.template_size)
            
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        template.tensors = template.tensors.to(self.device)
        if template.mask is not None:
            template.mask = template.mask.to(self.device)

        if self.fp16:
            template.tensors = template.tensors.half()
            
        with torch.inference_mode():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            crop_sz = torch.Tensor([self.template_size, self.template_size])
            box_in = torch.tensor(init_bbox)
            template_bbox = transform_image_to_crop(box_in, box_in, resize_factor, crop_sz, normalize=True)
            template_bbox = template_bbox.view(1, 1, 4).to(self.device)
            template_bbox = template_bbox.squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, self.device, template_bbox)

        self.state = init_bbox
        return init_bbox

    def track(self, frame_rgb: np.ndarray) -> list[float]:
        if self.state is None:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")
            
        H, W, _ = frame_rgb.shape
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            frame_rgb, self.state, self.search_factor, output_sz=self.search_size)
            
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        search.tensors = search.tensors.to(self.device)
        if search.mask is not None:
            search.mask = search.mask.to(self.device)

        if self.fp16:
            search.tensors = search.tensors.half()

        with torch.inference_mode():
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=search.tensors, ce_template_mask=self.box_mask_z)

        # Apply Hann window to score map
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        
        # Mean of predicted boxes
        pred_box = (pred_boxes.mean(dim=0) * self.search_size / resize_factor).tolist()
        
        # map_box_back
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        real_box = [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]
        
        # clip_box
        self.state = clip_box(real_box, H, W, margin=10)
        return self.state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    
    print("Initializing TrackerRuntime...")
    tracker = TrackingMambaRuntime(
        checkpoint=args.checkpoint, 
        config=args.config,
        device=args.device,
        fp16=args.fp16
    )
    print("model loaded")
