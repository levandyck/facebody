import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as ndi
from contextlib import contextmanager

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

EPS = 1e-10

# ------------------------------ Guided GradCAM ------------------------------ #
class GuidedGradCAM:
    """Combine GradCAM and Guided Backpropagation to visualize feature preferences."""
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        cam_layer=None,
        device: str="cuda",
        upsample_mode: str="bilinear",
        align_corners: bool = False,
        post_blur_sigma: float=5.0,
        fusion_alpha: float=0.5,
    ):
        self.model = model.eval()
        self.target_layer = target_layer
        self.device = device
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners
        self.post_blur_sigma = post_blur_sigma
        self.fusion_alpha = fusion_alpha

        if cam_layer is None:
            cam_layer = self._auto_pick_cam_layer(self.model, target_layer)
        if not isinstance(cam_layer, nn.Conv2d):
            raise ValueError("cam_layer must be a conv layer.")
        self.cam_layer = cam_layer

    def __call__(
        self,
        img_t: torch.Tensor,
        objective_fn,
    ):
        """Compute Grad-CAM, Guided Backprop, and Guided Grad-CAM."""
        img_t = img_t.to(self.device)
        if img_t.dim() != 4 or img_t.size(0) != 1:
            raise ValueError("img_t must be a 4D tensor with batch size 1.")

        H, W = img_t.shape[2], img_t.shape[3]
        out_hw = (H, W)

        # Grad-CAM pass (standard backprop)
        cam_map = self._gradcam_pass(img_t, objective_fn, out_hw)

        # Guided Backprop pass (modified ReLUs)
        gb_map = self._guided_backprop_pass(img_t, objective_fn, out_hw)

        # Fusion into Guided Grad-CAM (geometric mean or simple product)
        cam_t = torch.from_numpy(cam_map + EPS)
        gb_t = torch.from_numpy(gb_map + EPS)

        if self.fusion_alpha is None:
            fused = cam_t * gb_t
        else:
            a = float(self.fusion_alpha)
            fused = cam_t.pow(a) * gb_t.pow(1.0 - a)

        ggc_map = self._to_numpy_01(fused, sigma=0.0)

        return {
            "cam": cam_map,
            "guided_backprop": gb_map,
            "guided_gradcam": ggc_map,
        }

    def _gradcam_pass(
        self,
        img_t: torch.Tensor,
        objective_fn,
        out_hw,
    ):
        img = img_t.detach().clone().requires_grad_(True)

        acts_tgt_holder = {}
        acts_cam_holder = {}

        def _hook_tgt(_m, _i, o):
            acts_tgt_holder["f"] = o

        def _hook_cam(_m, _i, o):
            acts_cam_holder["f"] = o

        ht = self.target_layer.register_forward_hook(_hook_tgt)
        hc = self.cam_layer.register_forward_hook(_hook_cam)

        try:
            _ = self.model(img)
        finally:
            ht.remove()
            hc.remove()

        acts_tgt = acts_tgt_holder["f"]
        acts_cam = acts_cam_holder["f"] # [1, C, h, w]
        T = objective_fn(acts_tgt)

        # dT/dA_cam
        grads = torch.autograd.grad(T, acts_cam, retain_graph=False, create_graph=False)[0]
        weights = grads.mean(dim=(2, 3), keepdim=True) # [1, C, 1, 1]
        cam = torch.relu((weights * acts_cam).sum(1, keepdim=True)) # [1, 1, h, w]

        cam = F.interpolate(
            cam,
            size=out_hw,
            mode=self.upsample_mode,
            align_corners=self.align_corners if self.upsample_mode != "nearest" else None,
        )[0, 0]

        return self._to_numpy_01(cam, sigma=self.post_blur_sigma)

    def _guided_backprop_pass(
        self,
        img_t: torch.Tensor,
        objective_fn,
        out_hw,
    ):
        img = img_t.detach().clone().requires_grad_(True)

        with self._retain_relu_guided_backprop():
            acts_tgt_holder = {}

            def _hook_tgt(_m, _i, o):
                acts_tgt_holder["f"] = o

            ht = self.target_layer.register_forward_hook(_hook_tgt)
            try:
                _ = self.model(img)
            finally:
                ht.remove()

            acts_tgt = acts_tgt_holder["f"]
            T = objective_fn(acts_tgt)

            self.model.zero_grad(set_to_none=True)
            T.backward()

            gb = img.grad  # [1, C, H, W]
            gb = torch.clamp(gb, min=0.0).sum(1, keepdim=True) # [1, 1, H, W]

            gb = F.interpolate(
                gb,
                size=out_hw,
                mode=self.upsample_mode,
                align_corners=self.align_corners if self.upsample_mode != "nearest" else None,
            )[0, 0]

        return self._to_numpy_01(gb, sigma=self.post_blur_sigma)

    @staticmethod
    def _to_numpy_01(x: torch.Tensor, sigma: float=0.0):
        """Tensor HxW -> numpy float32 in [0,1] with optional Gaussian blur."""
        arr = x.detach().float().cpu().numpy()
        if sigma > 0:
            arr = ndi.gaussian_filter(arr, sigma=sigma)
        arr = np.clip(arr, 0.0, None)
        denom = (arr.max() - arr.min()) if arr.max() > arr.min() else 1.0
        return ((arr - arr.min()) / denom).astype(np.float32)

    @staticmethod
    def _auto_pick_cam_layer(model: nn.Module, target_layer: nn.Module):
        """
        Pick a sensible conv layer for CAM:
        - If target is Linear: use the last Conv2d in the model.
        - If target is Conv2d: use the last Conv2d before target.
        - Otherwise: fallback to last Conv2d anywhere.
        """
        last_conv_before = None
        last_conv_any = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv_any = m
            if m is target_layer:
                break
            if isinstance(m, nn.Conv2d):
                last_conv_before = m

        if isinstance(target_layer, nn.Linear):
            if last_conv_any is None:
                raise ValueError("Model has no Conv2d layers for CAM.")
            return last_conv_any

        if isinstance(target_layer, nn.Conv2d):
            if last_conv_before is None:
                raise ValueError(
                    "Need a conv layer earlier than the target layer for Grad-CAM."
                )
            return last_conv_before

        if last_conv_any is None:
            raise ValueError("Model has no Conv2d layers for CAM.")
        return last_conv_any

    @contextmanager
    def _retain_relu_guided_backprop(self):
        """
        Context manager that enables Guided Backprop on all nn.ReLU modules.
        Implements the 'guided backprop' rule by zeroing out negative gradients at ReLU.
        """
        bwd_handles = []

        def bwd_hook(module, grad_in, grad_out):
            if not grad_in:
                return grad_in
            gi = grad_in[0]
            if gi is None:
                return grad_in

            # Guided backprop: keep only positive gradients
            return (torch.clamp(gi, min=0.0),)

        # Register backward hooks on all ReLUs in this model instance
        for m in self.model.modules():
            if isinstance(m, nn.ReLU):
                bwd_handles.append(m.register_full_backward_hook(bwd_hook))

        try:
            yield
        finally:
            for h in bwd_handles:
                h.remove()
