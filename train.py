import os
import torch
import csv
import glob
from datetime import datetime
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from sys import getsizeof
import lpips


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# ----------------------------------------------------------------------------------
# RefinedGaussianModel Class
# ----------------------------------------------------------------------------------

class RefinedGaussianModel(GaussianModel):
    """
    An extension of the GaussianModel that includes logic for adaptive pruning and merging
    to reduce the final model size while preserving quality.
    """
    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)

    def refine_and_prune(self, merge_threshold_xyz, merge_threshold_color, merge_threshold_scale, prune_percent, radii):
        """
        The core refinement method. It prunes unimportant Gaussians and merges redundant ones.
        """
        self.tmp_radii = radii

        if self._xyz.shape[0] == 0:
            return

        # 1. Importance-Score Pruning
        # A simple importance score could be opacity * max_scale. More complex scores can be used.
        importance_scores = self.get_opacity.squeeze(-1) * self.get_scaling.max(dim=1).values
        num_to_prune = int(self._xyz.shape[0] * prune_percent)
        
        # Get the indices of the Gaussians with the lowest importance scores
        prune_indices = torch.topk(importance_scores, k=num_to_prune, largest=False).indices
        
        prune_mask = torch.zeros(self._xyz.shape[0], dtype=torch.bool, device="cuda")
        prune_mask[prune_indices] = True
        
        # Invert mask to keep the important points
        keep_mask = ~prune_mask
        self.prune_points(prune_mask)
        
        # 2. Redundancy-Driven Merging
        # This is a simplified example. A production version would use a spatial hash grid for efficiency.
        # Here we do a pairwise check for simplicity, which is slow but demonstrates the logic.
        
        # Only check a random subset to speed things up
        num_points = self._xyz.shape[0]
        if num_points < 2: return
        
        # Randomly select pairs of points to check for merging
        # Note: This is not efficient but serves as a proof of concept.
        # A real implementation would use a k-d tree or spatial hash grid.
        perm = torch.randperm(num_points, device="cuda")
        p1_indices = perm[:num_points//2]
        p2_indices = perm[num_points//2 : num_points//2 * 2] # Ensure we have pairs

        p1 = self._xyz[p1_indices]
        p2 = self._xyz[p2_indices]

        # Calculate distances and similarities
        dist = torch.norm(p1 - p2, dim=1)
        color_dist = torch.norm(self._features_dc[p1_indices] - self._features_dc[p2_indices], dim=2).squeeze()
        scale_dist = torch.norm(self._scaling[p1_indices] - self._scaling[p2_indices], dim=1)

        # Create a mask for points that are close enough to be merged
        merge_mask_indices = torch.where(
            (dist < merge_threshold_xyz) &
            (color_dist < merge_threshold_color) &
            (scale_dist < merge_threshold_scale)
        )[0]
        
        if merge_mask_indices.numel() == 0:
            return
            
        # Get the actual indices to merge from p1_indices and p2_indices
        p1_to_merge_idx = p1_indices[merge_mask_indices]
        p2_to_merge_idx = p2_indices[merge_mask_indices]
        
        # Create new merged Gaussians
        new_xyz = (self._xyz[p1_to_merge_idx] + self._xyz[p2_to_merge_idx]) / 2.0
        new_f_dc = (self._features_dc[p1_to_merge_idx] + self._features_dc[p2_to_merge_idx]) / 2.0
        new_f_rest = (self._features_rest[p1_to_merge_idx] + self._features_rest[p2_to_merge_idx]) / 2.0
        new_opacity = torch.max(self._opacity[p1_to_merge_idx], self._opacity[p2_to_merge_idx])
        new_scaling = torch.max(self._scaling[p1_to_merge_idx], self._scaling[p2_to_merge_idx])
        new_rotation = self._rotation[p1_to_merge_idx] # Just take one of the rotations

        # Create a dictionary for the new Gaussians
        new_gaussians_dict = {
            "xyz": new_xyz, "f_dc": new_f_dc, "f_rest": new_f_rest,
            "opacity": new_opacity, "scaling": new_scaling, "rotation": new_rotation
        }
        
        # Get optimizable tensors for the new merged gaussians
        new_optimizable_tensors = self.get_optimizable_tensors(**new_gaussians_dict)

        # Create a mask to remove the old, merged points
        merged_points_mask = torch.zeros(self._xyz.shape[0], dtype=torch.bool, device="cuda")
        merged_points_mask[p1_to_merge_idx] = True
        merged_points_mask[p2_to_merge_idx] = True

        # Prune the old points
        self.prune_points(merged_points_mask)
        
        # Add the new merged points
        self.densify_tensors_add(new_optimizable_tensors)
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------
# Your Original Script (with modifications to use RefinedGaussianModel)
# ----------------------------------------------------------------------------------

# Define iteration numbers and other constants.
iter_num = 30100
sh_num = 1000

common_iterations = [i for i in range(1000, iter_num, 1000)]
test_iterations = common_iterations + [iter_num]
save_iterations = common_iterations + [iter_num]

# Loss weights for combining L1 and L2 losses.
lambda_l1 = 0.5
lambda_l2 = 0.5
lambda_cov = 0.1
cov_threshold = 1.0
densification_factor = 0.5

# Initialize LPIPS metric
lpips_fn = lpips.LPIPS(net='alex')
if torch.cuda.is_available():
    lpips_fn = lpips_fn.cuda()

def measure_model_sizes(output_dir):
    # ... (function remains unchanged)
    model_sizes = {}
    model_dir_pattern = os.path.join(output_dir, "Model*", "point_cloud", "iteration_*")
    model_dirs = glob.glob(model_dir_pattern)
    for model_dir in model_dirs:
        iteration = os.path.basename(model_dir).split('_')[-1]
        ply_files = glob.glob(os.path.join(model_dir, "*.ply"))
        total_size = 0 
        for ply_file in ply_files:
            total_size += os.path.getsize(ply_file)
        model_sizes[iteration] = total_size / 1024 / 1024
    return model_sizes

def update_csv_with_model_sizes(csv_path, model_sizes):
    # ... (function remains unchanged)
    with open(csv_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        lines = list(reader)
    header = lines[0]
    if len(header) < 8:
        header.extend([''] * (8 - len(header)))  
    header[7] = 'Model Size (MB)'
    for i, line in enumerate(lines[1:]):
        iteration = line[0]  
        if iteration in model_sizes:
            if len(line) < 8:
                line.extend([''] * (8 - len(line)))
            line[7] = str(model_sizes[iteration])  
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(lines)

def compute_covariance_regularization(gaussians):
    # ... (function remains unchanged)
    if hasattr(gaussians, 'get_scaling'):
        # Penalize scales instead of covariances directly
        penalty = torch.sum(torch.clamp(gaussians.get_scaling - cov_threshold, min=0))
        return penalty
    return 0.0

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    # Use our new RefinedGaussianModel
    gaussians = RefinedGaussianModel(dataset.sh_degree)
    
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # --- New Hyperparameters for Refinement Phase ---
    refinement_start_iter = 15000  # Start refining after this iteration
    refinement_interval = 500       # Run refinement every N iterations
    merge_threshold_xyz = 0.01      # Max distance to consider for merging
    merge_threshold_color = 0.1     # Max color difference
    merge_threshold_scale = 0.1     # Max scale difference
    prune_percent = 0.01            # Prune 1% of least important points each refinement step

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    accumulated_time = 0.0

    csv_path = r'/mnt/d/Github/gaussian-splatting/CSVData/training_data.csv'
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Iteration', 'SSIM', 'L1', 'PSNR', 'LPIPS', 'Loss', 'FPS', 'Iteration Time'])

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, min(opt.iterations, iter_num+1)), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, min(opt.iterations, iter_num+1)):        
        if network_gui.conn is None: network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                # ... GUI code remains unchanged ...
                pass
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if iteration % sh_num == 0: gaussians.oneupSHdegree()

        if not viewpoint_stack: viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if (iteration - 1) == debug_from: pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        rec_loss_l1 = l1_loss(image, gt_image)
        rec_loss_l2 = torch.nn.functional.mse_loss(image, gt_image)
        ssim_val = ssim(image, gt_image)
        loss = lambda_l1 * rec_loss_l1 + lambda_l2 * rec_loss_l2 + opt.lambda_dssim * (1.0 - ssim_val)
        cov_penalty = compute_covariance_regularization(gaussians)
        loss = loss + lambda_cov * cov_penalty
        loss.backward()

        iter_end.record()
        torch.cuda.synchronize()
        elapsed_time = iter_start.elapsed_time(iter_end) / 1000
        accumulated_time += elapsed_time
        formatted_accumulated_time = "{:.3f}".format(accumulated_time)
        fps = 1 / elapsed_time

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 1 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}", "Points": f"{gaussians._xyz.shape[0]}"})
                progress_bar.update(1)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in testing_iterations:
                training_report(tb_writer, iteration, rec_loss_l1, loss, rec_loss_l2, formatted_accumulated_time, testing_iterations, scene, render, (pipe, background), csv_writer, ssim_val.item(), fps, elapsed_time)
                if (iteration in saving_iterations):
                    print(f"\n[ITER {iteration}] Saving Gaussians ({gaussians._xyz.shape[0]} points)")
                    scene.save(iteration)

            # --- TWO-STAGE TRAINING LOGIC ---
            if iteration < refinement_start_iter:
                # --- Stage 1: Growth Phase (Standard Densification) ---
                if iteration < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
            else:
                # --- Stage 2: Refinement Phase (Pruning and Merging) ---
                if iteration % refinement_interval == 0:
                    print(f"\n[ITER {iteration}] Entering refinement phase... Current points: {gaussians._xyz.shape[0]}")
                    gaussians.refine_and_prune(
                        merge_threshold_xyz,
                        merge_threshold_color,
                        merge_threshold_scale,
                        prune_percent,
                        radii  # --- FIX: Pass the radii tensor here ---
                    )
                    print(f"Refinement complete. New points: {gaussians._xyz.shape[0]}")

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt{iteration}.pth")

def prepare_output_and_logger(args):    
    # ... (function remains unchanged)
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.model_path = os.path.join("./output/", f"Model{current_time}")
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, l1_val, loss, rec_loss_l2, formatted_accumulated_time, testing_iterations, scene, renderFunc, renderArgs, csv_writer, ssim_value, fps, elapsed_time):
    # ... (function remains unchanged)
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', l1_val.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed_time, iteration)
    psnr_test = 0.0
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()}, 
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        )
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    lpips_test += lpips_fn(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test} LPIPS {lpips_test}")
        psnr_value = psnr_test.item() if torch.is_tensor(psnr_test) else psnr_test
        lpips_value = lpips_test.item() if torch.is_tensor(lpips_test) else lpips_test
        csv_writer.writerow([iteration, ssim_value, l1_val.item(), psnr_value, lpips_value, loss.item(), fps, formatted_accumulated_time])
        if tb_writer:
            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=test_iterations)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=save_iterations)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
    model_sizes = measure_model_sizes(r'/home/meic/Desktop/gaussian-splatting/output')
    csv_path = r'/mnt/d/Github/gaussian-splatting/CSVData/training_data.csv'
    update_csv_with_model_sizes(csv_path, model_sizes)

    final_ssim = None
    csv_path = r'/mnt/d/Github/gaussian-splatting/CSVData/training_data.csv'
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            final_ssim = row[1]
    if final_ssim:
        print(f"\nModel SSIM: {final_ssim}")