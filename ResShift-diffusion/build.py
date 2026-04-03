import os, datetime
from time import time
from tqdm import tqdm
import torch
import torchvision.utils as vutils
import math
import numpy as np
import csv

from networks.unet import UNetModelSwin
import torch.amp as amp
from contextlib import nullcontext
from diffusion.Gaussian_model import create_gaussian_diffusion
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from copy import deepcopy
from collections import OrderedDict
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure  # Updated imports
import argparse

from piq import vif_p, gmsd, multi_scale_gmsd, psnr, ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Model(object):
    def __init__(self,
                 name,
                 device,
                 data_loader,
                 test_data_loader,
                 FLAGS):
        self.name = name
        self.device = device
        self.data_loader = data_loader
        self.test_data_loader = test_data_loader
        self.flags = FLAGS
        self.state = {'epoch': 0, 'step': 0}  # Initialize state
        assert self.name == 'Super_resolution'

        self.setup_dist()

        self.build_model()
        self.create_optim()


        if self.flags.resume:
            print(f"=> Loading checkpoint from {self.flags.out_dir}")
            checkpoint_file = os.path.join(self.flags.out_dir, 'content.pth')
            if os.path.exists(checkpoint_file):
                checkpoint = torch.load(checkpoint_file, map_location=self.device)
                self.init_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['model_state_dict'])
                torch.cuda.empty_cache()
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

                # Load EMA checkpoint if available
                ema_ckpt_path = os.path.join(self.flags.out_dir, 'ema_model.pth')
                if os.path.exists(ema_ckpt_path) and self.rank == 0 and hasattr(self, 'ema_rate'):
                    print(f"=> Loaded EMA checkpoint from {ema_ckpt_path}")
                    ema_ckpt = torch.load(ema_ckpt_path, map_location=f"cuda:{self.rank}")
                    self._load_ema_state(self.ema_state, ema_ckpt)
                else:
                    print(f"Warning: EMA checkpoint file {ema_ckpt_path} not found.")
            else:
                print(f"Warning: Checkpoint file {checkpoint_file} does not exist. Skipping resume.")
                self.global_step, self.init_epoch = 0, 0

            torch.cuda.empty_cache()

            # AMP scaler
            if self.amp_scaler is not None:
                if "amp_scaler" in checkpoint:
                    self.amp_scaler.load_state_dict(checkpoint["amp_scaler"])
                    if self.rank == 0:
                        print("Loading scaler from resumed state...")
        else:
            self.global_step, self.init_epoch = 0, 0



        ddconfig = {
                    'double_z': self.flags.ddconfig_double_z,
                    'z_channels': self.flags.ddconfig_z_channels,
                    'resolution': self.flags.ddconfig_resolution,
                    'in_channels': self.flags.ddconfig_in_channels,
                    'out_ch': self.flags.ddconfig_out_ch,
                    'ch': self.flags.ddconfig_ch,
                    'ch_mult': self.flags.ddconfig_ch_mult,
                    'num_res_blocks': self.flags.ddconfig_num_res_blocks,
                    'attn_resolutions': self.flags.ddconfig_attn_resolutions,
                    'dropout': self.flags.ddconfig_dropout,
                    'padding_mode': self.flags.ddconfig_padding_mode
                } # parameters of the autoencoder, but I am not using it.
        self.autoencoder = None #autoencoder(ddconfig=ddconfig,
                                       # n_embed=self.flags.n_embed,
                                       # embed_dim=self.flags.embed_dim
                                       # )
        params_diffusion = {
            'sf': self.flags.sf, # default=1.0, Scale factor
            'schedule_name': self.flags.schedule_name, #  default='exponential', Name of the schedule
            'schedule_kwargs': self.flags.schedule_kwargs,  # default={'power': 0.3}, Additional schedule arguments as a dictionary
            'etas_end': self.flags.etas_end,  # default=0.99, Ending value for etas
            'steps': self.flags.steps,  # default=4, Number of steps
            'min_noise_level': self.flags.min_noise_level, # default=0.2, Minimum noise level
            'kappa': self.flags.kappa,  # default=2.0, Kappa value
            'weighted_mse': self.flags.weighted_mse, # action='store_true', Use weighted mean squared error
            'predict_type': self.flags.predict_type,  # default='xstart', Type of prediction
            'timestep_respacing': self.flags.timestep_respacing, # default=None,  Timestep re-spacing value
            'scale_factor': self.flags.scale_factor, # default=1.0, help='Scale factor'
            'normalize_input': self.flags.normalize_input, # action='store_true',  Whether to normalize input
            'latent_flag': self.flags.latent_flag # action='store_true', Flag for using latent variables
        } # parameters of the diffusion model
        self.base_diffusion = create_gaussian_diffusion(**params_diffusion)

        self.l1_loss = self.create_l1_loss()


    def _load_ema_state(self, ema_state, ckpt):
        for key in ema_state.keys():
            if key not in ckpt and key.startswith('module'):
                ema_state[key] = deepcopy(ckpt[7:].detach().data)
            elif key not in ckpt and (not key.startswith('module')):
                ema_state[key] = deepcopy(ckpt['module.' + key].detach().data)
            else:
                ema_state[key] = deepcopy(ckpt[key].detach().data)

    def setup_dist(self):
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(
                    timeout=datetime.timedelta(seconds=3600),
                    backend='nccl',
                    init_method='env://',
                    )

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0


    def build_model(self):
        model = UNetModelSwin(image_size = self.flags.image_size,
                              in_channels = self.flags.in_channels,
                              model_channels = self.flags.model_channels,
                              out_channels = self.flags.out_channels,
                              num_res_blocks = self.flags.num_res_blocks,
                              attention_resolutions = self.flags.attention_resolutions,
                              lq_size=self.flags.image_size,
                              )
        model.to(self.device)


        if self.flags.compile_flag:
            if self.rank == 0:
                print("Begin compiling model...")
            model = torch.compile(model, mode=self.flags.compile_mode)
            # if self.rank == 0:
            #     print("Compiling Done")
        if self.num_gpus > 1:
            self.model = DDP(model, device_ids=[self.rank,], static_graph=False)  # wrap the network
        else:
            self.model = model

        # EMA
        if self.rank == 0 and self.flags.ema_rate is not None:
            self.ema_model = deepcopy(model).to(self.device)
            self.ema_state = OrderedDict(
                {key:deepcopy(value.data).float() for key, value in self.model.state_dict().items()}
                )
            self.ema_ignore_keys = [x for x in self.ema_state.keys() if ('running_' in x or 'num_batches_tracked' in x)]
        else:
            print("EMA is not used, as ema_rate is not defined.")

        # model information
        self.print_model_info()

    def print_model_info(self):
        if self.rank == 0:

            num_params = 0
            for param in self.model.parameters():
                num_params += param.numel()

            num_params = num_params / 1000**2
            print(f"Number of parameters: {num_params:.2f}M")


    def create_optim(self):
        self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.flags.lr,
                                         betas=self.flags.betas_G, weight_decay=self.flags.weight_decay)
        # amp settings
        self.amp_scaler = amp.GradScaler('cuda') if self.flags.use_amp else None

    def create_l1_loss(self):
        # Return a standard L1 loss (Mean Absolute Error)
        return torch.nn.L1Loss(reduction='mean')


    def train(self,
              epochs
              ):
        print(f"Number of total steps {len(self.data_loader) * self.flags.batch_size}")
        jj = 0
        total_loss = {}
        total_loss['mse'] = []
        total_loss['l1'] = []

        total_loss['mse_sample'] = []
        total_loss['l1_sample'] = []
        total_loss['train_time'] = []
        total_loss['valid_time'] = []

        best_loss = None
        epoch_rows = []
        s_time_all = time()
        # for epoch in tqdm(range(self.init_epoch, epochs)):
        for epoch in range(self.init_epoch, epochs):
            loss_mse_running, loss_l1_running = 0., 0.
            self.current_epoch = epoch + 1
            self.model.train()
            loop = tqdm(self.data_loader, ascii=True, desc=f'Epoch [{epoch + 1}/{epochs}]')
            s_time_epoch = time()
            for _jj, data in enumerate(loop):
                s_time = time()
                hq = data['hq'].to(self.device, dtype=torch.float)
                lq = data['lq'].to(self.device, dtype=torch.float)
                tt = torch.randint(0, self.flags.steps, (self.flags.batch_size,), device=self.device)
                if self.flags.cond_lq:
                    model_kwargs = {'lq': lq}
                else:
                    model_kwargs = None
                noise = torch.randn(size=hq.shape, device=self.device)
                compute_losses = self.base_diffusion.training_losses(
                    model=self.model,
                    x_start=hq,
                    y=lq,
                    t=tt,
                    first_stage_model=self.autoencoder,
                    model_kwargs=model_kwargs,
                    noise=noise,
                )
                if self.flags.num_gpus <= 1:
                    losses, z0_pred, z_t = self.backward_step(compute_losses, hq)
                else:
                    with self.model.no_sync():
                        losses, z0_pred, z_t = self.backward_step(compute_losses, hq)
                if self.flags.use_amp:
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                else:
                    self.optimizer.step()
                self.model.zero_grad()
                self.update_ema_model()
                with torch.no_grad():
                    loop.set_postfix(
                        mse=float(losses['mse'].mean()),
                        l1=float(losses['l1'].mean()),
                    )
                    loss_mse_running += float(losses['mse'].mean())
                    loss_l1_running += float(losses['l1'].mean())
                    total_loss['mse_sample'].append(float(losses['mse'].mean()))
                    total_loss['l1_sample'].append(float(losses['l1'].mean()))
                    total_loss['train_time'].append(time() - s_time)

            train_mse = loss_mse_running / len(self.data_loader)
            train_l1  = loss_l1_running / len(self.data_loader)
            total_loss['mse'].append(train_mse)
            total_loss['l1'].append(train_l1)
            print(f"Epoch {epoch + 1} completed in {(time() - s_time_epoch) / 60:.2f} min")
            
            # Save best model if needed
            epoch_loss = train_mse + train_l1
            if best_loss is None or epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save_to_best(path=self.flags.out_dir, name=self.flags.model_name, verbose=True)
            
            self.save_to(path=self.flags.out_dir, name=self.flags.model_name)
            
            # Append epoch metrics into the row list
            epoch_rows.append({
                "epoch": epoch + 1,
                "train_mse": train_mse,
                "train_l1": train_l1,
                "epoch_time_sec": time() - s_time_epoch
            })
        
        print(f"Training MSE per epoch: {total_loss['mse']}")
        print(f"Training L1 per epoch: {total_loss['l1']}")
        total_time = time() - s_time_all
        print('Total train time: {:.2f} hrs'.format(total_time / 3600.))
        
        # Save the epoch metrics to a CSV file instead of pickle
        csv_file = os.path.join(self.flags.out_dir, "training_metrics.csv")
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_mse", "train_l1", "validation_loss", "epoch_time_sec"])
            writer.writeheader()
            for row in epoch_rows:
                writer.writerow(row)

    @torch.no_grad()
    def update_ema_model(self):
        if self.flags.num_gpus > 1:
            dist.barrier()
        if self.rank == 0:
            source_state = self.model.state_dict()
            rate = self.flags.ema_rate
            for key, value in self.ema_state.items():
                if key in self.ema_ignore_keys:
                    self.ema_state[key] = source_state[key]
                else:
                    self.ema_state[key].mul_(rate).add_(source_state[key].detach().data, alpha=1 - rate)

    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate=1):
        loss_coef = self.flags.loss_coef
        context = torch.amp.autocast('cuda') if self.flags.use_amp else nullcontext()
        with context:
            losses, z_t, z0_pred = dif_loss_wrapper
            x0_pred = z0_pred
            losses["l1"] = self.l1_loss(x0_pred, micro_data)
            flag_nan = torch.any(torch.isnan(losses["l1"]))
            if flag_nan:
                losses["l1"] = torch.nan_to_num(losses["l1"], nan=0.0)
            losses["mse"] *= loss_coef[0]
            losses["l1"] *= loss_coef[1]
            if flag_nan:
                losses["loss"] = losses["mse"]
            else:
                losses["loss"] = losses["mse"] + losses["l1"]
            loss = losses["loss"].mean() / num_grad_accumulate
        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()
        return losses, x0_pred, z_t


    def validation(self, batch_idx):
        print("Performing validation ...")
        psnr = PeakSignalNoiseRatio().to(self.device)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        if self.rank == 0:
            if self.flags.use_ema_val:
                self.reload_ema_model()
                self.ema_model.eval()
            else:
                print(f"using model itself")
                self.model.eval()
            # self.model.eval()

            indices = np.linspace(
                0,
                self.flags.steps,
                self.flags.steps if self.flags.steps < 5 else 4,
                endpoint=False,
                dtype=np.int64,
            ).tolist()
            if not (self.flags.steps - 1) in indices:
                indices.append(self.flags.steps - 1)
            batch_size = self.flags.test_batch_size
            num_iters_epoch = math.ceil(len(self.test_data_loader) / batch_size)
            mean_psnr = mean_l1 = 0
            metrics = {}
            metrics['psnr_before'] = []
            metrics['psnr_after'] = []

            metrics['ssim_before'] = []
            metrics['ssim_after'] = []

            metrics['l1_before'] = []
            metrics['l1_after'] = []
            s_time_validation = time()

            data = next(iter(self.test_data_loader))
            im_lq = data['lq'].to(self.device, dtype=torch.float)
            im_gt = data['hq'].to(self.device, dtype=torch.float)

            num_iters = 0
            if self.flags.cond_lq:
                model_kwargs = {'lq': im_lq, }
            else:
                model_kwargs = None
            tt = torch.tensor(
                [self.flags.steps, ] * im_lq.shape[0],
                dtype=torch.int64,
            ).cuda()
            for sample in self.base_diffusion.p_sample_loop_progressive(
                    y=im_lq,
                    model=self.ema_model if self.flags.use_ema_val else self.model,
                    # model=self.model,
                    first_stage_model=self.autoencoder,
                    noise=None,
                    clip_denoised=True if self.autoencoder is None else False,
                    model_kwargs=model_kwargs,
                    device=f"cuda:{self.rank}",
                    progress=False,
            ):
                # print(f"{sample = }")
                # print(f"{indices = }")
                sample_decode = {}
                if num_iters in indices:
                    for key, value in sample.items():
                        if key in ['sample', ]:
                            sample_decode[key] = value.clamp(-1.0, 1.0)  # self.base_diffusion.decode_first_stage(
                            # value,
                            # self.autoencoder,
                            # ).clamp(-1.0, 1.0)
                    im_sr_progress = sample['sample']
                    if num_iters + 1 == 1:
                        im_sr_all = im_sr_progress
                    else:
                        im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)
                num_iters += 1
                tt -= 1



            viz_sample = torch.cat((im_gt * 0.5 + 0.5,
                                    im_lq * 0.5 + 0.5,
                                    sample_decode['sample'] * 0.5 + 0.5,
                                    ),
                                   dim=0)  # unet
            vutils.save_image(viz_sample,
                              os.path.join(self.flags.out_dir, 'samples_all_{}.png'.format(batch_idx + 1)),
                              # out_dir_outs
                              nrow=self.test_data_loader.batch_size,
                              normalize=True)

            print(f"Validation done in {(time() - s_time_validation) / 60:.2f} min")
            if not (self.flags.use_ema_val and self.flags.ema_rate is not None):
                self.model.train()

    @staticmethod
    def minimax_4d(X, min_value=0, max_value=1.0):
        _max_value = torch.amax(X, dim=(2, 3), keepdim=True)
        _min_value = torch.amin(X, dim=(2, 3), keepdim=True)

        X_std = (X - _min_value) / (_max_value - _min_value)
        return X_std * (max_value - min_value) + min_value


    def compute_nmse(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        NMSE = ||pred - gt||^2 / ||gt||^2
        Both pred and gt are numpy arrays in [-1,1].
        """
        pred = pred.astype(np.float64)
        gt   = gt.astype(np.float64)
        num = np.sum((pred - gt) ** 2)
        den = np.sum(gt ** 2) + 1e-12
        return num / den

    def inference_(self, batch_size=None, out_dir=None):
        if batch_size is None:
            batch_size = self.flags.test_batch_size
        if out_dir is None:
            out_dir = self.flags.out_dir_test

        os.makedirs(out_dir, exist_ok=True)
        print(f"Performing inference on {len(self.test_data_loader)} batches...")

        # 1) Prepare metric accumulators
        stats = {
            "All":  {"psnr": 0.0, "ssim": 0.0, "nmse": 0.0, "count": 0},
            "1.5T": {"psnr": 0.0, "ssim": 0.0, "nmse": 0.0, "count": 0},
            "3T":   {"psnr": 0.0, "ssim": 0.0, "nmse": 0.0, "count": 0},
        }
        slice_rows = []
        header = ["patient_id","slice_idx","modality","PSNR","SSIM","NMSE"]
        
        # 2) Loop
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.test_data_loader, desc="Infer"):
                # assume your dataset returns 'lq','hq','patient_id','slice_idx','modality'
                im_lq = data['lq'].to(self.device, dtype=torch.float)      # raw in [-1,1]
                im_gt = data['hq'].to(self.device, dtype=torch.float)
                #pid_batch = data['patient_id']
                #idx_batch = data['slice_idx']
                #mod_batch = data['modality']

                pid_batch = data['patient_id']
                idx_batch = data['slice_idx']
                #mod_batch = data['modality_label'].squeeze()
                # after (minimal & safe)
                mod_batch = data['modality_label']
                if torch.is_tensor(mod_batch):
                    mod_batch = mod_batch.view(-1)   # ensure 1-D even when B=1

                # 2a) run diffusion sampler to get raw output in [-1,1]
                results = self.base_diffusion.p_sample_loop(
                    y=im_lq,
                    model=self.model,
                    first_stage_model=self.autoencoder,
                    noise=None,
                    noise_repeat=False,
                    clip_denoised=True if self.autoencoder is None else False,
                    denoised_fn=None,
                    model_kwargs=( {'lq': im_lq} if self.flags.cond_lq else None ),
                    progress=False,
                )  # shape [B, C, H, W], values in [-1,1]

                # 2b) iterate per-sample
                for i in range(results.shape[0]):
                    pr_raw = results[i,0].cpu().numpy()          # numpy array H×W, in [-1,1]
                    gt_raw = im_gt[i,0].cpu().numpy()            # same
                    pid    = pid_batch[i]
                    sidx   = int(idx_batch[i])
                    mcode  = int(mod_batch[i].item())
                    mstr  = "1.5T" if mcode == 0 else "3T"
                    # 3) metrics in [-1,1] with data_range=2.0
                    p = peak_signal_noise_ratio(gt_raw, pr_raw, data_range=2.0)
                    s = structural_similarity(gt_raw, pr_raw, data_range=2.0)
                    n = self.compute_nmse(gt_raw, pr_raw)

                    # 4) accumulate
                    for key in ("All", mstr):
                        stats[key]["psnr"]  += p
                        stats[key]["ssim"]  += s
                        stats[key]["nmse"]  += n
                        stats[key]["count"] += 1

                    # 5) record row
                    slice_rows.append([pid, sidx, mstr, f"{p:.4f}", f"{s:.4f}", f"{n:.6f}"])

                    # 6) save PNG if desired (rescale to [0,255])
                    img8 = ((pr_raw + 1)/2 * 255).clip(0,255).astype(np.uint8)
                    save_dir = os.path.join(out_dir, mstr, pid)
                    os.makedirs(save_dir, exist_ok=True)
                    Image.fromarray(img8).save(os.path.join(save_dir, f"slice_{sidx:03d}.png"))

        # 7) write per-slice CSV
        slice_csv = os.path.join(out_dir, "slice_metrics.csv")
        with open(slice_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(slice_rows)

        # 8) compute & print averages, write summary CSV
        summary_csv = os.path.join(out_dir, "slice_metrics_summary.csv")
        with open(summary_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["modality","PSNR_mean","SSIM_mean","NMSE_mean"])
            print("\n=== Inference Results ===")
            for key in ("All","1.5T","3T"):
                cnt = stats[key]["count"]
                if cnt==0:
                    print(f"{key}: no samples.")
                    continue
                pm = stats[key]["psnr"] / cnt
                sm = stats[key]["ssim"] / cnt
                nm = stats[key]["nmse"] / cnt
                print(f"{key:4s} → PSNR {pm:.4f}, SSIM {sm:.4f}, NMSE {nm:.6f}")
                w.writerow([key, f"{pm:.4f}", f"{sm:.4f}", f"{nm:.6f}"])

        print(f"\n✅ Per-slice     → {slice_csv}")
        print(f"✅ Summary     → {summary_csv}")


    def to_int(self, x):
        x = x.cpu().numpy() * 0.5 + 0.5
        x = x * 255.0
        return x.round().astype(np.uint8)
    '''
    def load_model_inference(self,
                  path='',
                  name=None,
                  verbose=True):
        if name is None:
            name = 'content.pth'
    '''

    def load_model_inference(self, path='', name=None, verbose=True):
        if name is None:
            name = 'content.pth'
        checkpoint_file = os.path.join(path, name)
        if not os.path.exists(checkpoint_file):
            if verbose:
                print(f"Warning: Checkpoint file {checkpoint_file} does not exist. Skipping checkpoint loading.")
            return  # Skip loading if file does not exist
        if verbose:
            print('\nLoading models and checkpoint from {} ...'.format(name))
        #torch.serialization.add_safe_globals([("argparse.Namespace", argparse.Namespace)])
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only = False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        torch.cuda.empty_cache()


        # if not name.endswith('.pt'):
        #     name += '.pt'
        if verbose:
            print('\nLoading models and checkpoint from {} ...'.format(name))

        print(f"=> Loaded checkpoint from {path}")
        checkpoint_file = os.path.join(path, name)
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only = False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        torch.cuda.empty_cache()



    def save_to(self,
                path='',
                name=None,
                verbose=True):
        content = {'epoch': self.current_epoch,
                   'args': self.flags,
                   'model_state_dict': self.model.state_dict(),
                   'optimizer_state_dict': self.optimizer.state_dict(),
                   }

        if name is None:
            name = self.name

        if verbose:
            print('\nSaving models {} ...'.format(name))
        torch.save(self.model.state_dict(), os.path.join(path, '{}.pt'.format(name)))
        if self.amp_scaler is not None:
            content['amp_scaler'] = self.amp_scaler.state_dict()
        torch.save(content, os.path.join(path, 'content.pth'))

        torch.save(self.ema_state, os.path.join(path, 'ema_model.pth'))

    def save_to_best(self,
                path='',
                name=None,
                verbose=True):
        content = {'epoch': self.current_epoch,
                   'args': self.flags,
                   'model_state_dict': self.model.state_dict(),
                   'optimizer_state_dict': self.optimizer.state_dict(),
                   }

        if name is None:
            name = self.name

        if verbose:
            print('\nSaving the best models {} ...'.format(name))
        torch.save(self.model.state_dict(), os.path.join(path, '{}_best.pt'.format(name)))
        if self.amp_scaler is not None:
            content['amp_scaler'] = self.amp_scaler.state_dict()
        torch.save(content, os.path.join(path, 'content_best.pth'))

        torch.save(self.ema_state, os.path.join(path, 'ema_model_best.pth'))
    def reload_ema_model(self):
        if self.rank == 0:
            if self.num_gpus > 1:
                model_state = {key[7:]: value for key, value in self.ema_state.items()}
            else:
                model_state = self.ema_state
            self.ema_model.load_state_dict(model_state)
