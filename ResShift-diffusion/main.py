"""
Train U-Net
"""
import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as udata

import utils
from build import Model
# from build_seg import Model
#from datasets.load_data import MotionCorruptedMRIDataset2D as ImageDataset
from datasets.load_custom_data import PNG2DDataset as ImageDataset
from datasets.load_data_prostate import Res_SRDiff_onFly
from torch.utils.data import DataLoader
FLAGS = None


def main():
    print(FLAGS.message)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    num_gpus = torch.cuda.device_count()

    if FLAGS.train:
        train_datset = ImageDataset(
            dir_15T=FLAGS.dir_15T,
            dir_3T=FLAGS.dir_3T,
            split="train",
            augment=True
        )

        # train_datset = ImageDataset(gt_name=all_valid_gt)
        print(f"Loading data ...")

        if num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                train_datset,
                num_replicas=num_gpus,
                rank=int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0,
            )
        else:
            sampler = None


        train_dataloader = DataLoader(train_datset,
                                      batch_size=FLAGS.batch_size // num_gpus,
                                      shuffle=True,
                                      num_workers=min(FLAGS.train_num_workers, 4),
                                      drop_last=True,
                                      prefetch_factor= FLAGS.prefetch_factor,
                                      pin_memory=True,
                                      sampler=sampler,)

        valid_dataset = ImageDataset(
            dir_15T=FLAGS.dir_15T,
            dir_3T=FLAGS.dir_3T,
            split="val",
            augment=True
        )

        valid_dataloader = DataLoader(valid_dataset,
                                      batch_size=FLAGS.test_batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      drop_last=True)

        print('Creating model...\n')


        model = Model(name='Super_resolution',
                      device=device,
                      data_loader=train_dataloader,
                      test_data_loader=valid_dataloader,
                      FLAGS=FLAGS)

        # Count *all* parameters (trainable + frozen)
        #total_params = sum(p.numel() for p in model.parameters())

        # Count only the parameters that will be updated during training
        #trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        #print(f"Total parameters : {total_params:,}  ({total_params/1e6:.2f} M)")
        #print(f"Trainable params : {trainable_params:,}  ({trainable_params/1e6:.2f} M)")

        model.train(epochs=FLAGS.epochs)


    else:
        test_datset = ImageDataset(
            dir_15T=FLAGS.dir_15T,
            dir_3T=FLAGS.dir_3T,
            split="test",
            augment=True
        )


        test_dataloader = DataLoader(test_datset,
                                      batch_size=FLAGS.test_batch_size,
                                      shuffle=False,
                                      num_workers=4,
                                      pin_memory=True,
                                      drop_last=True)

        model = Model(name = FLAGS.model,
                      device=device,
                      data_loader=None,
                      test_data_loader=test_dataloader,
                      FLAGS=FLAGS)


        '''
        print('Loading Model')
        model.load_model_inference(path=FLAGS.out_dir)
        print('Evaluating Model')
        model.inference_(batch_size=FLAGS.test_batch_size, out_dir=FLAGS.out_dir_test)
        print('done')
        '''

        import time

        print('Evaluating Model')

        # --- Start timing ---
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        model.inference_(batch_size=FLAGS.test_batch_size, out_dir=FLAGS.out_dir_test)

        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f'Inference completed in {elapsed_time_ms / 1000:.3f} seconds')
        print('done')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UNetSwinTransformer')
    parser.add_argument('--message', type=str, default='Efficient DDPM model for LR --> HR', help='Diffusion model')
    parser.add_argument('--model', type=str, default='Super_resolution', help='Diffusion model')
    parser.add_argument('--model_name', type=str, default='unet_swuin', help='Diffusion model')
    parser.add_argument('--region', type=str, default='prostate', help='brain, prostate')

    parser.add_argument('--cuda', type=utils.boolean_string, default=True, help='enable CUDA.')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPU used.')
    parser.add_argument('--train', type=utils.boolean_string, default=False, help='train mode or eval mode.')
    parser.add_argument('--resume', type=utils.boolean_string, default=False, help='train mode or eval mode.')
    parser.add_argument('--train_tf_logging', type=utils.boolean_string, default=False, help='tensorboard logging')
    parser.add_argument('--dir_15T', type=str, default='./data/15T_to_7T', help='Path to the 15T dataset root.')
    parser.add_argument('--dir_3T', type=str, default='./data/3T_to_7T', help='Path to the 3T dataset root.')


    parser.add_argument('--out_dir', type=str, default='./outputs/resshift-diffusion/train', help='Directory for train output.')

    parser.add_argument('--out_dir_test', type=str, default='./outputs/resshift-diffusion/test', help='Directory for test output.')
    parser.add_argument('--model_ckpt_path', type=str, default=None, help='Directory for saved model.')

    ### Input image parameters
    parser.add_argument('--image_size', type=int, default=256, help='Image size 384')
    parser.add_argument('--in_channels', type=int, default=1, help="input channels")
    parser.add_argument('--out_channels', type=int, default=1, help="output channels")
    parser.add_argument('--model_channels', type=int, default=64, help='model channels')
    parser.add_argument('--num_res_blocks', type=tuple, default=[2, 2, 2, 2], help="Number of residual blocks")
    parser.add_argument('--attention_resolutions', type=tuple, default=(64,32,16,8), help="Attention resolutions")
    parser.add_argument('--cond_lq', type=utils.boolean_string, default=True, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')


    ### Diffusion model parameters

    # Arguments for parameters (add as many as required for your function or class)
    parser.add_argument('--sf', type=float, default=1.0, help='Scale factor')
    parser.add_argument('--schedule_name', type=str, default='exponential', help='Name of the schedule')
    parser.add_argument('--schedule_kwargs', type=dict, default={'power': 0.3}, help='Additional schedule arguments as a dictionary')
    parser.add_argument('--etas_end', type=float, default=0.99,  help='Ending value for etas')
    parser.add_argument('--steps', type=int, default=15, help='Number of steps')
    parser.add_argument('--min_noise_level', type=float, default=0.2, help='Minimum noise level')
    parser.add_argument('--kappa', type=float, default=2.0, help='Kappa value')
    parser.add_argument('--weighted_mse', action='store_true', help='Use weighted mean squared error')
    parser.add_argument('--predict_type', type=str, default='xstart', help='Type of prediction')
    parser.add_argument('--timestep_respacing', type=int, default=None, help='Timestep respacing value')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Scale factor')
    parser.add_argument('--normalize_input', action='store_true',  help='Whether to normalize input')
    parser.add_argument('--latent_flag', action='store_true', help='Flag for using latent variables')


    ### Autoencoder parameters
    parser.add_argument('--ch_mult', type=int, nargs='+', default=[1, 2, 4], help='Channel multipliers')
    parser.add_argument('--ae_num_res_blocks', type=int, default=2, help='AutoEncoder number of residual blocks')
    parser.add_argument('--ae_double_z', type=utils.boolean_string, default=False, help='AutoEncoder number of residual blocks')

    parser.add_argument('--ckpt_path', type=str, default='./outputs/resshift-diffusion/autoencoder_vq.pth',
                        help='Path to the checkpoint file')
    parser.add_argument('--tune_decoder', type=bool, default=True,
                        help='Whether to fine-tune the decoder')

    # Parameters configuration
    parser.add_argument('--embed_dim', type=int, default=1, help='Dimensionality of the embeddings in the VQ model')
    parser.add_argument('--n_embed', type=int, default=8192,  help='Number of embeddings in the codebook')

    # ddconfig nested parameters
    parser.add_argument('--ddconfig_double_z', type=bool, default=False,help='Whether to double the latent channels')
    parser.add_argument('--ddconfig_z_channels', type=int, default=1, help='Number of latent channels')
    parser.add_argument('--ddconfig_resolution', type=int, default=256, help='Resolution of the input images')
    parser.add_argument('--ddconfig_in_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--ddconfig_out_ch', type=int, default=1, help='Number of output channels')
    parser.add_argument('--ddconfig_ch', type=int, default=128, help='Base number of channels')
    parser.add_argument('--ddconfig_ch_mult', type=int, nargs='+', default=[1, 2, 4], help='Channel multiplier for each level')
    parser.add_argument('--ddconfig_num_res_blocks', type=int, default=2, help='Number of residual blocks per level')
    parser.add_argument('--ddconfig_attn_resolutions', type=int, nargs='+', default=[], help='Resolutions at which attention is applied')
    parser.add_argument('--ddconfig_dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--ddconfig_padding_mode', type=str, default='zeros', help='Padding mode for convolutional layers')

    ### Training and validation parameters
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='size of batches in training')
    parser.add_argument('--test_batch_size', type=int, default=1, help='size of batches in inference')
    parser.add_argument('--betas_G', type=tuple, default=(0.9, 0.999), help='learning rate')
    parser.add_argument('--train_num_workers', type=int, default=8, help='number of CPU to load data')

    # Learning rate configuration
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--lr_min', type=float, default=2e-5, help='Minimum learning rate')
    parser.add_argument('--lr_schedule', type=str, default='cosin', help='Learning rate schedule type')
    parser.add_argument('--warmup_iterations', type=int, default=5000, help='Number of warmup iterations')

    # Dataloader configuration
    # parser.add_argument('--batch', type=int, nargs=2, default=[8, 4], help='Batch size for train and validation')
    # parser.add_argument('--microbatch', type=int, default=8, help='Microbatch size for gradient accumulation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--prefetch_factor', type=int, default=8, help='Prefetch factor for data loading')

    # Optimization settings
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--ema_rate', type=float, default=0.999, help='Exponential moving average rate')
    # parser.add_argument('--iterations', type=int, default=400000, help='Total number of training iterations')

    # Save and logging settings
    # parser.add_argument('--save_freq', type=int, default=10000, help='Frequency of saving checkpoints')
    # parser.add_argument('--log_freq', type=int, nargs=3, default=[500, 5000, 4],
    #                     help='Log frequencies for training loss, training images, validation images')
    parser.add_argument('--loss_coef', type=float, nargs=2, default=[4.0, 1.0],
                        help='Loss coefficients for [mse, lpips]; default=[1.0, 1.0]')
    parser.add_argument('--local_logging', type=bool, default=True, help='Enable local image logging')
    parser.add_argument('--tf_logging', type=bool, default=False, help='Enable TensorBoard logging')

    # Validation settings
    parser.add_argument('--use_ema_val', type=bool, default=False, help='Use EMA model for validation')
    parser.add_argument('--val_freq', type=int, default=10000, help='Frequency of validation')
    parser.add_argument('--val_y_channel', type=bool, default=True, help='Evaluate PSNR on Y channel of images')
    parser.add_argument('--val_resolution', type=int, default=None, help='Resolution used during validation')
    parser.add_argument('--val_padding_mode', type=str, default='reflect', help='Padding mode for validation')

    # Training settings
    parser.add_argument('--use_amp', type=bool, default=False, help='Use Automatic Mixed Precision (AMP) for training')
    parser.add_argument('--seed', type=int, default=123456, help='Random seed for reproducibility')
    parser.add_argument('--global_seeding', type=bool, default=False, help='Use global seeding')

    # Model compilation settings
    parser.add_argument('--compile_flag', type=bool, default=False, help='Enable model compilation')
    parser.add_argument('--compile_mode', type=str, default='max-autotune', help='Mode for model compilation')

    # Augmentation settings for latent LPIPS
    parser.add_argument('--augpipe_flag', type=bool, default=True, help='Enable augmentation for latent LPIPS')
    parser.add_argument('--augpipe_target', type=str, default='utils.augment_ada.AugmentPipeGeoCut',
                        help='Augmentation target class')
    parser.add_argument('--augpipe_prob', type=float, default=0.5, help='Probability of applying augmentation')
    parser.add_argument('--augpipe_scale', type=float, default=0.5, help='Scale parameter for augmentation')
    parser.add_argument('--augpipe_rotate', type=float, default=0.5, help='Rotation parameter for augmentation')
    parser.add_argument('--augpipe_aniso', type=float, default=0.5,
                        help='Anisotropic scaling parameter for augmentation')
    parser.add_argument('--augpipe_xfrac', type=float, default=0.5, help='X-fraction parameter for augmentation')
    parser.add_argument('--augpipe_cutout', type=float, default=0.5, help='Cutout parameter for augmentation')

    torch.cuda.empty_cache()
    torch.cuda.synchronize()


    FLAGS = parser.parse_args()
    FLAGS.cuda = FLAGS.cuda and torch.cuda.is_available()
    torch.set_float32_matmul_precision('high')


    if FLAGS.seed is not None:
        torch.manual_seed(FLAGS.seed)
        if FLAGS.cuda:
            torch.cuda.manual_seed(FLAGS.seed)
            torch.cuda.manual_seed_all(FLAGS.seed)
        np.random.seed(FLAGS.seed)

    cudnn.benchmark = True

    if FLAGS.train:
        if FLAGS.resume:
            log_file = os.path.join(FLAGS.out_dir, 'log_resume.txt')
            print("Logging to {}\n".format(log_file))
            sys.stdout = utils.StdOut(log_file)
        else:
            utils.clear_folder(FLAGS.out_dir)
            log_file = os.path.join(FLAGS.out_dir, 'log.txt')
            print("Logging to {}\n".format(log_file))
            sys.stdout = utils.StdOut(log_file)
    else:
        utils.clear_folder(FLAGS.out_dir_test)
        log_file = os.path.join(FLAGS.out_dir_test, 'log1.txt')
        print("Logging to {}\n".format(log_file))
        sys.stdout = utils.StdOut(log_file)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version (PyTorch built with): {torch.version.cuda}")

    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # print("Similar to out3 with my perceptual loss function\nSelf Att. Unet\nNN loss and LeakyReLU activations.")
    print(" " * 9 + "Args" + " " * 9 + "|    " + "Type" + \
          "    |    " + "Value")
    print("-" * 50)
    for arg in vars(FLAGS):
        arg_str = str(arg)
        var_str = str(getattr(FLAGS, arg))
        type_str = str(type(getattr(FLAGS, arg)).__name__)
        print("  " + arg_str + " " * (20-len(arg_str)) + "|" + \
              "  " + type_str + " " * (10-len(type_str)) + "|" + \
              "  " + var_str)

    main()

