import argparse
import logging
import gc
import os

import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from evaluate import evaluate
from unet import UNet3,UNet5, UNet7, UNet9
from utils.data_loading import BasicDataset, CarvanaDataset, Custom_dataset
from utils.dice_score import dice_loss

from torchvision import transforms
from npz_preprocess import RandomGenerator

base_dir = Path('D:/UNet_py/dataset_split/npzgood')
list_dir = Path('D:/UNet_py/dataset_split/dataset_split')
# dir_img = Path('D:/copy_model/npz2good/original_images')
# dir_mask = Path('D:/copy_model/npz2good/label_images')
dir_checkpoint = Path('./checkpoints/')


def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)



def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    try:
        # base_dir是所有npz文件存放的路径，list_dir是记录挑选进训练集（train）的病例的txt文件
        train_set = Custom_dataset(base_dir, list_dir, split="train",
                                   transform=transforms.Compose(
                                       [RandomGenerator(output_size=[224, 224])]))
        val_set = Custom_dataset(base_dir, list_dir, split="val", transform=None)
    except (AssertionError, RuntimeError, IndexError):
        print(f"data loading error")

    # 2. Split into train / validation partitions
    n_train = len(train_set)
    n_val = len(val_set)

    # 3. 训练和验证数据加载器
    loader_args = dict(num_workers=0, pin_memory=False)
    # os.cpu_count(),True
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn,
                              **loader_args)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP


    #optimizer = optim.RMSprop(model.parameters(),lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training

    low_accuracy_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['label']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded imge have {images.shape[1]} channels. Please check that ' \
                    'the imge are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                # 自动混合精度训练 二元交叉熵 dice混合
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                # 使用梯度缩放器进行反向传播和梯度计算。
                # 对梯度进行裁剪，防止梯度爆炸。
                # 更新优化器参数，并更新梯度缩放器的缩放值。
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                '''division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0: '''

        histograms = {}
        for tag, value in model.named_parameters():
            tag = tag.replace('/', '.')
            if not (torch.isinf(value) | torch.isnan(value)).any():
                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

        # 验证阶段
        val_score = evaluate(model, val_loader, device, "val")
        scheduler.step(val_score)

        logging.info('Validation Dice score: {}'.format(val_score))
        # try:
        #     experiment.log({
        #         'learning rate': optimizer.param_groups[0]['lr'],
        #         'validation Dice': val_score,
        #         'imge': wandb.Image(images[0].cpu()),
        #         'masks': {
        #             'true': wandb.Image(true_masks[0].float().cpu()),
        #             'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
        #         },
        #         'step': global_step,
        #         'epoch': epoch,
        #         **histograms
        #     })
        # except:
        #     pass
        # if save_checkpoint and epoch % save_interval == 0:
        # Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        # state_dict = model.state_dict()
        # state_dict['mask_values'] = dataset.mask_values
        # torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
        # logging.info(f'Checkpoint {epoch} saved!')
        if epoch >= 20:
            if val_score < 0.65:
                # 如果准确率低于 0.65，增加计数器
                low_accuracy_count += 1
            else:
                # 如果准确率高于 0.65，重置计数器
                low_accuracy_count = 0

            # 如果连续三轮都低于 0.65，跳出循环
            if low_accuracy_count >= 3:
                print(f"Stopping early at epoch {epoch} due to low accuracy.")
                break

    if val_score >= 0.80:
        save_dir = "D:/UNet_py/Pytorch-UNet-master/Pytorch-UNet-master/good_model"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_name = f"model_dice_{val_score:.4f}.pth"
        save_path = os.path.join(save_dir, model_name)
        torch.save(model, save_path)  # 保存整个模型
        print(f"model has saved,val_sccore={val_score}")


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=60, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet9(3, 2, 64, 3, 2, 0, 0, 0, bilinear=False)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
