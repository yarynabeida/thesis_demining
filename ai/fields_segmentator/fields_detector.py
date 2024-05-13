import os
import zipfile
import json
import tempfile
import argparse

import glob
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import jaccard_score
from tqdm import tqdm


class FieldDetector(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder_layers = torch.nn.ModuleList([
            torch.nn.ModuleList([
                torch.nn.Conv2d(3, 16, kernel_size=5, padding="same"),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(16, 16, kernel_size=3, padding="same"),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.BatchNorm2d(16),
            ]),
            torch.nn.ModuleList([
                torch.nn.Conv2d(16, 32, kernel_size=3, padding="same"),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(32, 32, kernel_size=3, padding="same"),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.BatchNorm2d(32),
            ]),
            torch.nn.ModuleList([
                torch.nn.Conv2d(32, 64, kernel_size=3, padding="same"),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(64, 64, kernel_size=3, padding="same"),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.BatchNorm2d(64),
            ]),
            torch.nn.ModuleList([
                torch.nn.Conv2d(64, 128, kernel_size=3, padding="same"),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(128, 128, kernel_size=3, padding="same"),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.BatchNorm2d(128),
            ]),
            torch.nn.ModuleList([
                torch.nn.Conv2d(128, 256, kernel_size=3, padding="same"),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(256, 256, kernel_size=3, padding="same"),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.BatchNorm2d(256),
            ])
        ])
        self.decoder_layers = torch.nn.ModuleList([
            torch.nn.ModuleList([
                torch.nn.Conv2d(256, 128, kernel_size=3, padding="same"),
                torch.nn.ReLU(inplace=True),
                torch.nn.Upsample(scale_factor=2, mode="nearest")
            ]),
            torch.nn.ModuleList([
                torch.nn.Conv2d(256, 128, kernel_size=3, padding="same"),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(128),
                torch.nn.Conv2d(128, 64, kernel_size=3, padding="same"),
                torch.nn.ReLU(inplace=True),
                torch.nn.Upsample(scale_factor=2, mode="nearest")
            ]),
            torch.nn.ModuleList([
                torch.nn.Conv2d(128, 64, kernel_size=3, padding="same"),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(64),
                torch.nn.Conv2d(64, 32, kernel_size=3, padding="same"),
                torch.nn.ReLU(inplace=True),
                torch.nn.Upsample(scale_factor=2, mode="nearest")
            ]),
            torch.nn.ModuleList([
                torch.nn.Conv2d(64, 32, kernel_size=3, padding="same"),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(32),
                torch.nn.Conv2d(32, 16, kernel_size=3, padding="same"),
                torch.nn.ReLU(inplace=True),
                torch.nn.Upsample(scale_factor=2, mode="nearest"),
            ]),
            torch.nn.ModuleList([
                torch.nn.Conv2d(32, 16, kernel_size=3, padding="same"),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(16),
                torch.nn.Conv2d(16, 3, kernel_size=3, padding="same"),
                torch.nn.ReLU(inplace=True),
                torch.nn.Upsample(scale_factor=2, mode="nearest")
            ]),
            nn.ModuleList([
                nn.Conv2d(6, num_classes, kernel_size=3, padding="same"),
                nn.Conv2d(num_classes, num_classes, kernel_size=3, padding="same")
            ]),
        ])

    def forward(self, x):
        enc_0 = torch.nn.Sequential(*self.encoder_layers[0])(x)
        enc_1 = torch.nn.Sequential(*self.encoder_layers[1])(enc_0)
        enc_2 = torch.nn.Sequential(*self.encoder_layers[2])(enc_1)
        enc_3 = torch.nn.Sequential(*self.encoder_layers[3])(enc_2)
        enc_4 = torch.nn.Sequential(*self.encoder_layers[4])(enc_3)
        dec_0 = torch.nn.Sequential(*self.decoder_layers[0])(enc_4)
        cat_0 = torch.cat((enc_3, dec_0), axis=1)
        dec_1 = torch.nn.Sequential(*self.decoder_layers[1])(cat_0)
        cat_1 = torch.cat((enc_2, dec_1), axis=1)
        dec_2 = torch.nn.Sequential(*self.decoder_layers[2])(cat_1)
        cat_2 = torch.cat((enc_1, dec_2), axis=1)
        dec_3 = torch.nn.Sequential(*self.decoder_layers[3])(cat_2)
        cat_3 = torch.cat((enc_0, dec_3), axis=1)
        dec_4 = torch.nn.Sequential(*self.decoder_layers[4])(cat_3)
        cat_4 = torch.cat((x, dec_4), axis=1)
        dec_5 = torch.nn.Sequential(*self.decoder_layers[5])(cat_4)
        return dec_5


class DataSetSegmentation(Dataset):
    def __init__(self, folder_path, train_fraction, is_train, transform=None, num_classes=1):
        super(DataSetSegmentation, self).__init__()
        self.transform = transform
        self.num_classes = num_classes
        self.img_files = []
        self.mask_files = []
        self.train_info = {}

        subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        for subfolder in subfolders:
            img_folder = os.path.join(subfolder, "images")
            mask_folder = os.path.join(subfolder, "masks")
            img_files = sorted(glob.glob(os.path.join(img_folder, "*.png")))
            self.train_info[os.path.basename(subfolder)] = len(img_files)
            if is_train:
                x = round(len(img_files) * train_fraction)
                self.img_files.extend(img_files[:x])
            else:
                x = round(len(img_files) * train_fraction)
                self.img_files.extend(img_files[x:])

            for img_path in img_files[:x] if is_train else img_files[x:]:
                self.mask_files.append(os.path.join(mask_folder, os.path.basename(img_path)))

    def get_train_info(self):
        return self.train_info

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.num_classes == 1:
            mask = np.where(mask > 0, 1, 0).astype(np.float32)
            mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
        else:
            mask_one_hot = np.zeros((*mask.shape, self.num_classes), dtype=np.float32)
            for i, value in enumerate(
                    [0, 155, 255][:self.num_classes]):
                mask_one_hot[:, :, i] = (mask == value).astype(np.float32)
            mask = mask_one_hot

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

    def __len__(self):
        return len(self.img_files)


def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device


def get_loaders(inp_dir, batch_size, train_transform, val_transform, train_fraction, num_classes):
    train_ds = DataSetSegmentation(inp_dir, train_fraction, is_train=True, transform=train_transform,
                                   num_classes=num_classes)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = DataSetSegmentation(inp_dir, train_fraction, is_train=False, transform=val_transform,
                                 num_classes=num_classes)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    train_info = train_ds.get_train_info()
    return train_loader, val_loader, train_info


def get_transforms(image_height, image_width, is_train=True):
    if is_train:
        transform = A.Compose(
            [
                A.Resize(height=image_height, width=image_width),
                A.ColorJitter(p=0.2),
                A.RandomBrightnessContrast(p=0.5),
                A.HorizontalFlip(p=0.5),
                ToTensorV2(),
            ],
        )
    else:
        transform = A.Compose(
            [
                A.Resize(height=image_height, width=image_width),
                ToTensorV2(),
            ],
        )
    return transform


def validation_fn(epoch, loader, model, loss_fn, steps_in_val_epoch, writer, device, num_classes):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    avg_loss_val = 1.0

    model.eval()
    with torch.no_grad():
        for batch_idx, (img, mask) in enumerate(tqdm(loader), start=1):
            img = img.to(device)
            mask = mask.to(device=device)
            predictions = model(img)
            predictions = predictions.permute(0, 2, 3, 1)
            if num_classes == 1:
                loss = loss_fn(predictions.squeeze(), mask.squeeze().float())
            else:
                loss = loss_fn(predictions, mask)

            preds = torch.sigmoid(predictions)
            preds = (preds > 0.5).float()
            num_correct += (preds == mask).sum()
            num_pixels += torch.numel(preds)
            dice_score += dice_coefficient(preds, mask)
            avg_loss_val = avg_loss_val * batch_idx / (batch_idx + 1) + loss / (batch_idx + 1)

            cur_iteration = batch_idx + epoch * steps_in_val_epoch
            writer.add_scalar("BatchLoss/val", loss, cur_iteration)

    steps_in_val_epoch = batch_idx + 1
    writer.add_scalar("AverageLoss/val", avg_loss_val, epoch + 1)
    print(f"[Epoch {epoch}] Average validation loss: {avg_loss_val}. Steps in epoch: {batch_idx + 1}")
    print(f"Got {num_correct}/{num_pixels} with pixel accuracy {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)*100:.2f}")
    model.train()
    return avg_loss_val, steps_in_val_epoch


def train_fn(epoch, loader, model, optimizer, loss_fn, lr, steps_in_train_epoch, writer, device, num_classes):
    avg_loss_train = 1.0
    loop = tqdm(loader)
    for batch_idx, (image, mask) in enumerate(tqdm(loader), start=1):
        optimizer.zero_grad()
        image = image.to(device=device)
        mask = mask.to(device=device)

        predictions = model(image)
        predictions = predictions.permute(0, 2, 3, 1).squeeze(-1)
        mask = mask.squeeze(-1)
        if num_classes == 1:
            loss = loss_fn(predictions.squeeze(), mask.squeeze().float())
        else:
            loss = loss_fn(predictions, mask)

        avg_loss_train = avg_loss_train * batch_idx / (batch_idx + 1) + loss / (batch_idx + 1)
        cur_iteration = batch_idx + epoch * steps_in_train_epoch
        writer.add_scalar("LearningRate", lr, cur_iteration)
        writer.add_scalar("BatchLoss/train", loss, cur_iteration)
        writer.add_scalar("AverageLoss/train", avg_loss_train, cur_iteration)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    steps_in_train_epoch = (batch_idx + 1)
    print("[Epoch %d] Average training loss: %f. Steps in epoch: %d" % (epoch, avg_loss_train, batch_idx+1))
    return steps_in_train_epoch


def train(train_loader, val_loader, model, optimizer, loss_fn, lr_scheduler, writer, num_classes, num_epoch, save_dir,
          device, batch_size, height, width, task, training_info_dict):
    steps_in_train_epoch = 1
    steps_in_val_epoch = 1

    for epoch in range(num_epoch):
        lr = lr_scheduler.get_last_lr()[0]
        print(f"EPOCH: {str(epoch)}, Learning rate: {str(lr)}")
        steps_in_train_epoch = train_fn(epoch, train_loader, model, optimizer, loss_fn, lr, steps_in_train_epoch,
                                        writer, device, num_classes)
        lr_scheduler.step()
        avg_loss, steps_in_val_epoch = validation_fn(epoch, val_loader, model, loss_fn, steps_in_val_epoch, writer,
                                                     device, num_classes)

        avg_loss_value = avg_loss.item()
        archive_name = f"{task}-{num_classes}_{epoch}_{avg_loss_value:.3f}"

        checkpoint_path = os.path.join(save_dir, f"{archive_name}.pth")
        info_json_path = os.path.join(save_dir, f"{archive_name}.json")

        torch.save(model, checkpoint_path)

        training_info_dict["post_training_info"] = {
            "epoch": epoch,
            "learning_rate": lr,
            "average_loss": avg_loss_value,
            "checkpoint": f"{archive_name}.pth"
        }
        with open(info_json_path, "w") as f:
            json.dump(training_info_dict, f, indent=4)

        zip_path = os.path.join(save_dir, f"{archive_name}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as myzip:
            myzip.write(checkpoint_path, os.path.basename(checkpoint_path))
            myzip.write(info_json_path, os.path.basename(info_json_path))

        os.remove(checkpoint_path)
        os.remove(info_json_path)

    writer.flush()

    modelfile = os.path.join(save_dir, f"{task}-{num_classes}-{epoch}.onnx")
    input_shape = (batch_size, 3, height, width)
    x = torch.randn(input_shape).to(device)
    traced_model = torch.jit.trace(model, x)
    torch.onnx.export(traced_model, x, modelfile, verbose=True, input_names=["input"], output_names=["output"])

    writer.close()


def create_tensorboard(train_loader, model, device, SAVE_DIR):
    tensorboarddir = os.path.join(SAVE_DIR, "tensorboard")
    writer = SummaryWriter(log_dir=tensorboarddir)
    dataloader_iter = iter(train_loader)
    image_sample, _ = next(dataloader_iter)
    image_sample = image_sample.to(device)
    writer.add_graph(model, image_sample)
    return writer


def dice_loss(output, target):
    return 1 - dice_coefficient(output, target)


def dice_coefficient(mask_pred_logits, mask_true):
    SMOOTH = 1.0e-3
    mask_pred = torch.sigmoid(mask_pred_logits)
    intersection = (mask_pred*mask_true).sum(axis=[-2, -1])
    union = mask_pred.sum(axis=[-2, -1]) + mask_true.sum(axis=[-2, -1])
    iou = (2.0*intersection + SMOOTH)/(union + SMOOTH)
    dice = torch.mean(iou)
    return dice


def dice_bce_loss(mask_pred_logits, mask_true):
    dice_loss_val = dice_loss(mask_pred_logits, mask_true)
    bce_loss_val  = torch.nn.functional.binary_cross_entropy_with_logits(mask_pred_logits, mask_true)
    return (dice_loss_val + bce_loss_val)/2


def iou_coefficient(y_true, y_pred):
    jaccard = jaccard_score(y_true, y_pred)
    return jaccard


def dice_metric(y_true, y_pred):
    jaccard = jaccard_score(y_true, y_pred)
    return 2*jaccard / (1 + jaccard)


def load_model_from_zip(zipfile_path, device):
    if not os.path.isfile(zipfile_path):
        raise FileNotFoundError(f"{zipfile_path} does not exist")

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        checkpoint_file = [f for f in os.listdir(temp_dir) if f.endswith(".pth")][0]
        json_file = [f for f in os.listdir(temp_dir) if f.endswith(".json")][0]

        checkpoint_path = os.path.join(temp_dir, checkpoint_file)
        model = torch.load(checkpoint_path, map_location=torch.device(device))
        model.to(device)
        model.eval()

        json_path = os.path.join(temp_dir, json_file)
        with open(json_path, 'r') as f:
            training_info_dict = json.load(f)

    return model, training_info_dict


def test_segmentation_model(model, test_loader, device, threshold):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            predictions = model(images)
            predictions = predictions.permute(0, 2, 3, 1)
            preds = torch.sigmoid(predictions)
            preds = (preds > threshold).cpu().numpy().astype(
                np.uint8)

            masks_cpu = masks.squeeze().cpu().numpy().astype(np.uint8)

            y_true.extend(masks_cpu.flatten())
            y_pred.extend(preds.flatten())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    iou = iou_coefficient(y_true, y_pred)
    dice = dice_metric(y_true, y_pred)

    return accuracy, precision, recall, f1, iou, dice


def morphological_processing(predicted_mask, kernel_size=7, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(predicted_mask.astype(np.uint8), kernel, iterations=iterations)
    eroded_mask = cv2.erode(dilated_mask, kernel, iterations=iterations)
    return eroded_mask


def visualize(folder_path, model, output_folder, num_classes, image_height, image_width, threshold, device):
    class_colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}

    image_files = glob.glob(os.path.join(folder_path, "*.png"))
    with torch.no_grad():
        for image_file in image_files:
            image = cv2.imread(image_file)
            image = cv2.resize(image, (image_height, image_width))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            scaled_image = image.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(scaled_image).permute(2, 0, 1).unsqueeze(0).to(device)

            predictions = model(image_tensor).float()

            if num_classes > 1:
                pred_mask = torch.argmax(predictions, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            else:
                pred_mask = torch.sigmoid(predictions)
                pred_mask = (pred_mask > threshold).squeeze().cpu().numpy().astype(np.uint8)
                output_file = os.path.join(output_folder, os.path.basename(image_file))
                pred_mask_255 = 255.*pred_mask
                cv2.imwrite(output_file, pred_mask_255)

            image_contours = image.copy()
            for segmentation_class in range(1, max(2, num_classes)):
                pred_mask_class = pred_mask.copy()
                pred_mask_class[pred_mask_class != segmentation_class] = 0
                contours, _ = cv2.findContours(pred_mask_class, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                image_contours = cv2.drawContours(image_contours, contours, -1, class_colors.get(segmentation_class, (255, 255, 255)), 2)


def train_handler(args):
    device = get_device()
    print(f"Using {device} device")

    training_info_dict = {
        "task": args.task,
        "model_name": args.model,
        "hyperparameters": {
            "learning_rate": args.lr,
            "batch_size": args.batchsize,
            "num_epochs": args.epochs,
            "image_size": [args.height, args.width],
            "num_classes": args.num_classes,
            "train_fraction": args.train_fraction,
            "lr_decay_epochs": args.lr_decay_epochs,
            "lr_decay_factor": args.lr_decay_factor,
            "loss_function": args.loss,
        },
        "train_info": {},
        "post_training_info": {}
    }
    traindir = os.path.normpath(args.traindir)
    if not os.path.isdir(traindir):
        print("Train dir %s cannot be opened." % traindir)
        return
    output_dir = os.path.normpath(args.savedir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_transform = get_transforms(args.height, args.width)
    val_transform = get_transforms(args.height, args.width, is_train=False)

    train_loader, val_loader, train_info = get_loaders(traindir, args.batchsize, train_transform, val_transform,
                                                       args.train_fraction, args.num_classes)
    training_info_dict["train_info"] = train_info
    if args.model == "Unet with efficientnet-b3":
        model = smp.Unet(encoder_name="efficientnet-b3", in_channels=3, classes=args.num_classes, activation=None)
    elif args.model == "BuildingDetector":
        model = FieldDetector(args.num_classes)

    model = model.to(device)

    writer = create_tensorboard(train_loader, model, device, args.savedir)
    if args.loss == "dice":
        loss_fn = dice_loss
    elif args.loss == "dice_bce":
        loss_fn = dice_bce_loss
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_decay_epochs,
        gamma=args.lr_decay_factor)

    train(train_loader, val_loader, model, optimizer, loss_fn, lr_scheduler, writer, args.num_classes, args.epochs,
          output_dir, device, args.batchsize, args.height, args.width, args.task, training_info_dict)


def test_handler(args):
    device = get_device()
    print(f"Using {device} device")

    testdir = os.path.normpath(args.testdir)
    if not os.path.isdir(testdir):
        print("Test dir %s cannot be opened." % testdir)
        return

    modelzip = os.path.normpath(args.modelzip)
    if not os.path.isfile(modelzip):
        print("Model zip %s cannot be opened." % modelzip)
        return

    model, training_info_dict = load_model_from_zip(modelzip, device)

    height, width = training_info_dict["hyperparameters"]["image_size"]

    num_classes = training_info_dict["hyperparameters"]["num_classes"]
    batch_size = args.batchsize

    test_transform = get_transforms(height, width, is_train=False)
    test_ds = DataSetSegmentation(testdir, 0, is_train=False, transform=test_transform, num_classes=num_classes)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    model.eval()

    accuracy, precision, recall, f1, iou, dice = test_segmentation_model(model, test_loader, device, args.threshold)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"IOU: {iou:.4f}")
    print(f"Dice: {dice:.4f}")


def visualize_handler(args):
    device = get_device()
    print(f"Using {device} device")

    imagesdir = os.path.normpath(args.imagesdir)
    if not os.path.isdir(imagesdir):
        print("Image dir %s cannot be opened." % imagesdir)
        return

    output_dir = os.path.normpath(args.outputdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    modelzip = os.path.normpath(args.modelzip)
    if not os.path.isfile(modelzip):
        print("Model zip %s cannot be opened." % modelzip)
        return

    model, training_info_dict = load_model_from_zip(modelzip, device)

    hyperparameters = training_info_dict["hyperparameters"]
    num_classes = hyperparameters["num_classes"]

    image_height, image_width = hyperparameters["image_size"]

    visualize(imagesdir, model, output_dir, num_classes, image_height, image_width, args.threshold, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--task", help="Task name.", default="building_seg",
                              choices=["building_seg", "rural_vs_residential_seg"])
    parser_train.add_argument("--model", help="Which model to train.", default="Unet with efficientnet-b3",
                              choices=["BuildingDetector", "Unet with efficientnet-b3"])
    parser_train.add_argument("--traindir", help="Path to training images and masks.", required=True)
    parser_train.add_argument("--savedir", help="Directory where to save checkpoints.", required=True)
    parser_train.add_argument("--loss", help="Loss Function.", default="bce",
                              choices=["dice", "dice_bce", "bce"])
    parser_train.add_argument("--train_fraction", help="Fraction of images for validation.", type=float, default=0.9)
    parser_train.add_argument("--height", help="Tile height.", type=int, required=True)
    parser_train.add_argument("--width", help="Tile width.", type=int, required=True)
    parser_train.add_argument("--lr", help="Learning rate at start.", type=float, default=0.001)
    parser_train.add_argument("--lr_decay_epochs", help="Decay learning rate after this many epochs.", type=int,
                              default=10)
    parser_train.add_argument("--lr_decay_factor", help="Decay learning rate by this factor.", type=float, default=0.1)
    parser_train.add_argument("--batchsize", help="Batch size.", type=int, default=64)
    parser_train.add_argument("--epochs", help="Number of epochs to train for.", type=int)
    parser_train.add_argument("--num_classes", help="Number of segmentation classes.", type=int)
    parser_train.set_defaults(func=train_handler)

    parser_test = subparsers.add_parser("test")
    parser_test.add_argument("--testdir", help="Path to testing images and masks.", required=True)
    parser_test.add_argument("--modelzip", help="Path to zip file containing checkpoint and json.", required=True)
    parser_test.add_argument("--batchsize", help="Batch size.", type=int, default=8)
    parser_test.add_argument("--threshold", help="Threshold used for binary segmentation", default=0.05, type=float)
    parser_test.set_defaults(func=test_handler)

    parser_visualize = subparsers.add_parser("pred_visualize")
    parser_visualize.add_argument("--imagesdir", help="Paths to images", required=True)
    parser_visualize.add_argument("--modelzip", help="Path to zip file containing checkpoint and json.", required=True)
    parser_visualize.add_argument("--outputdir", required=True)
    parser_visualize.add_argument("--threshold", help="Threshold used for binary segmentation", default=0.05, type=float)

    parser_visualize.set_defaults(func=visualize_handler)

    args = parser.parse_args()
    args.func(args)
