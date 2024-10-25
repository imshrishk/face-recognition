import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchsummary import summary

from torch.utils.tensorboard import SummaryWriter

from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

import timm

import onnx
import onnxruntime

from PIL import UnidentifiedImageError

from pathlib import Path

def printg(string): print("\033[92m{}\033[00m".format(string))
def printr(string): print("\033[91m{}\033[00m".format(string))

log = True

if log:
    writer = SummaryWriter()

batch_size = 64
epochs = 500
learning_rate = 1e-3
loss_lr = 1e-4
factor = 0.3
patience = 5
delta = 0
delta_early = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
printg(f"Using device: {device}")

num_classes = 20 # ~100*12
embedding_size = 320

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if log:
    writer.add_scalar('Hyperparameters/Batch_size', batch_size, 0)
    writer.add_scalar('Hyperparameters/Epochs', epochs, 0)
    writer.add_scalar('Hyperparameters/Learning_rate', learning_rate, 0)
    writer.add_scalar('Hyperparameters/Loss_lr', loss_lr, 0)
    writer.add_scalar('Hyperparameters/Num_classes', num_classes, 0)
    writer.add_scalar('Hyperparameters/Embedding_size', embedding_size, 0)
    writer.add_scalar('Hyperparameters/Epochs', factor, 0)
    writer.add_scalar('Hyperparameters/Epochs', patience, 0)
    writer.add_scalar('Hyperparameters/Epochs', delta, 0)
    writer.add_scalar('Hyperparameters/Epochs', delta_early, 0)

# Shouldn't really throw an error, but just in case
class RobustImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except UnidentifiedImageError:
            print(f"\033[91mSkipping Corrupt Image:\033[0m {Path(path)}")            
            # return None, None
            return self.__getitem__(index + 1)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
    
train_dataset = RobustImageFolder('../split/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = RobustImageFolder('../split/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
class ConvNeXtArcFace(nn.Module):
    def __init__(self, model_name, embedding_size, pretrained=True):
        super(ConvNeXtArcFace, self).__init__()
        self.convnext = timm.create_model(model_name, pretrained=pretrained)
        self.convnext.reset_classifier(num_classes=0, global_pool='avg')
        
    def forward(self, x):
        x = self.convnext.forward_features(x)
        embeddings = F.avg_pool2d(x, 7).flatten(1)
        return embeddings
class EarlyStopping:
    def __init__(self, patience=3*patience, verbose=False, delta=delta_early):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, epoch, optimizer, scheduler, criterion, loss_optimizer, loss_scheduler, running_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer, scheduler, criterion, loss_optimizer, loss_scheduler, running_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer, scheduler, criterion, loss_optimizer, loss_scheduler, running_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, optimizer, scheduler, criterion, loss_optimizer, loss_scheduler, running_loss):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_optimizer_state_dict': loss_optimizer.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'loss_scheduler_state_dict': loss_scheduler.state_dict(),
                'loss': running_loss,
                }, f"checkpoints/best_{epoch}.pth")        
        self.val_loss_min = val_loss

model_name = 'convnextv2_atto'
model = ConvNeXtArcFace(model_name, embedding_size)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience, verbose=True, threshold=delta)
criterion = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size, margin=4).to(device)
loss_optimizer = optim.Adam(criterion.parameters(), lr=loss_lr)
loss_scheduler = ReduceLROnPlateau(loss_optimizer, mode='max', factor=factor, patience=patience, verbose=True, threshold=delta)

start_epoch = 1

def load_checkpoint(filepath, model, optimizer, scheduler, loss_optimizer, loss_scheduler, criterion):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    criterion.load_state_dict(checkpoint['criterion_state_dict'])
    loss_optimizer.load_state_dict(checkpoint['loss_optimizer_state_dict'])
    loss_scheduler.load_state_dict(checkpoint['loss_scheduler_state_dict'])
    epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    return model, optimizer, scheduler, loss_optimizer, loss_scheduler, criterion, epoch, loss

checkpoint = None
if checkpoint:
    model, optimizer, scheduler, loss_optimizer, loss_scheduler, criterion, start_epoch, loss = load_checkpoint(
        f"checkpoints/{checkpoint}", model, optimizer, scheduler, loss_optimizer, loss_scheduler, criterion
        )

def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1, device=torch.device('cpu'))

prev_lr = None
prev_loss_lr = None

early_stopping = EarlyStopping()
ckpt = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 90, 110, 130, 150, 175, 200, 240, 270, 300, 340, 380, 420, 460, 500]
for epoch in range(start_epoch, epochs+1):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"Epoch: {epoch}, Batch: {batch_idx + 1}/{len(train_loader)}")
        inputs = inputs.to(device)        
        targets = targets.to(device)
        inputs = inputs.float()

        optimizer.zero_grad()
        loss_optimizer.zero_grad()

        embeddings = model(inputs)
        loss = criterion(embeddings, targets)
        if log:    
            writer.add_scalar('Loss/train', loss.item(), (epoch-1) * len(train_loader) + batch_idx + 1)  
          
        loss.backward()
        optimizer.step()
        loss_optimizer.step()
        
        running_loss += loss.item()

    model.eval()

    train_embeddings, train_labels = get_all_embeddings(train_dataset, model)
    val_embeddings, val_labels = get_all_embeddings(val_dataset, model)

    train_labels = train_labels.squeeze(1)
    val_labels = val_labels.squeeze(1)

    ### TRAINING ACCURACY
    accuracies = accuracy_calculator.get_accuracy(
            train_embeddings, train_labels, train_embeddings, train_labels, False)
    training_accuracy = accuracies['precision_at_1']
    if log:
        writer.add_scalar('Accuracy/Training', training_accuracy, epoch)
    printg(f"Train Set Accuracy = {training_accuracy}")

    ### VALIDATION ACCURACY
    accuracies = accuracy_calculator.get_accuracy(
            val_embeddings, val_labels, train_embeddings, train_labels, False)
    validation_accuracy = accuracies['precision_at_1']
    if log:
        writer.add_scalar('Accuracy/Validation', validation_accuracy, epoch)
    printg(f"Test Set Accuracy = {validation_accuracy}")

    scheduler.step(validation_accuracy)
    loss_scheduler.step(validation_accuracy)

    curr_lr = optimizer.param_groups[0]['lr']
    curr_loss_lr = loss_optimizer.param_groups[0]['lr']

    if prev_lr is not None and curr_lr != prev_lr:
        printr(f"LR {prev_lr} --> {curr_lr}")
        printr(f"Loss LR {prev_loss_lr} --> {curr_loss_lr}")
    prev_lr = curr_lr
    prev_loss_lr = curr_loss_lr

    if log:
        writer.add_scalar('Hyperparameters/Learning_rate', curr_lr, epoch)
        writer.add_scalar('Hyperparameters/Loss_lr', curr_loss_lr, epoch)
 
    printr(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")

    if (epoch) in ckpt:
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss_optimizer_state_dict': loss_optimizer.state_dict(),
                    'loss_scheduler_state_dict': loss_scheduler.state_dict(),
                    'criterion_state_dict': criterion.state_dict(),
                    'loss': running_loss,
                    }, f"checkpoints/epoch_{epoch}.pth")
        if log:
            writer.flush()

    early_stopping(-validation_accuracy, model, epoch, optimizer, scheduler, criterion, loss_optimizer, loss_scheduler, running_loss)
    if early_stopping.early_stop:
        print("Early Stopping")
        break

if log:
    writer.close()

torch.save(model.state_dict(), 'convnext_atto_arcface.pth')