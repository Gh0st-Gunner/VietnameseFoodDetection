"""
Training script for VNFood Dataset (103 Vietnamese Food Classes)
Dataset: https://www.kaggle.com/datasets/meowluvmatcha/vnfood-30-100
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import time
from pathlib import Path
import json
import argparse


class VNFoodTrainer:
        def save_train_overview(self, history, y_true, y_pred, model_name):
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            from sklearn.metrics import confusion_matrix, classification_report
            import numpy as np
            overview_dir = 'Train_overview'
            os.makedirs(overview_dir, exist_ok=True)

            # Save loss/accuracy curves
            if 'train_loss' in history:
                plt.figure()
                plt.plot(history['train_loss'], label='Train Loss')
                plt.plot(history['val_loss'], label='Val Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'{model_name} Loss Curve')
                plt.legend()
                plt.savefig(os.path.join(overview_dir, f'{model_name}_loss_curve.png'))
                plt.close()
            if 'train_acc' in history:
                plt.figure()
                plt.plot(history['train_acc'], label='Train Acc')
                plt.plot(history['val_acc'], label='Val Acc')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title(f'{model_name} Accuracy Curve')
                plt.legend()
                plt.savefig(os.path.join(overview_dir, f'{model_name}_acc_curve.png'))
                plt.close()

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
            plt.title(f'{model_name} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(overview_dir, f'{model_name}_confusion_matrix.png'))
            plt.close()

            # Classification report
            report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(os.path.join(overview_dir, f'{model_name}_classification_report.csv'))
            # Pie chart for support
            plt.figure(figsize=(8, 8))
            report_df = report_df.iloc[:-3]  # Remove avg/total rows
            plt.pie(report_df['support'], labels=report_df.index, autopct='%1.1f%%')
            plt.title(f'{model_name} Class Support Pie Chart')
            plt.savefig(os.path.join(overview_dir, f'{model_name}_support_pie.png'))
            plt.close()

            # Bar chart for F1-score
            plt.figure(figsize=(12, 6))
            plt.bar(report_df.index, report_df['f1-score'])
            plt.xticks(rotation=90)
            plt.title(f'{model_name} F1-score by Class')
            plt.ylabel('F1-score')
            plt.tight_layout()
            plt.savefig(os.path.join(overview_dir, f'{model_name}_f1score_bar.png'))
            plt.close()

            # Save text log
            with open(os.path.join(overview_dir, f'{model_name}_log.txt'), 'w', encoding='utf-8') as f:
                f.write('Classification Report:\n')
                f.write(classification_report(y_true, y_pred, target_names=self.class_names))
                f.write('\n\nConfusion Matrix:\n')
                f.write(np.array2string(cm))
                f.write('\n\nHistory (metrics per epoch):\n')
                for k, v in history.items():
                    f.write(f'{k}: {v}\n')
        def __init__(self, config):
            self.config = config
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f'Using device: {self.device}')

            # Setup data transforms
            self.train_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

            self.val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

            # Load datasets
            self.setup_datasets()

            # Setup model
            self.setup_model()

            # Setup training
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=config['learning_rate'],
                                    weight_decay=config['weight_decay'])

            # Learning rate scheduler
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=3
            )

            # Best accuracy tracking
            self.best_acc = 0.0
            self.start_epoch = 1
            self.best_checkpoints = []  # List of (epoch, acc, path) tuples for top-K models

            # Early stopping
            self.early_stopping_patience = config.get('early_stopping_patience', 7)
            self.early_stopping_counter = 0
            self.early_stopping_best = None
            
        def setup_datasets(self):
            """Load train, validation, and test datasets"""
            data_dir = self.config['data_dir']
            
            train_dir = os.path.join(data_dir, 'train')
            val_dir = os.path.join(data_dir, 'val')
            test_dir = os.path.join(data_dir, 'test')
            
            self.train_dataset = datasets.ImageFolder(train_dir, transform=self.train_transform)
            self.val_dataset = datasets.ImageFolder(val_dir, transform=self.val_transform)
            self.test_dataset = datasets.ImageFolder(test_dir, transform=self.val_transform)
            
            self.train_loader = DataLoader(
                self.train_dataset, 
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers'],
                pin_memory=True
            )
            
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=True
            )
            
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=True
            )
            
            self.num_classes = len(self.train_dataset.classes)
            self.class_names = self.train_dataset.classes
            
            print(f'\nDataset Information:')
            print(f'Number of classes: {self.num_classes}')
            print(f'Training samples: {len(self.train_dataset)}')
            print(f'Validation samples: {len(self.val_dataset)}')
            print(f'Test samples: {len(self.test_dataset)}')
            print(f'Class names: {self.class_names[:5]}... (showing first 5)')
            
        def setup_model(self):
            """Initialize the model"""
            model_name = self.config['model_name']
            
            if model_name == 'resnet50':
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, self.num_classes)
                
            elif model_name == 'resnet101':
                self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, self.num_classes)
                
            elif model_name == 'efficientnet_b0':
                self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
                num_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(num_features, self.num_classes)
                
            elif model_name == 'efficientnet_b3':
                self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
                num_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(num_features, self.num_classes)

            elif model_name == 'efficientnet_b7':
                self.model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
                num_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(num_features, self.num_classes)

            elif model_name == 'mobilenet_v3_large':
                self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
                num_features = self.model.classifier[3].in_features
                self.model.classifier[3] = nn.Linear(num_features, self.num_classes)

            else:
                raise ValueError(f'Model {model_name} not supported')
            
            self.model = self.model.to(self.device)
            print(f'\nModel: {model_name}')
            print(f'Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}')
            
        def train_epoch(self, epoch):
            """Train for one epoch"""
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            start_time = time.time()
            
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if (batch_idx + 1) % self.config['print_freq'] == 0:
                    print(f'Epoch: {epoch} [{batch_idx + 1}/{len(self.train_loader)}] '
                        f'Loss: {running_loss / (batch_idx + 1):.4f} '
                        f'Acc: {100. * correct / total:.2f}%')
            
            epoch_time = time.time() - start_time
            train_loss = running_loss / len(self.train_loader)
            train_acc = 100. * correct / total
            
            return train_loss, train_acc, epoch_time
        
        def validate(self):
            """Validate the model"""
            self.model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            val_loss = running_loss / len(self.val_loader)
            val_acc = 100. * correct / total
            
            return val_loss, val_acc
        
        def test(self, history=None, model_name=None):
            """Test the model and save overview"""
            import numpy as np
            self.model.eval()
            correct = 0
            total = 0
            y_true = []
            y_pred = []
            class_correct = [0] * self.num_classes
            class_total = [0] * self.num_classes
            with torch.no_grad():
                for inputs, targets in self.test_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    y_true.extend(targets.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
                    # Per-class accuracy
                    c = predicted.eq(targets)
                    for i in range(len(targets)):
                        label = targets[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1
            test_acc = 100. * correct / total
            print(f'\n{"="*60}')
            print(f'Test Accuracy: {test_acc:.2f}%')
            print(f'{"="*60}')
            # Print per-class accuracy
            print('\nPer-class Accuracy:')
            for i in range(min(10, self.num_classes)):
                if class_total[i] > 0:
                    acc = 100. * class_correct[i] / class_total[i]
                    print(f'{self.class_names[i]:20s}: {acc:.2f}%')
            # Save overview
            if history is not None and model_name is not None:
                self.save_train_overview(history, y_true, y_pred, model_name)
            return test_acc
        
        def save_checkpoint(self, epoch, val_acc, is_best=False):
            """Save model checkpoint"""
            checkpoint_dir = self.config['checkpoint_dir']
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_acc': val_acc,
                'best_acc': self.best_acc,
                'config': self.config,
                'class_names': self.class_names,
                'best_checkpoints': self.best_checkpoints
            }
            
            # Save last checkpoint
            last_path = os.path.join(checkpoint_dir, 'last_checkpoint.pth')
            torch.save(checkpoint, last_path)
            
            # Save periodic checkpoint
            if self.config['save_freq'] > 0 and epoch % self.config['save_freq'] == 0:
                periodic_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save(checkpoint, periodic_path)
                print(f'Saved periodic checkpoint at epoch {epoch}')
            
            # Save best checkpoint
            if is_best:
                best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
                torch.save(checkpoint, best_path)
                
                # Also save to vnfood.pth in the root directory
                vnfood_path = 'vnfood.pth'
                torch.save(checkpoint, vnfood_path)
                print(f'Saved best model with accuracy: {val_acc:.2f}%')
            
            # Manage top-K best checkpoints
            keep_best_k = self.config.get('keep_best_k', 0)
            if keep_best_k > 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'best_epoch_{epoch}_acc_{val_acc:.2f}.pth')
                
                # Add current checkpoint to the list
                self.best_checkpoints.append((epoch, val_acc, checkpoint_path))
                
                # Sort by accuracy (descending)
                self.best_checkpoints.sort(key=lambda x: x[1], reverse=True)
                
                # Keep only top-K
                if len(self.best_checkpoints) <= keep_best_k:
                    # Save this checkpoint
                    torch.save(checkpoint, checkpoint_path)
                    print(f'Saved top-{len(self.best_checkpoints)} model (epoch {epoch}, acc: {val_acc:.2f}%)')
                else:
                    # Remove the worst checkpoint file and entry
                    _, _, path_to_remove = self.best_checkpoints.pop()
                    if os.path.exists(path_to_remove):
                        os.remove(path_to_remove)
                    
                    # Save current checkpoint if it's in top-K
                    torch.save(checkpoint, checkpoint_path)
                    print(f'Saved top-{keep_best_k} model (epoch {epoch}, acc: {val_acc:.2f}%)')
        
        def load_checkpoint(self, checkpoint_path):
            """Load checkpoint and resume training"""
            print(f'Loading checkpoint from {checkpoint_path}...')
            checkpoint = torch.load(checkpoint_path)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_acc = checkpoint.get('best_acc', 0.0)
            
            if 'best_checkpoints' in checkpoint:
                self.best_checkpoints = checkpoint['best_checkpoints']
            
            print(f'Resumed from epoch {checkpoint["epoch"]} with best accuracy: {self.best_acc:.2f}%')
            print(f'Continuing from epoch {self.start_epoch}')
        
        def train(self):
            """Main training loop with early stopping and overview saving"""
            print(f'\n{"="*60}')
            print(f'Starting Training')
            print(f'{"="*60}\n')

            history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
            self.total_training_time = 0.0

            for epoch in range(self.start_epoch, self.config['epochs'] + 1):
                print(f'\nEpoch {epoch}/{self.config["epochs"]}')
                print(f'{"-"*60}')

                # Train
                train_loss, train_acc, epoch_time = self.train_epoch(epoch)
                self.total_training_time += epoch_time
                # Validate
                val_loss, val_acc = self.validate()
                # Update learning rate
                self.scheduler.step(val_acc)
                # Log
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                # Print epoch summary
                print(f'\nEpoch Summary:')
                print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
                print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
                print(f'Time: {epoch_time:.2f}s')
                print(f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
                # Save checkpoint
                is_best = val_acc > self.best_acc
                if is_best:
                    self.best_acc = val_acc
                    self.early_stopping_counter = 0
                    self.early_stopping_best = val_acc
                else:
                    self.early_stopping_counter += 1
                    print(f"Early stopping counter: {self.early_stopping_counter} / {self.early_stopping_patience}")
                self.save_checkpoint(epoch, val_acc, is_best)
                # Early stopping check
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {self.early_stopping_patience} epochs without improvement.")
                    break
            print(f'\n{"="*60}')
            print(f'Training Complete!')
            print(f'Best Validation Accuracy: {self.best_acc:.2f}%')
            print(f'{"="*60}\n')
            # Load best model and test
            print('Loading best model for testing...')
            best_checkpoint = torch.load(os.path.join(self.config['checkpoint_dir'], 'best_checkpoint.pth'))
            self.model.load_state_dict(best_checkpoint['model_state_dict'])
            # Test and save overview
            self.test_acc = self.test(history=history, model_name=self.config['model_name'])


def compare_models(results):
    """Compare multiple trained models and save comparison results"""
    import pandas as pd
    
    print(f'\n{"="*80}')
    print('MODEL COMPARISON RESULTS')
    print(f'{"="*80}')
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Best Val Accuracy': f"{result['best_val_acc']:.2f}%",
            'Final Test Accuracy': f"{result.get('final_test_acc', 'N/A'):.2f}%" if result.get('final_test_acc') else 'N/A',
            'Training Time (s)': f"{result.get('training_time', 0):.1f}" if result.get('training_time') else 'N/A'
        })
    
    df = pd.DataFrame(comparison_data)
    print('\nModel Performance Comparison:')
    print(df.to_string(index=False))
    
    # Save comparison to file
    os.makedirs('model_comparison', exist_ok=True)
    comparison_file = os.path.join('model_comparison', 'model_comparison_results.txt')
    with open(comparison_file, 'w') as f:
        f.write('MODEL COMPARISON RESULTS\n')
        f.write('='*80 + '\n\n')
        f.write(df.to_string(index=False))
        f.write('\n\nDetailed Results:\n')
        for model_name, result in results.items():
            f.write(f'\n{model_name.upper()}:\n')
            f.write(f'  Best Validation Accuracy: {result["best_val_acc"]:.2f}%\n')
            if result.get('final_test_acc'):
                f.write(f'  Final Test Accuracy: {result["final_test_acc"]:.2f}%\n')
            if result.get('training_time'):
                f.write(f'  Total Training Time: {result["training_time"]:.1f} seconds\n')
    
    # Create bar chart comparison
    try:
        import matplotlib.pyplot as plt
        
        models = list(results.keys())
        val_accs = [results[m]['best_val_acc'] for m in models]
        test_accs = [results[m].get('final_test_acc', 0) for m in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Validation accuracy
        bars1 = ax1.bar(models, val_accs, color='skyblue')
        ax1.set_title('Best Validation Accuracy Comparison')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_xticklabels(models, rotation=45, ha='right')
        for bar, acc in zip(bars1, val_accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # Test accuracy (if available)
        if any(test_accs):
            bars2 = ax2.bar(models, test_accs, color='lightgreen')
            ax2.set_title('Final Test Accuracy Comparison')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_xticklabels(models, rotation=45, ha='right')
            for bar, acc in zip(bars2, test_accs):
                if acc > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                            f'{acc:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join('model_comparison', 'model_comparison_chart.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'\nComparison chart saved to: model_comparison/model_comparison_chart.png')
        
    except ImportError:
        print('\nMatplotlib not available for chart generation')
    
    print(f'\nDetailed comparison saved to: {comparison_file}')
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['best_val_acc'])
    print(f'\n🏆 BEST MODEL: {best_model[0].upper()} (Val Acc: {best_model[1]["best_val_acc"]:.2f}%)')


def main():
    parser = argparse.ArgumentParser(description='Train VNFood Classification Model')
    parser.add_argument('--data_dir', type=str, default='vnfood_combined_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--models', nargs='+', default=['efficientnet_b0'],
                   choices=['resnet50', 'resnet101', 'efficientnet_b0', 
                       'efficientnet_b3', 'efficientnet_b7', 'mobilenet_v3_large'],
                   help='Model architectures to train (space-separated)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--print_freq', type=int, default=50,
                       help='Print frequency')
    
    args = parser.parse_args()
    
    # Train multiple models and compare
    results = {}
    for model_name in args.models:
        print(f'\n{"="*80}')
        print(f'TRAINING MODEL: {model_name.upper()}')
        print(f'{"="*80}')
        
        config = {
            'data_dir': args.data_dir,
            'model_name': model_name,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'num_workers': args.num_workers,
            'checkpoint_dir': f'checkpoints_{model_name}',  # Separate checkpoint dir per model
            'print_freq': args.print_freq,
            'save_freq': 10,  # Save checkpoint every 10 epochs
            'keep_best_k': 3,  # Keep top 3 best models
            'early_stopping_patience': 7  # Stop if no val_acc improvement for 7 epochs
        }
        
        # Print configuration
        print('\nConfiguration:')
        print(json.dumps(config, indent=2))
        
        # Create trainer and start training
        trainer = VNFoodTrainer(config)
        
        # Auto-resume from last checkpoint if it exists
        last_checkpoint = os.path.join(config['checkpoint_dir'], 'last_checkpoint.pth')
        if os.path.exists(last_checkpoint):
            print(f'\nFound existing checkpoint for {model_name}, resuming training...')
            trainer.load_checkpoint(last_checkpoint)
        else:
            print(f'\nNo checkpoint found for {model_name}, starting from scratch...')
        
        trainer.train()
        
        # Store results
        results[model_name] = {
            'best_val_acc': trainer.best_acc,
            'training_time': trainer.total_training_time,
            'final_test_acc': trainer.test_acc if hasattr(trainer, 'test_acc') else None
        }
    
    # Compare models
    if len(results) > 1:
        compare_models(results)


if __name__ == '__main__':
    main()
