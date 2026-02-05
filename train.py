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
import optuna


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
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
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
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing
            self.optimizer = optim.AdamW(self.model.parameters(),
                                       lr=config['learning_rate'],
                                       weight_decay=config['weight_decay'],
                                       betas=(0.9, 0.999))

            # Learning rate scheduler
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=3
            )

            # Best accuracy tracking
            self.best_acc = 0.0
            self.best_loss = float('inf')
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
            from sklearn.metrics import f1_score, precision_score, recall_score
            import numpy as np
            
            self.model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            y_true = []
            y_pred = []
            
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    y_true.extend(targets.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
            
            val_loss = running_loss / len(self.val_loader)
            val_acc = 100. * correct / total
            
            # Calculate additional metrics
            try:
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            except:
                f1, precision, recall = 0.0, 0.0, 0.0
            
            return val_loss, val_acc, f1, precision, recall
        
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
        
        def save_checkpoint(self, epoch, val_acc, is_best=False, f1=0.0, precision=0.0, recall=0.0):
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
                'best_loss': self.best_loss,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
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
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            
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
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
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


def compare_models(results, epochs):
    """Compare multiple trained models and save comparison results"""
    import pandas as pd
    
    print(f'\n{"="*80}')
    print('MODEL COMPARISON RESULTS')
    print(f'{"="*80}')
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, result in results.items():
        avg_time_per_epoch = result.get('training_time', 0) / epochs if result.get('training_time') else 0
        comparison_data.append({
            'Model': model_name,
            'Best Val Accuracy': f"{result['best_val_acc']:.2f}%",
            'Final Test Accuracy': f"{result.get('final_test_acc', 'N/A'):.2f}%" if result.get('final_test_acc') else 'N/A',
            'Training Time (s)': f"{result.get('training_time', 0):.1f}" if result.get('training_time') else 'N/A',
            'Avg Time/Epoch (s)': f"{avg_time_per_epoch:.1f}" if avg_time_per_epoch > 0 else 'N/A'
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
                f.write(f'  Average Time per Epoch: {result["training_time"]/epochs:.1f} seconds\n')
    
    # Create comprehensive model comparison charts
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        models = list(results.keys())
        val_accs = [results[m]['best_val_acc'] for m in models]
        test_accs = [results[m].get('final_test_acc', 0) for m in models]
        training_times = [results[m].get('training_time', 0) for m in models]
        
        # Create main comparison chart (4 panels)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Panel 1: Validation Accuracy Comparison
        bars1 = ax1.bar(models, val_accs, color='skyblue', alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('Best Validation Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_xlabel('Models', fontsize=12)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, max(val_accs) * 1.1)  # Set y-axis to show full range
        
        # Add value labels on bars
        for bar, acc in zip(bars1, val_accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Panel 2: Test Accuracy Comparison (if available)
        if any(test_accs):
            bars2 = ax2.bar(models, test_accs, color='lightgreen', alpha=0.8, edgecolor='black', linewidth=1)
            ax2.set_title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Accuracy (%)', fontsize=12)
            ax2.set_xlabel('Models', fontsize=12)
            ax2.set_xticklabels(models, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_ylim(0, max(test_accs) * 1.1)
            
            for bar, acc in zip(bars2, test_accs):
                if acc > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                            f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No test data available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Final Test Accuracy (N/A)', fontsize=14, fontweight='bold')
        
        # Panel 3: Training Time Comparison
        bars3 = ax3.bar(models, training_times, color='orange', alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_title('Total Training Time Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Time (seconds)', fontsize=12)
        ax3.set_xlabel('Models', fontsize=12)
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, time_val in zip(bars3, training_times):
            if time_val > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_times) * 0.02, 
                        f'{time_val:.0f}s', ha='center', va='bottom', fontweight='bold')
        
        # Panel 4: Accuracy vs Training Time Scatter Plot
        scatter = ax4.scatter(training_times, val_accs, color='red', s=150, alpha=0.7, edgecolors='black', linewidth=1)
        ax4.set_title('Accuracy vs Training Time Trade-off', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Training Time (seconds)', fontsize=12)
        ax4.set_ylabel('Validation Accuracy (%)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Add model labels to scatter points with better positioning
        for i, model in enumerate(models):
            ax4.annotate(model.upper(), (training_times[i], val_accs[i]), 
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                        fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join('model_comparison', 'model_comparison_chart.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create separate accuracy-focused chart
        fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison with ranking
        sorted_indices = np.argsort(val_accs)[::-1]  # Sort by validation accuracy descending
        sorted_models = [models[i] for i in sorted_indices]
        sorted_val_accs = [val_accs[i] for i in sorted_indices]
        sorted_test_accs = [test_accs[i] for i in sorted_indices]
        
        # Ranked validation accuracy
        colors = ['gold', 'silver', '#CD7F32'] + ['skyblue'] * (len(models) - 3)  # Gold, silver, bronze, then blue
        bars5 = ax5.bar(range(len(sorted_models)), sorted_val_accs, color=colors[:len(sorted_models)], 
                       alpha=0.8, edgecolor='black', linewidth=1)
        ax5.set_title('Models Ranked by Validation Accuracy', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Accuracy (%)', fontsize=12)
        ax5.set_xticks(range(len(sorted_models)))
        ax5.set_xticklabels([f'{i+1}. {model}' for i, model in enumerate(sorted_models)], rotation=45, ha='right')
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_ylim(0, max(sorted_val_accs) * 1.1)
        
        for i, (bar, acc) in enumerate(zip(bars5, sorted_val_accs)):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Accuracy difference (Val - Test) if test data available
        if any(test_accs):
            acc_diff = [val - test for val, test in zip(sorted_val_accs, sorted_test_accs)]
            colors_diff = ['green' if diff >= 0 else 'red' for diff in acc_diff]
            bars6 = ax6.bar(range(len(sorted_models)), acc_diff, color=colors_diff, 
                           alpha=0.8, edgecolor='black', linewidth=1)
            ax6.set_title('Accuracy Difference (Val - Test)', fontsize=14, fontweight='bold')
            ax6.set_ylabel('Accuracy Difference (%)', fontsize=12)
            ax6.set_xticks(range(len(sorted_models)))
            ax6.set_xticklabels(sorted_models, rotation=45, ha='right')
            ax6.grid(True, alpha=0.3, axis='y')
            ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            for i, (bar, diff) in enumerate(zip(bars6, acc_diff)):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if diff >= 0 else -0.01), 
                        f'{diff:+.1f}%', ha='center', va=('bottom' if diff >= 0 else 'top'), 
                        fontweight='bold', color='white')
        else:
            ax6.text(0.5, 0.5, 'Test accuracy not available\nfor difference calculation', 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=12)
            ax6.set_title('Accuracy Difference (N/A)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join('model_comparison', 'accuracy_comparison_chart.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'\n📊 Charts saved:')
        print(f'   - Full comparison: model_comparison/model_comparison_chart.png')
        print(f'   - Accuracy focus: model_comparison/accuracy_comparison_chart.png')
        
    except ImportError:
        print('\nMatplotlib not available for chart generation')
    
    print(f'\nDetailed comparison saved to: {comparison_file}')
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['best_val_acc'])
    print(f'\n🏆 BEST MODEL: {best_model[0].upper()} (Val Acc: {best_model[1]["best_val_acc"]:.2f}%)')


def objective(trial, model_name, base_config):
    """Optuna objective function for hyperparameter optimization"""
    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    epochs = trial.suggest_int('epochs', 5, 20)  # Shorter for tuning

    # Update config with suggested values
    config = base_config.copy()
    config.update({
        'learning_rate': lr,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'epochs': epochs,
        'checkpoint_dir': f'temp_checkpoints_{model_name}_{trial.number}',  # Temp dir for tuning
        'save_freq': epochs + 1,  # Don't save during tuning
        'early_stopping_patience': epochs  # Disable early stopping for short runs
    })

    # Create trainer and train
    trainer = VNFoodTrainer(config)
    trainer.train()

    # Return the best validation loss (to minimize)
    return trainer.best_loss


def main():
    parser = argparse.ArgumentParser(description='Train VNFood Classification Model')
    parser.add_argument('--data_dir', type=str, default='vnfood_combined_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--models', nargs='+', default=['efficientnet_b7'],
                   choices=['resnet50', 'resnet101', 'efficientnet_b0', 
                       'efficientnet_b3', 'efficientnet_b7', 'mobilenet_v3_large'],
                   help='Model architectures to train (space-separated)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')                    
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate (default: 0.0001 for transfer learning)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--print_freq', type=int, default=50,
                       help='Print frequency')
    parser.add_argument('--tune', action='store_true',
                       help='Enable hyperparameter tuning with Optuna')
    parser.add_argument('--tune_model', type=str, default=['efficientnet_b7'],
                       choices=['resnet50', 'resnet101', 'efficientnet_b0', 
                               'efficientnet_b3', 'efficientnet_b7', 'mobilenet_v3_large'],
                       help='Model to tune hyperparameters for (used with --tune)')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of Optuna trials (used with --tune)')
    
    args = parser.parse_args()
    
    if args.tune:
        # Hyperparameter tuning mode
        print(f'\n{"="*80}')
        print(f'HYPERPARAMETER TUNING FOR: {args.tune_model.upper()}')
        print(f'{"="*80}')
        
        base_config = {
            'data_dir': args.data_dir,
            'model_name': args.tune_model,
            'num_workers': args.num_workers,
            'print_freq': args.print_freq,
            'save_freq': 1000,  # Don't save during tuning
            'keep_best_k': 1,
            'early_stopping_patience': 1000  # Disable
        }
        
        def objective_wrapper(trial):
            return objective(trial, args.tune_model, base_config)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_wrapper, n_trials=args.n_trials)
        
        print(f'\nBest hyperparameters for {args.tune_model}:')
        print(study.best_params)
        print(f'Best validation loss: {study.best_value:.4f}')
        
        # Save best params
        with open(f'best_params_{args.tune_model}.json', 'w') as f:
            json.dump(study.best_params, f, indent=2)
        print(f'Best parameters saved to: best_params_{args.tune_model}.json')
        
    else:
        # Normal training mode
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
                'save_freq': 5,  # Save checkpoint every 10 epochs
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
            compare_models(results, args.epochs)


if __name__ == '__main__':
    main()
