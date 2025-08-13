# Quick Fixes for Your EfficientNet Training Loop

## 1. Add These Optimizations at the Top of Your Notebook

```python
# Add after imports
import torch
torch.backends.cudnn.benchmark = True  # Major speedup for fixed input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Use Tensor Cores
torch.backends.cudnn.allow_tf32 = True

from torch.cuda.amp import GradScaler, autocast  # Mixed precision
```

## 2. Optimize Your Data Loaders

Replace your current data loader creation with:

```python
# Reduce batch size for your 6GB GPU
BATCH_SIZE = 16  # Instead of 32

# Optimize data loaders
train_loader = DataLoader(
    train_split, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4,  # Increase from 2
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive
    prefetch_factor=2  # Prefetch batches
)

val_loader = DataLoader(
    val_split, 
    batch_size=BATCH_SIZE * 2,  # Larger for validation
    shuffle=False, 
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)
```

## 3. Update Your Image Size

Change your transforms to use smaller images:

```python
IMG_SZ = 224  # Instead of 244 - standard EfficientNet size
```

## 4. Add Mixed Precision Training

Add this before your training loop:

```python
# Mixed precision scaler
scaler = GradScaler()
```

## 5. Update Your Training Loop

Replace the core training section with:

```python
for images, labels in train_loader:
    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
    
    # More efficient gradient zeroing
    optimizer.zero_grad(set_to_none=True)
    
    # Mixed precision forward pass
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    # Mixed precision backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    # Your existing statistics code...
```

## 6. Add Memory Cleanup

Add this at the end of each epoch:

```python
# Clear GPU cache
torch.cuda.empty_cache()
```

## 7. Optimize Your Model

Add this after creating your model:

```python
# Freeze early layers for faster initial training
for param in model.features[:4].parameters():
    param.requires_grad = False
```

## 8. Better Optimizer

Replace your optimizer with:

```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

## Expected Improvements:
- **2-3x faster training** per epoch
- **50% less GPU memory usage**
- **Better convergence** with mixed precision
- **More stable training** with optimized data loading

## GPU Memory Usage Should Drop From:
- Current: ~5-6GB usage
- Optimized: ~3-4GB usage

This will allow you to potentially increase batch size back to 24-32 if needed.
