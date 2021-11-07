# optimizer
lr = 0.001  # max learning rate
optimizer = dict(type='Adam', lr=lr, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
# runtime settings
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=327,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)

total_epochs = 40