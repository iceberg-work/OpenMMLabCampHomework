_base_ = ['../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py', '../_base_/default_runtime.py']

model = dict(
    head=dict(
        num_classes=10,
        topk=(1,)
    ))

data = dict(
    # 根据实验环境调整每个 batch_size 和 workers 数量
    samples_per_gpu=32,
    workers_per_gpu=2,
    # 指定训练集路径
    train=dict(
        data_prefix='data/cifar-10-batches-dataset/train',
        ann_file='data/cifar-10-batches-dataset/train.txt',
        classes='data/cifar-10-batches-dataset/classes.txt'
    ),
    # 指定验证集路径
    val=dict(
        data_prefix='data/cifar-10-batches-dataset/val',
        ann_file='data/cifar-10-batches-dataset/val.txt',
        classes='data/cifar-10-batches-dataset/classes.txt'
    ),
)

# 定义评方法
evaluation = dict(metric_options={'topk': (1,)})

# 优化器
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# 学习率策略
lr_config = dict(
    policy='step',
    step=[1])
runner = dict(type='EpochBasedRunner', max_epochs=2)

# 预训练模型
# load_from = '${YOUPATH}/mmclassification/checkpoints/resnet18_batch256_imagenet_20200708-34ab8f90.pth'
load_from = '/HOME/scz5336/run/mmclassification/checkpoints/resnet18_batch256_imagenet_20200708-34ab8f90.pth'