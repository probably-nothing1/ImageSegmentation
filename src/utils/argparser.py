import argparse

def parse_args(cmd=None):
    parser = argparse.ArgumentParser(description='Parameters for train/test script')
    parser.add_argument('--data-dir', help='Path to data folders', required=True)
    parser.add_argument('--use-cpu', action='store_true')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='training epochs')
    parser.add_argument('--train-batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--test-batch-size', type=int, default=32, help='Testing batch size')
    parser.add_argument('--loss-function', choices=['Dice', 'IoU', 'PixelwiseCrossEntropy'], default='Dice')
    parser.add_argument('--optimizer', choices=['SGD', 'Adam', 'AdamW', 'RMSprop', 'Adagrad'], default='Adam')
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--augmentations', nargs='*', default=[], choices=('flip', 'rot90', 'rot270'))
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--lr-scheduler', default=None, choices=[None, 'StepLR', 'MultiStepLR', 'CosineAnnealingLR', 'CyclicLR', 'OneCycleLR', 'CosineAnnealingWarmRestarts'])
    parser.add_argument('--step-lr-step-size', type=int, default=1)
    parser.add_argument('--step-lr-gamma', type=float, default=0.9)
    parser.add_argument('--multistep-lr-milestones', nargs='+', type=int, default=[5, 10])
    parser.add_argument('--multistep-lr-gamma', type=float, default=0.3)
    return parser.parse_args() if cmd is None else parser.parse_args(cmd)