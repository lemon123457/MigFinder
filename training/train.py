"""Train the detection model on a single-class COCO dataset."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from detect import DetectionModel

dataset_dir = '/path/of/dataset'
output_dir = '/path/of/results/training'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=dataset_dir, help='COCO root with train, valid and test folders.')
    parser.add_argument('--output', default=output_dir)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--resolution', type=int, default=1008)
    args = parser.parse_args()
    if args.dataset.startswith('/path/of/') or args.output.startswith('/path/of/'):
        parser.error('Replace dataset_dir/output_dir or provide --dataset and --output with your own paths.')
    for split in ['train', 'valid', 'test']:
        annotation = Path(args.dataset) / split / '_annotations.coco.json'
        if not annotation.is_file():
            parser.error(f'Missing COCO annotation file: {annotation}')
    model = DetectionModel()
    model.train(dataset_dir=args.dataset, epochs=args.epochs, batch_size=args.batch_size,
                grad_accum_steps=1, lr=1e-5, resolution=args.resolution,
                output_dir=args.output, early_stopping=True)


if __name__ == '__main__':
    main()
