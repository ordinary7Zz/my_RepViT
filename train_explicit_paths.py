from pathlib import Path

from main import get_args_parser, main


def parse_args():
    parser = get_args_parser()
    parser.add_argument('--train-data-path', default='', type=str,
                        help='training dataset path')
    parser.add_argument('--test-data-path', default='', type=str,
                        help='testing dataset path')
    args = parser.parse_args()

    has_train_path = bool(args.train_data_path)
    has_test_path = bool(args.test_data_path)
    if has_train_path != has_test_path:
        parser.error('--train-data-path and --test-data-path must be provided together')

    return args


if __name__ == '__main__':
    args = parse_args()
    if args.output_dir:
        args.output_dir = f'{args.output_dir}/{args.model}_{args.input_size}_{args.weight_decay}_{args.lr}_{args.reprob}'
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
