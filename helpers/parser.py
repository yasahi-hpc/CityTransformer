import argparse

def parse():
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument('-data_dir', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='./dataset', \
                        type=str, \
                        choices=None, \
                        help='directory of datasets', \
                        metavar=None
                       )
    parser.add_argument('--device', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='cuda', \
                        type=str, \
                        choices=None, \
                        help='Device to run the model', \
                        metavar=None
                       )
    parser.add_argument('--model_name', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='CityTransformer', \
                        type=str, \
                        choices=None, \
                        help='model name (default: CityTransformer)', \
                        metavar=None
                       )
    parser.add_argument('--batch_size', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=64, \
                        type=int, \
                        choices=None, \
                        help='Batch size', \
                        metavar=None
                       )
    parser.add_argument('--n_epochs', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=3, \
                        type=int, \
                        choices=None, \
                        help='Number of epochs', \
                        metavar=None
                       )
    parser.add_argument('--checkpoint_idx', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=-1, \
                        type=int, \
                        choices=None, \
                        help='Index of checkpoint used for inference', \
                        metavar=None
                       )
    parser.add_argument('--n_digits', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=3, \
                        type=int, \
                        choices=None, \
                        help='Number of digits to predict', \
                        metavar=None
                       )
    parser.add_argument('--n_stations', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=14, \
                        type=int, \
                        choices=None, \
                        help='Number of monitoring stations choose from (5, 10, 14)', \
                        metavar=None
                       )
    parser.add_argument('--nz_scan', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=100, \
                        type=int, \
                        choices=None, \
                        help='Number of vertical points at monitoring stations (25, 50, 100)', \
                        metavar=None
                       )
    parser.add_argument('--n_precision_enhancers', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=0, \
                        type=int, \
                        choices=None, \
                        help='Number of precision_enhancers', \
                        metavar=None
                       )
    parser.add_argument('--version', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=0, \
                        type=int, \
                        choices=None, \
                        help='Model version', \
                        metavar=None
                       )
    parser.add_argument('--lr', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=0.0001, \
                        type=float, \
                        choices=None, \
                        help='Learning rate', \
                        metavar=None
                       )
    parser.add_argument('--momentum', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=0.9, \
                        type=float, \
                        choices=None, \
                        help='Momentum', \
                        metavar=None
                       )
    parser.add_argument('--beta_1', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=0.9, \
                        type=float, \
                        choices=None, \
                        help='beta_1', \
                        metavar=None
                       )
    parser.add_argument('--beta_2', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=0.999, \
                        type=float, \
                        choices=None, \
                        help='beta_1', \
                        metavar=None
                       )
    parser.add_argument('--activation', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='ReLU', \
                        type=str, \
                        choices=None, \
                        help='Type of activation function', \
                        metavar=None
                       )
    parser.add_argument('--loss_type', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='MSE', \
                        type=str, \
                        choices=None, \
                        help='Type of loss function', \
                        metavar=None
                       )
    parser.add_argument('--opt_type', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='Adam', \
                        type=str, \
                        choices=None, \
                        help='Type of optimizer', \
                        metavar=None
                       )
    parser.add_argument('--clip', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=100, \
                        type=float, \
                        choices=None, \
                        help='criteria for clipping', \
                        metavar=None
                       )
    parser.add_argument('--gradient-predivide-factor', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=1.0, \
                        type=float, \
                        choices=None, \
                        help='apply gradient predivide factor in optimizer (default: 1.0)', \
                        metavar=None
                       )
    parser.add_argument('--inference_mode', \
                        action='store_true', \
                        default=False, \
                        help='training or inference'
                       )
    parser.add_argument('--use_adasum', \
                        action='store_true', \
                        default=False, \
                        help='use adasum algorithm to do reduction'
                       )
    parser.add_argument('--fp16_allreduce', \
                        action='store_true', \
                        default=False, \
                        help='use fp16 compression during allreduce'
                       )
    parser.add_argument('--super_precision', \
                        action='store_true', \
                        default=False, \
                        help='Super precision or not'
                       )
    parser.add_argument('--UNet', \
                        action='store_true', \
                        default=False, \
                        help='Use UNet model or not'
                       )

    args = parser.parse_args()
    return args
