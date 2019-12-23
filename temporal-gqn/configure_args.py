import math

def configure_argparser(parser):
    parser.add_argument('--dataset', default='action_colorcube_dataset',
                        choices=[
                            'colorshapes',
                        ],
                        help='dataset: {%(choices)s}')

    parser.add_argument('--load-model', default=None, type=str,
                        help='Provide the model path if you want to load one.')
    parser.add_argument('--load-baseline-model', default=None, type=str,
                        help='Provide the baseline model path if you want to load one.')

    parser.add_argument('--recon-loss', default='bernoulli',
                        choices=[
                            'bernoulli',
                            'scalar_gaussian',
                        ],
                        help='reconstruction loss: {%(choices)s}')

    parser.add_argument('--state-dict', action='store_true', default=False,
                        help='Load given model and set its state-dict')

    # net architecture
    parser.add_argument('--model', default='tgqn',
                        choices=['tgqn','tgqn-pd','gqn'],
                        help='model: {%(choices)s}')

    # net params
    parser.add_argument('--nc-enc', type=int, default=32,
                        help='kernel size (number of channels) for encoder')
    parser.add_argument('--nc-lstm', type=int, default=64,
                        help='kernel size (number of channels) for lstm')
    parser.add_argument('--nc-context', type=int, default=256,
                        help='kernel size (number of channels) for representation')
    parser.add_argument('--nz', type=int, default=10,
                        help='size of latent variable')
    parser.add_argument('--num-draw-steps', type=int, default=12,
                        help=' number of steps in Draw')
    parser.add_argument('--context-type', default='same',
                        choices=[
                            'same',
                            'all',
                            'backward',
                        ],
                        help='context type: same | all | backward')
    parser.add_argument('--use-ssm-context', action='store_true', default=False,
                        help='Gives context to RSSM deterministic state.')

    parser.add_argument('--nc-action-emb', type=int, default=32,
                        help='Size of action embedding')

    # type of data
    parser.add_argument('--nheight', type=int, default=64,
                        help='the height / width of the input to network')
    parser.add_argument('--nchannels', type=int, default=3,
                        help='number of channels in input')
    parser.add_argument('--num-actions', type=int, default=0,
                        help='number of actions in the data')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit')
    parser.add_argument('--query-size', type=int, default=2,
                        help='query size')

    # training
    parser.add_argument('--train-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 20)')
    parser.add_argument('--eval-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for test (default: 10)')
    parser.add_argument('--data-size', type=int, default=None,  # 0.25,
                        help='data size')
    parser.add_argument('--optimizer', default='adam',
                        choices=['sgd', 'adam'],
                        help='optimization methods: sgd | adam')

    # std annealing
    parser.add_argument('--std', type=float, default=1.0, help='fixed std')
    parser.add_argument('--beta', type=float, default=1.0, help='fixed beta')

    # log
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--cuda-device', type=int,  nargs='+', default=[0], help='gpu device numbers (starts at 0)')
    parser.add_argument('--log-interval', type=int, default=10, help='report interval')
    parser.add_argument('--vis-interval', type=int, default=5000, help='visualization interval')
    parser.add_argument('--ckpt-interval', type=int, default=50000, help='checkpoint interval')
    parser.add_argument('--cache', default=None, help='path to cache')
    parser.add_argument('--experiment', default=None, help='name of experiment')
    parser.add_argument('--vis-template', default=None, type=str, help='Provide path of template.')


    # model num_states
    parser.add_argument('--sssm-num-state', type=int, default=40,
                        help='Size of the deterministic state in sSSM Temporal CGQN Model')

    parser.add_argument('--shared-core', action='store_true', default=False,
                        help='Enables shared core in DRAW')
    parser.add_argument('--concatenate-latents', action='store_true', default=False,
                        help='Concatenates DRAW latents')

    # data set
    parser.add_argument('--num-timesteps', type=int, default=10,
                        help='Number of timesteps in the data set')
    parser.add_argument('--num-views', type=int, default=10,
                        help='Number of total views per timestep')

    # manual curriculum
    parser.add_argument('--manual-curriculum-bound', type=int, nargs='+', default=[5,5,5,5,5,0,0,0,0,0], help='upper bound on number of contexts')

    parser.add_argument('--q-bernoulli-pick', type=float, nargs='+', default=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        help='Bernoulli probabilities for picking Q for Posterior Dropout')

    parser.add_argument('--allow-empty-context', action='store_true', default=False,
                        help='Enables empty context')

    parser.add_argument('--moving-color-shapes-num-objs', type=int, default=2,
                        help='Number of objects in the color shapes data set.')

    parser.add_argument('--moving-color-shapes-canvas-size', type=int, default=96,
                            help='Size of the canvas in the color shapes data set.')

    # eval
    parser.add_argument('--iwae-k', type=int, default=50,
                        help='K for number of IWAE samples.')
    parser.add_argument('--disable-eval-nll', action='store_true', default=False,
                        help='Disables evaluation of NLL.')
    parser.add_argument('--disable-eval-mse', action='store_true', default=False,
                        help='Disables evaluation of MSE.')