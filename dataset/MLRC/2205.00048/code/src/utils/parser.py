from argparse import ArgumentParser

def parser_args():
    parser = ArgumentParser(description="JMEF")
    parser.add_argument('--data', type=str, default='ml-1m', choices=['ml-1m', 'lt'],
                        help="File path for data")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0, help="Seed (For reproducability)")
    parser.add_argument('--model', type=str, default='Pop')
    parser.add_argument('--gamma', type=float, default=0.8, help="patience factor")
    parser.add_argument('--temp', type=float, default=0.1, help="temperature. how soft the ranks to be")
    parser.add_argument('--s_ep', type=int, default=5)
    parser.add_argument('--r_ep', type=int, default=1)
    parser.add_argument('--norm', type=str, default='N')
    parser.add_argument('--coll', type=str, default='Y')
    parser.add_argument('--age', type=str, default='N')
    parser.add_argument('--ndatapoints', type=int, default=1)
    parser.add_argument('--conduct', type=str, default='sh')
    return parser.parse_args()