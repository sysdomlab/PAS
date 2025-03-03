import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time, random, math

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

from threading import Lock,Thread

from fedavg_server import FedAvg_AggregationService
from cas_server import CAS_AggregationService, CASAPF_AggregationService
from gift_server import GIFT_AggregationService
from adacomm_server import AdaComm_AggregationService
from fedlama_server import FedLAMA_AggregationService


torch.set_printoptions(precision=4)
CUDA = torch.cuda.is_available()

reporting_fraction = 1

    

def logging(string):
    print('['+str(datetime.datetime.now())+'] [Server] '+str(string))
    sys.stdout.flush()


if __name__ == "__main__":

    ''' Parse arguments and create profile. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=20000)
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--trial_no', type=int, default=0)
    parser.add_argument('--remarks', type=str, default='Remarks Missing...')
    parser.add_argument('--interlayer_type', type=str, default='Defualt')
    parser.add_argument('--init_period', type=int, default=128)
    parser.add_argument('--test_process_port', type=int, default=30000)

    args = parser.parse_args()
    print('Trial ID: ' + str(args.trial_no) + '; Exp Setup Remarks: ' + args.remarks + '\n')
    sys.stdout.flush()

    random.seed(0)

    if args.interlayer_type == 'CAS':
        service = CAS_AggregationService(args.world_size, args.init_period, args.test_process_port, reporting_fraction)
    elif args.interlayer_type == 'CAS-APF':
        service = CASAPF_AggregationService(args.world_size, args.init_period, args.test_process_port, reporting_fraction)
    elif args.interlayer_type == 'GIFT':
        service = GIFT_AggregationService(args.world_size, args.init_period, args.test_process_port, reporting_fraction)
    elif args.interlayer_type == 'AdaComm':
        service = AdaComm_AggregationService(args.world_size, args.init_period, args.test_process_port, reporting_fraction)
    elif args.interlayer_type == 'FedLAMA':
        service = FedLAMA_AggregationService(args.world_size, args.init_period, args.test_process_port, reporting_fraction)
    else:
        service = FedAvg_AggregationService(args.world_size, args.init_period, args.test_process_port, reporting_fraction)
    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config['allow_pickle'] = True
    server = ThreadedServer(service, port=args.port, protocol_config=rpyc_config)
    server.start()