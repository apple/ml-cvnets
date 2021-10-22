#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#


import socket
import torch
import torch.distributed as dist
from utils import logger


def is_master(opts) -> bool:
    node_rank = getattr(opts, "ddp.rank", 0)
    return (node_rank == 0)


def distributed_init(opts) -> int:
    ddp_url = getattr(opts, "ddp.dist_url", None)
    ddp_port = getattr(opts, "ddp.dist_port", 6006)
    is_master_node = is_master(opts)
    if ddp_url is None:
        hostname = socket.gethostname()
        ddp_url = 'tcp://{}:{}'.format(hostname, ddp_port)
        setattr(opts, "ddp.dist_url", ddp_url)

    node_rank = getattr(opts, "ddp.rank", 0)
    world_size = getattr(opts, "ddp.world_size", 0)
    if torch.distributed.is_initialized():
        logger.warning('DDP is already initialized and cannot be initialize twice!')
    else:
        logger.info('distributed init (rank {}): {}'.format(node_rank, ddp_url))

        dist_backend = "gloo"
        if dist.is_nccl_available():
            dist_backend = 'nccl'
            if is_master_node:
                logger.log('Using NCCL as distributed backend with version={}'.format(torch.cuda.nccl.version()))

        dist.init_process_group(
            backend=dist_backend,
            init_method=ddp_url,
            world_size=world_size,
            rank=node_rank
        )

        # perform a dummy all-reduce to initialize the NCCL communicator
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda())

    node_rank = torch.distributed.get_rank()
    setattr(opts, "ddp.rank", node_rank)
    return node_rank