# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
from torch import Tensor
from torch.nn.parallel._functions import _get_stream


def _as_torch_device(device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == -1:
        return torch.device('cpu')
    return torch.device('cuda', device)


def scatter(input: Union[List, Tensor],
            devices: List,
            streams: Optional[List] = None) -> Union[List, Tensor]:
    """Scatters tensor across multiple GPUs."""
    if streams is None:
        streams = [None] * len(devices)

    if isinstance(input, list):
        chunk_size = (len(input) - 1) // len(devices) + 1
        outputs = [
            scatter(input[i], [devices[i // chunk_size]],
                    [streams[i // chunk_size]]) for i in range(len(input))
        ]
        return outputs
    elif isinstance(input, Tensor):
        output = input.contiguous()
        # TODO: copy to a pinned buffer first (if copying from CPU)
        stream = streams[0] if output.numel() > 0 else None
        if devices != [-1]:
            target_device = _as_torch_device(devices[0])
            with torch.cuda.device(target_device), torch.cuda.stream(stream):
                output = output.cuda(target_device, non_blocking=True)

        return output
    else:
        raise Exception(f'Unknown type {type(input)}.')


def synchronize_stream(output: Union[List, Tensor], devices: List,
                       streams: List) -> None:
    if isinstance(output, list):
        chunk_size = len(output) // len(devices)
        for i in range(len(devices)):
            for j in range(chunk_size):
                synchronize_stream(output[i * chunk_size + j], [devices[i]],
                                   [streams[i]])
    elif isinstance(output, Tensor):
        if output.numel() != 0:
            with torch.cuda.device(_as_torch_device(devices[0])):
                main_stream = torch.cuda.current_stream()
                main_stream.wait_stream(streams[0])
                output.record_stream(main_stream)
    else:
        raise Exception(f'Unknown type {type(output)}.')


def get_input_device(input: Union[List, Tensor]) -> int:
    if isinstance(input, list):
        for item in input:
            input_device = get_input_device(item)
            if input_device != -1:
                return input_device
        return -1
    elif isinstance(input, Tensor):
        return input.get_device() if input.is_cuda else -1
    else:
        raise Exception(f'Unknown type {type(input)}.')


class Scatter:

    @staticmethod
    def forward(target_gpus: List[int], input: Union[List, Tensor]) -> tuple:
        input_device = get_input_device(input)
        streams = None
        if input_device == -1 and target_gpus != [-1]:
            # Perform CPU to GPU copies in a background stream
            streams = [_get_stream(_as_torch_device(device))
                       for device in target_gpus]

        outputs = scatter(input, target_gpus, streams)
        # Synchronize with the copy stream
        if streams is not None:
            synchronize_stream(outputs, target_gpus, streams)

        return tuple(outputs) if isinstance(outputs, list) else (outputs, )
