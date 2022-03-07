import torch

class Timer:
    def __init__(self, device='cuda'):
        self.device = device
        if self.device == 'cuda':
            self._start = torch.cuda.Event(enable_timing=True)
            self._end   = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self.device == 'cuda':
            self._start.record()
        else:
            self._start_time = time.time()

    def stop(self):
        if self.device == 'cuda':
            self._end.record()
            torch.cuda.synchronize()
            self.elapsed_ms_ = self._start.elapsed_time(self._end)
        else:
            self.elapsed_ms_ = (time.time() - self._start_time) * 1.e3

    def elapsed_ms(self):
        return self.elapsed_ms_

    def elapsed_seconds(self):
        return self.elapsed_ms_ * 1.e-3

class MeasureMemory:
    def __init__(self, device = 'cuda'):
        self._reserved = 0.
        self._allocated = 0.
        self._device = device

    def measure(self):
        if 'cuda' in self._device:
            self._reserved = torch.cuda.memory_reserved(device=self._device) / 1.e9
            self._allocated = torch.cuda.memory_allocated(device=self._device) / 1.e9

    @property
    def reserved(self):
        return self._reserved

    @property
    def allocated(self):
        return self._allocated
