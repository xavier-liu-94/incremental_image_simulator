from torch.utils.tensorboard import SummaryWriter
from torch.utils.hooks import RemovableHandle
import torch
from typing import List

class TrainMonitor():

    def __init__(self) -> None:
        self.infos = {}
        self.handles: List[RemovableHandle] = []
    
    def close(self):
        for h in self.handles:
            h.remove()
        self.writer.close()

    def register_backward(self, module: torch.nn.Module, namespace: str):

        def iter_regist(module: torch.nn.Module, prefix=[]):
            for n, m in module.named_children():
                if isinstance(m, (torch.nn.ModuleList, torch.nn.ModuleDict)):
                    iter_regist(m, prefix+[n])
                else:
                    self.handles.append(
                        m.register_full_backward_hook(self.get_back_hook(".".join(prefix + [n]), namespace))
                    )

        iter_regist(module)

    def get_back_hook(self, name: str, name_space: str=""):
        full_name = name_space + "-" + name
        def hook(module, grad_input, grad_output):
            for idx, t in enumerate(grad_output):
                if t is not None:
                    s, m = torch.std_mean(t)
                    self.infos[full_name+"-grad_out-mean_"+str(idx)] = m.detach().item()
                    self.infos[full_name+"-grad_out-std_"+str(idx)] = s.detach().item()
            for idx, t in enumerate(grad_input):
                if t is not None:
                    s, m = torch.std_mean(t)
                    self.infos[full_name+"-grad_in-mean_"+str(idx)] = m.detach().item()
                    self.infos[full_name+"-grad_in-std_"+str(idx)] = s.detach().item()
            return grad_input

        return hook