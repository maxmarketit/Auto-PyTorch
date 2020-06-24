__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import inspect
import torch
import numpy as np # numpy를 여기서 import하면 부하가 커지나???
import warnings

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

from torch.nn.modules.loss import _Loss
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.config.config_option import ConfigOption

class QuantileLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean', qs=(0.5,)):
        super(QuantileLoss, self).__init__(size_average, reduce, reduction)
        #if not(isinstance(qs, tuple)) and not(isinstance(qs, np.ndarray)):
        #    raise ValueError('qs must be either list or np.array')
        #if isinstance(qs, tuple):
        #    qs = np.array(qs)
        #if isinstance(qs, np.ndarray) and not (len(qs.shape) == 1):
        #    raise ValueError('qs must be shape of 1 dim')

        #self.qs = qs

        if not(isinstance(qs, tuple)):
            raise ValueError('qs must be tuple')

        self.qs = torch.tensor(qs).float().reshape(1,-1)

    def to(self, device):
        self.qs.to(device)
        super(QuantileLoss, self).to(device)

    def forward(self, input, target):
        reduction = self.reduction
        if not (target.size() == input.size()):
            warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                          "This will likely lead to incorrect results due to broadcasting. "
                          "Please ensure they have the same size.".format(target.size(), input.size()),
                          stacklevel=2)
            
        qs = torch.tensor(self.qs, dtype = input.dtype, requires_grad=False).to(input.device)
        if True:
            e = target - input           
            ret = torch.max(qs * e, (qs - 1) * e)
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
        else:
            raise ValueError('not(target.requires_grad): not yet implemented')

class QinputLoss(_Loss):
    # import numpy as np
    import warnings

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean', ws=(0,1)):
            # size_average와 reduce는 앞으로 deprecated될 예정, reduction을 사용하라
            # reduction = 'mean' or 'sum' or 'none'
        super(QinputLoss, self).__init__(size_average, reduce, reduction)
        #if not(isinstance(ws, tuple)) and not(isinstance(ws, np.ndarray)):
        #    raise ValueError('ws must be either list or np.array')
        #if isinstance(ws, tuple):
        #    ws = np.array(ws)
        #if isinstance(ws, np.ndarray) and not (len(ws.shape) == 1):
        #    raise ValueError('qs must be shape of 1 dim')

        if not(isinstance(ws, tuple)):
            raise ValueError('ws must be tuple')

        #self.ws = ws
        self.ws = torch.tensor(ws).float().reshape(1,-1)


    def to(self, device):
        self.ws.to(device)
        super(QinputLoss, self).to(device)


    def forward(self, input, target):
        # 위의 l1_loss에서 따옴
        # l1_loss(input, target, size_average=None, reduce=None, reduction='mean')
        reduction = self.reduction
        #if not torch.jit.is_scripting():  # 뭐 하는 곳인지 모름
        #    raise ValueError('QuantileLoss: torch.jit.is_scripting() not implemented')
        #    #tens_ops = (input, target)
        #    #if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
        #    #    return handle_torch_function(
        #    #        l1_loss, tens_ops, input, target, size_average=size_average, reduce=reduce,
        #    #        reduction=reduction)
        if not (target.size() == input.size()):
            warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                          "This will likely lead to incorrect results due to broadcasting. "
                          "Please ensure they have the same size.".format(target.size(), input.size()),
                          stacklevel=2)
            
        #ws = torch.tensor(self.ws, dtype = input.dtype, requires_grad=False).to(input.device)
        # ws : weights on quantile loss and weights on output 
        w_q = self.ws[0]
        w_input = self.ws[1:].reshape(1,-1)
        q = target[:,0:1]
        #if target.requires_grad: # 아마 target이 고정되어 있는 경우 else:에서 빠른 처리가 가능한 듯.
        if True:
            e_q = torch.abs(target[:,0]-input[:,0]) # q error, if w_q is 0, ignored
            e= target[:,1:] - input[:,1:] # target error
            
            # target[:,0:1] is q! 
            ret = w_input*torch.max(q * e, (q - 1) * e)
            #ret = torch.abs(input - target) # MAE 
        if reduction != 'none':
            ret = torch.mean(ret) +torch.mean(w_q* e_q) if reduction == 'mean' else torch.sum(ret) + torch.sum(w_q*e_q)
        else:
            raise ValueError('not(target.requires_grad): not yet implemented')
            #expanded_input, expanded_target = torch.broadcast_tensors(input, target)
            #ret = torch._C._nn.l1_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))
        return ret 



class LossModuleSelector(PipelineNode):
    def __init__(self):
        super(LossModuleSelector, self).__init__()
        self.loss_modules = dict()

    def fit(self, hyperparameter_config, pipeline_config, X, Y, train_indices):
        hyperparameter_config = ConfigWrapper(self.get_name(), hyperparameter_config)

        weights = None
        loss_module = self.loss_modules[hyperparameter_config["loss_module"]]
        if (loss_module.weight_strategy != None):
            weights = loss_module.weight_strategy(pipeline_config, X[train_indices], Y[train_indices])
            weights = torch.from_numpy(weights).float()

        # pass weights to loss module
        loss = loss_module.module
        if "pos_weight" in inspect.getfullargspec(loss)[0] and weights is not None and inspect.isclass(loss):
            loss = loss(pos_weight=weights)
        elif "weight" in inspect.getfullargspec(loss)[0] and weights is not None and inspect.isclass(loss):
            loss = loss(weight=weights)
        elif inspect.isclass(loss):
            loss = loss()
        loss_module.set_loss_function(loss)
        return {'loss_function': loss_module}

    def add_loss_module(self, name, loss_module, weight_strategy=None, requires_target_class_labels=False):
        """Add a loss module, has to be a pytorch loss module type
        
        Arguments:
            name {string} -- name of loss module for definition in config
            loss_module {type} -- a pytorch loss module type
            weight_strategy {function} -- callable that computes label weights
        """

        self.loss_modules[name] = AutoNetLossModule(loss_module, weight_strategy, requires_target_class_labels)

    def remove_loss_module(self, name):
        del self.loss_modules[name]

    def get_hyperparameter_search_space(self, dataset_info=None, **pipeline_config):
        pipeline_config = self.pipeline.get_pipeline_config(**pipeline_config)
        cs = ConfigSpace.ConfigurationSpace()

        possible_loss_modules = set(pipeline_config["loss_modules"]).intersection(self.loss_modules.keys())
        cs.add_hyperparameter(CSH.CategoricalHyperparameter('loss_module', list(possible_loss_modules)))
        self._check_search_space_updates(self.loss_modules.keys(), "*")
        return cs
        

    def get_pipeline_config_options(self):
        loss_module_names = list(self.loss_modules.keys())
        options = [
            ConfigOption(name="loss_modules", default=loss_module_names, type=str, list=True, choices=loss_module_names),
        ]
        return options

class AutoNetLossModule():
    def __init__(self, module, weight_strategy, requires_target_class_labels):
        self.module = module
        self.weight_strategy = weight_strategy
        self.requires_target_class_labels = requires_target_class_labels
        self.function = None

    def set_loss_function(self, function):
        self.function = function

    def __call__(self, x, y):
        if not self.requires_target_class_labels:
            return self.function(x, y)
        elif len(y.shape) == 1:
            return self.function(x, y)
        else:
            return self.function(x, y.max(1)[1])

    def to(self, device):
        result = AutoNetLossModule(self.module, self.weight_strategy, self.requires_target_class_labels)
        result.set_loss_function(self.function.to(device))
        return result
