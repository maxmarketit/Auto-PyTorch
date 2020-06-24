import numpy as np
import torch
import torch.nn as nn
import copy
from autoPyTorch.core.autonet_classes.autonet_feature_data import AutoNetFeatureData

class AutoNetRegression(AutoNetFeatureData):
    preset_folder_name = "feature_regression"

    # OVERRIDE
    @staticmethod
    def _apply_default_pipeline_settings(pipeline):
        from autoPyTorch.pipeline.nodes.network_selector import NetworkSelector
        from autoPyTorch.pipeline.nodes.loss_module_selector import LossModuleSelector
        from autoPyTorch.pipeline.nodes.metric_selector import MetricSelector
        from autoPyTorch.pipeline.nodes.train_node import TrainNode
        from autoPyTorch.pipeline.nodes.cross_validation import CrossValidation

        import torch.nn as nn
        from autoPyTorch.components.metrics.standard_metrics import mae, rmse

        AutoNetFeatureData._apply_default_pipeline_settings(pipeline)

        net_selector = pipeline[NetworkSelector.get_name()]
        net_selector.add_final_activation('none', nn.Sequential())

        loss_selector = pipeline[LossModuleSelector.get_name()]
        loss_selector.add_loss_module('l1_loss', nn.L1Loss)
        #QuantileLoss나 QinputLoss는 quantile 또는 weights를 입력으로 받아야 하므로
        #여기서 add_loss_module하기 힘들다.

        metric_selector = pipeline[MetricSelector.get_name()]
        metric_selector.add_metric('mean_abs_error', mae, loss_transform=False, requires_target_class_labels=False)
        metric_selector.add_metric('rmse', rmse, loss_transform=False, requires_target_class_labels=False)
        #metric_ql과 metric_qil은 모두 quantile을 입력으로 받아야 하므로 여기서 add_metric 하기 힘들다.
        #metric_selector.add_metric('metric_ql', metric_ql, loss_transform=False, requires_target_class_labels=False)
        #metric_selector.add_metric('metric_qil', metric_qil, loss_transform=False, requires_target_class_labels=False)



        train_node = pipeline[TrainNode.get_name()]
        train_node.default_minimize_value = True

        cv = pipeline[CrossValidation.get_name()]
        cv.use_stratified_cv_split_default = False
