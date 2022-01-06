import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class LabelSmoothingCrossEntropyLoss(nn.Layer):
    def __init__(self, smoothing=0.1):
        super().__init__()
        assert 0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1-smoothing

    def forward(self, x, target):
        log_probs = F.log_softmax(x) # [N, num_classes]
        # target_index 用于获取N个样本中每个样本的概率
        target_index = paddle.zeros([x.shape[0], 2], dtype='int64') # [N, 2]
        target_index[:, 0] = paddle.arange(x.shape[0])
        target_index[:, 1] = target

        nll_loss = -log_probs.gather_nd(index=target_index) # index: [N]
        smooth_loss = -log_probs.mean(axis=-1)
        loss = self.confidence*nll_loss + self.smoothing*smooth_loss
        return loss.mean()

class SoftTargetCrossEntropyLoss(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, target):
        loss = paddle.sum(-target * F.log_softmax(x, axis=-1), axis=-1)
        return loss.mean()

class DistillationLoss(nn.Layer): # Distillation蒸馏
    def __init__(self, 
                 base_criterion,
                 teacher_model,
                 distillation_type,
                 alpha,
                 tau):
        super().__init__()
        assert distillation_type in ['none', 'soft', 'hard']
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, targets):
        outputs, outputs_kd = outputs[0], outputs[1]
        base_loss = self.base_criterion(outputs, targets)
        if self.type == 'none':
            return base_loss

        with paddle.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.type == 'soft':
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / self.tau, axis=1),
                F.log_softmax(teacher_outputs / self.tau, axis=1),
                reduction='sum') * (self.tau * self.tau) / outputs_kd.numel()
        elif self.type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(axis=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss