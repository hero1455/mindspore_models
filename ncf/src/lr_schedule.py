"""lr generator for ncf"""
import math

def _linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    learning_rate = float(init_lr) + lr_inc * current_step
    return learning_rate

def _cosine_learning_rate(current_step, base_lr, warmup_steps, decay_steps):
    base = float(current_step - warmup_steps) / float(decay_steps)
    learning_rate = (1 + math.cos(base * math.pi)) / 2 * base_lr
    return learning_rate

def dynamic_lr(base_lr, total_steps, warmup_steps):
    """dynamic learning rate generator"""
    lr = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr.append(_linear_warmup_learning_rate(i, warmup_steps, base_lr, base_lr * 0.01))
        else:
            lr.append(_cosine_learning_rate(i, base_lr, warmup_steps, total_steps))

    return lr
