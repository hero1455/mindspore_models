"""Using for eval the model checkpoint"""
import os

from absl import logging

from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context, Model

import src.constants as rconst
from src.dataset import create_dataset
from src.metrics import NCFMetric
from src.ncf import NCFModel, NetWithLossClass, TrainStepWrap, PredictWithSigmoid

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id

logging.set_verbosity(logging.INFO)

@moxing_wrapper()
def run_eval():
    """eval method"""
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target,
                        save_graphs=False,
                        device_id=get_device_id())

    layers = config.layers
    num_factors = config.num_factors
    topk = rconst.TOP_K
    num_eval_neg = rconst.NUM_EVAL_NEGATIVES

    ds_eval, num_eval_users, num_eval_items = create_dataset(test_train=False, data_dir=config.data_path,
                                                             dataset=config.dataset, train_epochs=0,
                                                             eval_batch_size=config.eval_batch_size)
    print("ds_eval.size: {}".format(ds_eval.get_dataset_size()))

    ncf_net = NCFModel(num_users=num_eval_users,
                       num_items=num_eval_items,
                       num_factors=num_factors,
                       model_layers=layers,
                       mf_regularization=0,
                       mlp_reg_layers=[0.0, 0.0, 0.0, 0.0],
                       mf_dim=16)
    param_dict = load_checkpoint(config.checkpoint_file_path)
    load_param_into_net(ncf_net, param_dict)

    loss_net = NetWithLossClass(ncf_net)
    train_net = TrainStepWrap(loss_net)
    eval_net = PredictWithSigmoid(ncf_net, topk, num_eval_neg)

    ncf_metric = NCFMetric()
    model = Model(train_net, eval_network=eval_net, metrics={"ncf": ncf_metric})

    ncf_metric.clear()
    out = model.eval(ds_eval)

    eval_file_path = os.path.join(config.output_path, config.eval_file_name)
    eval_file = open(eval_file_path, "a+")
    eval_file.write("EvalCallBack: HR = {}, NDCG = {}\n".format(out['ncf'][0], out['ncf'][1]))
    eval_file.close()
    print("EvalCallBack: HR = {}, NDCG = {}".format(out['ncf'][0], out['ncf'][1]))
    print("=" * 100 + "Eval Finish!" + "=" * 100)

if __name__ == '__main__':
    run_eval()
