import sys
import copy
import time
import paddle
import paddle.nn as nn
import paddle.distributed as dist
# from paddle.optimizer import lr
# from paddle.optimizer.lr import LRScheduler

sys.path.append('.')
# sys.path.append("/home/aistudio/PaddleOCR-2.6.0/")
# sys.path.append('/home/aistudio/external-libraries')
# from ppocr.utils.logging import get_logger
# from ppocr.utils.utility import set_seed
# from ppocr.modeling.architectures import apply_to_static
# from ppocr.utils.save_load import load_model, save_model

from test_util import save_model, apply_to_static, load_model, save_model, build_optimizer
from test_program import preprocess
from test_evaluation import validate
from test_data import build_dataloader
from test_model import SPTS

import importlib
def build_model(config):
    config = copy.deepcopy(config)
    if not "name" in config:
        from test_model import BaseModel
        arch = BaseModel(config)
    else:
        name = config.pop("name")
        mod = importlib.import_module(__name__)
        arch = getattr(mod, name)(config)
    return arch

def build_loss(config):
    weight = paddle.ones([config['num_classes']])
    weight[config['eos_index']] = config['eos_loss_coef']
    module_class = nn.CrossEntropyLoss(weight=weight, ignore_index=config['padding_index'])
    return module_class

def main(config, device, logger, vdl_writer):
    # init dist environment
    if config['Global']['distributed']:
        dist.init_parallel_env()
    global_config = config['Global']
    epoch_num = global_config['epoch_num']

    # build dataloder
    train_dataloader = build_dataloader(config, 'Train', device, logger)    
    if config['Eval']:
        valid_dataloader = build_dataloader(config, 'Eval', device, logger)
    else:
        valid_dataloader = None

    # build model
    model = build_model(config['Architecture'])
    model = apply_to_static(model, config, logger)

    # build loss
    loss_class = build_loss(config['Loss'])

    # build optim
    # from ppocr.optimizer import build_optimizer
    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=len(train_dataloader),
        model=model)

    # # build metric
    # eval_class = build_metric(config['Metric'])
    # eval_class = validate

    logger.info('train dataloader has {} iters'.format(len(train_dataloader)))
    
    # load pretrain model
    pre_best_model_dict = load_model(config, model, optimizer)
    # model_after_load = model.state_dict()

    if config['Global']['distributed']:
        model = paddle.DataParallel(model)
    global_step = 0
    
    for epoch in range(epoch_num):
        model.train()

        # training log
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()
        batch_past = 0
        print_freq = 1
        for batch_idx, data in enumerate(train_dataloader):
            image = data['image']
            data['image'] = paddle.concat([i for i in image], axis=0)
            sequence = data['sequence']
            data['sequence'] = paddle.concat([i for i in sequence], axis=0)
            mask = data['mask']
            data['mask'] = paddle.concat([i for i in mask], axis=0)

            train_reader_cost += time.time() - reader_start
            train_start = time.time()

            lr = optimizer.get_lr()
            sequence = data['sequence']
            out_seq_label = sequence[:, 1, :].astype("int")
            outputs = model(data)
            loss = loss_class(outputs, out_seq_label)
            loss.backward()
            optimizer.clear_grad()
            optimizer.step()
            global_step += 1
            lr_scheduler.step()

            train_run_cost += time.time() - train_start
            # acc = utils.accuracy(output, target).item()
            total_samples += data['image'].shape[0]
            batch_past += 1
            if global_step >= 0 and global_step % print_freq == 0:
                msg = "[Epoch {}/{}, iter: {}] lr: {}, loss: {}, avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {}, avg_ips: {:.5f} images/sec.".format(
                    epoch+1, epoch_num, batch_idx+1,
                    optimizer.get_lr(),
                    loss.item(), train_reader_cost / batch_past,
                    (train_reader_cost + train_run_cost) / batch_past,
                    total_samples / batch_past,
                    total_samples / (train_reader_cost + train_run_cost))
                # just log on 1st device
                if paddle.distributed.get_rank() <= 0:
                    logger.info(msg)
                    # print(msg)
                # sys.stdout.flush()
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
                batch_past = 0
        lr_scheduler.step()

        if epoch % 2 == 0:
            save_model_dir = global_config['save_model_dir']
            save_model(
                    model,
                    optimizer,
                    save_model_dir,
                    logger,
                    config,
                    is_best=True,
                    prefix='epoch_{}'.format(epoch),
                    # best_model_dict=best_model_dict,
                    epoch=epoch,
                    global_step=global_step)
            save_model(
                    model,
                    optimizer,
                    save_model_dir,
                    logger,
                    config,
                    is_best=True,
                    prefix='latest',
                    # best_model_dict=best_model_dict,
                    epoch=epoch,
                    global_step=global_step)

        # validate(model, valid_dataloader, epoch, config, logger)


if __name__ == '__main__':
    config, device, logger, vdl_writer = preprocess(is_train=True)
    # seed = config['Global']['seed'] if 'seed' in config['Global'] else 1024
    # set_seed(seed)
    main(config, device, logger, vdl_writer)
