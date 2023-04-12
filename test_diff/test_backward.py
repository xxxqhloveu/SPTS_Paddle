import sys
import cv2
import paddle
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from PIL import Image

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
FLAGS_cudnn_deterministic = True

sys.path.append("/data/xiaoqihang/myproject/SPTS_Paddle")
sys.path.append(".")
from reprod_log import ReprodDiffHelper, ReprodLogger


def make_data(idx=3):
    # np.save(np.random.random((3, 720, 1280)), "./test_diff/image.npy")
    # import torchvision.transforms.functional as F
    # image = Image.open("./test_diff/img_1.jpg")
    # image_torch = F.to_tensor(image)
    # np.save("./test_diff/image_1.npy", image_torch.detach().cpu().numpy())
    # image = cv2.imread("./test_diff/img_1.jpg")
    # image = image.transpose(2, 0, 1)
    # image = np.load("./test_diff/image_1.npy")
    
    # image = np.load(f"./test_diff/image_{idx}.npy")
    # with open(f"./test_diff/image_{idx}.txt", 'r') as fp:
    #     lines = fp.read().split('\n')
    
    image = np.load(f"/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/image_{idx}.npy")
    with open(f"/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/image_{idx}.txt", 'r') as fp:
        lines = fp.read().split('\n')
        
        input_seq = np.array([[int(data) for data in lines[0].split(', ')[:-1]]])
        output_seq = np.array([[int(data) for data in lines[1].split(', ')[:-1]]])
    
    pad_img = np.zeros((3, 896, 1600))
    pad_img[:, :image.shape[1], :image.shape[2]] = image
    
    mask = np.ones(pad_img.shape[1:])
    mask[:image.shape[1], :image.shape[2]] = np.zeros(image.shape[1:])
    
    # mask = np.zeros(pad_img.shape[1:])
    
    return pad_img, mask, input_seq, output_seq

def build_paddle_model(reprod_logger):
    max_iter = 3
    lr = 1e-3
    lr_gamma = 0.1
    # load paddle model
    paddle.set_device("gpu")
    from test_program import preprocess
    config, device, logger, vdl_writer = preprocess(is_train=True)
    from test_main import build_model
    from test_util import save_model, apply_to_static, load_model, save_model, build_optimizer
    paddle_model = build_model(config['Architecture'])
    paddle_model = apply_to_static(paddle_model, config, logger)
    paddle_model.train()
    paddle_model_dict = paddle.load("./pretrain/paddle_ic15.pdparams")
    paddle_model.set_state_dict(paddle_model_dict)

    from test_main import build_loss
    criterion = build_loss(config['Loss'])

    from test_util import build_optimizer, build_lr_scheduler
    optimizer = build_optimizer(paddle_model, config)
    lr_scheduler = build_lr_scheduler(optimizer, -1, config)
    # optimizer, lr_scheduler = build_optimizer(
    #     config['Optimizer'],
    #     epochs=config['Global']['epoch_num'],
    #     step_each_epoch=1,
    #     model=paddle_model)

    for idx in range(1, 4):
        image, mask, input_seq, output_seq = make_data(idx)
        data = {}
        data['image'] = paddle.to_tensor(np.expand_dims(image, axis=0), dtype=paddle.float32)
        data['mask'] = paddle.to_tensor(np.expand_dims(mask, axis=0), dtype=paddle.float32)
        data['sequence'] = paddle.to_tensor(np.expand_dims(np.concatenate([input_seq, output_seq]), axis=0))
        sequence = data['sequence']
        output_seq_label = sequence[:, 1, :].astype("int")

        # reprod_logger.add(f"input_{idx}", data['image'].detach().cpu().numpy())
        
        output = paddle_model(data)
        
        # diff_helper.compare_info({"total_output":output.numpy()}, 
        #                         {"total_output":np.load("./test_diff/save_npy/total_output.npy")})
        # diff_helper.report(path="./test_diff/save_npy/ab_diff.log", diff_threshold=1e-5)

        loss = criterion(output, output_seq_label)

        # diff_helper.compare_info({"loss":loss.numpy()}, 
        #                         {"loss":np.load(f"./test_diff/save_npy/loss_{idx}.npy")})
        # diff_helper.report(path="./test_diff/save_npy/ab_diff.log", diff_threshold=1e-5)

        reprod_logger.add(f"loss_{idx}", loss.numpy())
        reprod_logger.add(f"lr_{idx}", np.array(lr_scheduler.get_lr()))
        print(loss.numpy(), lr_scheduler.get_lr())

        optimizer.clear_grad()
        loss.backward()
        # optimizer.step()
        lr_scheduler.step()
        pass
    
    reprod_logger.save("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/save_npy/losses_paddle.npy")
    pass

def build_torch_model(reprod_logger):
    # load torch model
    sys.path.append("..")
    from SPTS.model import build_model
    from SPTS.utils.parser import DefaultParser
    from SPTS.utils.checkpointer import Checkpointer
    from SPTS.utils.nested_tensor import NestedTensor
    from SPTS.optim import build_criterion, build_lr_scheduler, build_optimizer
    parser = DefaultParser()
    args = parser.parse_args()
    args.resume = "/data/xiaoqihang/myproject/SPTS_Paddle/pretrain/ic15.pth"
    args.checkpoint_freq = 1
    args.tfm_pre_norm = True
    args.distributed = False
    torch_model = build_model(args)
    checkpointer = Checkpointer(False)
    epoch = checkpointer.load(args.resume, torch_model)

    # # init loss
    # criterion_torch = torch.nn.CrossEntropyLoss()

    # # init optimizer
    # opt_torch = torch.optim.AdamW(torch_model.parameters(),
    #                             lr=lr)
    # lr_scheduler_torch = lr_scheduler.StepLR(
    #     opt_torch, step_size=max_iter // 3, gamma=lr_gamma)
    
    criterion = build_criterion(args)
    optimizer = build_optimizer(torch_model, args)
    lr_scheduler = build_lr_scheduler(optimizer, -1, args)
    torch_model.train()

    for idx in range(1, 4):
        image, mask, input_seq, output_seq = make_data(idx)
        tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        samples = NestedTensor(tensor, mask)
        seq = torch.tensor(input_seq, dtype=torch.long)
        device = torch.device('cuda')
        torch_model = torch_model.to(device)
        samples = samples.to(device)
        seq = seq.to(device)

        # reprod_logger.add(f"input_{idx}", tensor.detach().cpu().numpy())

        torch_output = torch_model(samples, seq)

        # np.save("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/save_npy/total_output.npy", torch_output.detach().cpu().numpy())

        seq = torch.tensor(output_seq, dtype=torch.long)
        seq = seq.to(device)
        loss = criterion(torch_output.transpose(1, 2), seq)

        # np.save(f"/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/save_npy/loss_{idx}.npy", loss.detach().cpu().numpy())
        reprod_logger.add(f"loss_{idx}", loss.detach().cpu().numpy())
        reprod_logger.add(f"lr_{idx}", np.array(lr_scheduler.get_lr()))
        print(loss.detach().cpu().numpy(), lr_scheduler.get_lr())

        optimizer.zero_grad()
        loss.backward()
        # optimizer.step()
        lr_scheduler.step()
        pass
    reprod_logger.save("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/save_npy/losses_torch.npy")
    pass


if __name__ == "__main__":
    reprod_logger = ReprodLogger()

    build_torch_model(reprod_logger)
    build_paddle_model(reprod_logger)

    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./test_diff/save_npy/losses_torch.npy")
    paddle_info = diff_helper.load_info("./test_diff/save_npy/losses_paddle.npy")

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="./result/log/backward_diff.log")
