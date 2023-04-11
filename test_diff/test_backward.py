import sys
import cv2
import paddle
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
sys.path.append("/data/xiaoqihang/myproject/SPTS_Paddle")
from reprod_log import ReprodLogger, ReprodDiffHelper
from PIL import Image

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
FLAGS_cudnn_deterministic = True

sys.path.append("/data/xiaoqihang/myproject/SPTS_Paddle")
from reprod_log import ReprodDiffHelper, ReprodLogger
diff_helper = ReprodDiffHelper()
reprod_logger = ReprodLogger()

def make_data():
    # np.save(np.random.random((3, 720, 1280)), "/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/image.npy")
    # import torchvision.transforms.functional as F
    # image = Image.open("/data/xiaoqihang/dataset/icdar2015/test_images/img_1.jpg")
    # image_torch = F.to_tensor(image)
    # np.save("/data/xiaoqihang/myproject/test/image_1.npy", image_torch.detach().cpu().numpy())
    # image = cv2.imread("/data/xiaoqihang/dataset/icdar2015/test_images/img_1.jpg")
    # image = image.transpose(2, 0, 1)
    # image = np.load("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/image.npy")
    image = np.load("/data/xiaoqihang/myproject/test/image_1.npy")
    pad_img = np.zeros((3, 896, 1600))
    pad_img[:, :image.shape[1], :image.shape[2]] = image
    
    mask = np.ones(pad_img.shape[1:])
    mask[:image.shape[1], :image.shape[2]] = np.zeros(image.shape[1:])
    
    # mask = np.zeros(pad_img.shape[1:])
    
    input_seq = np.array([[
        1098, 
        328, 171, 1039, 1069, 1078, 1065, 1088, 1073, 1083, 1000, 1052, 1072, 1069, 1065, 1084, 1082, 1069, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096,
        395, 170, 1059, 1016, 1022, 1061, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 
        407, 222, 1022, 1018, 1013, 1016, 1019, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 
        311, 384, 1035, 1065, 1082, 1080, 1065, 1082, 1075, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 
    ]])
    output_seq = np.array([[
        328, 171, 1039, 1069, 1078, 1065, 1088, 1073, 1083, 1000, 1052, 1072, 1069, 1065, 1084, 1082, 1069, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096,
        395, 170, 1059, 1016, 1022, 1061, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 
        407, 222, 1022, 1018, 1013, 1016, 1019, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 
        311, 384, 1035, 1065, 1082, 1080, 1065, 1082, 1075, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 
        1097
    ]])
    seq = np.array([[1098]])
    return pad_img, mask, input_seq, output_seq

def build_paddle_model():
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
    paddle_model_dict = paddle.load("/data/xiaoqihang/myproject/SPTS_Paddle/pretrain/paddle_ic15.pdparams")
    paddle_model.set_state_dict(paddle_model_dict)

    # # init loss
    # criterion_paddle = paddle.nn.CrossEntropyLoss()

    # # init optimizer
    # lr_scheduler_paddle = paddle.optimizer.lr.StepDecay(
    #     lr, step_size=max_iter // 3, gamma=lr_gamma)
    # opt_paddle = paddle.optimizer.AdamW(
    #     learning_rate=lr,
    #     parameters=paddle_model.parameters())

    from test_main import build_loss
    criterion = build_loss(config['Loss'])

    from test_util import build_optimizer
    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=1,
        model=paddle_model)

    for idx in range(1):
        image, mask, input_seq, output_seq = make_data()
        data = {}
        data['image'] = paddle.to_tensor(np.expand_dims(image, axis=0), dtype=paddle.float32)
        data['mask'] = paddle.to_tensor(np.expand_dims(mask, axis=0), dtype=paddle.float32)
        data['sequence'] = paddle.to_tensor(np.expand_dims(np.concatenate([input_seq, output_seq]), axis=0))
        sequence = data['sequence']
        output_seq_label = sequence[:, 1, :].astype("int")

        output = paddle_model(data)
        
        diff_helper.compare_info({"total_output":output.numpy()}, 
                                {"total_output":np.load("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/total_output.npy")})
        diff_helper.report(path="/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/ab_diff.log", diff_threshold=1e-5)

        loss = criterion(output, output_seq_label)

        diff_helper.compare_info({"loss":loss.numpy()}, 
                                {"loss":np.load(f"/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/loss_{idx}.npy")})
        diff_helper.report(path="/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/ab_diff.log", diff_threshold=1e-5)

        # reprod_logger.add("loss_{}".format(idx), loss.cpu().detach().numpy())
        # reprod_logger.add("lr_{}".format(idx), np.array(lr_scheduler.get_lr()))

        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        pass

    pass

def build_torch_model():
    max_iter = 3
    lr = 1e-3
    momentum = 0.9
    lr_gamma = 0.1
    # load torch model
    sys.path.append("/data/xiaoqihang/myproject")
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
    
    for idx in range(1):
        image, mask, input_seq, output_seq = make_data()
        tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        samples = NestedTensor(tensor, mask)
        seq = torch.tensor(input_seq, dtype=torch.long)
        device = torch.device('cuda')
        torch_model = torch_model.to(device)
        samples = samples.to(device)
        seq = seq.to(device)
        torch_output = torch_model(samples, seq)
        np.save("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/total_output.npy", torch_output.detach().cpu().numpy())
        
        seq = torch.tensor(output_seq, dtype=torch.long)
        seq = seq.to(device)
        loss = criterion(torch_output.transpose(1, 2), seq)

        np.save(f"/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/loss_{idx}.npy", loss.detach().cpu().numpy())
        # reprod_logger.add("loss_{}".format(idx), loss.cpu().detach().numpy())
        # reprod_logger.add("lr_{}".format(idx), np.array(lr_scheduler.get_lr()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pass

    pass


if __name__ == "__main__":
    build_paddle_model()
    # build_torch_model()

    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./result/losses_ref.npy")
    paddle_info = diff_helper.load_info("./result/losses_paddle.npy")

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="./result/log/backward_diff.log")



# label = {
#     {
#         "area": 1246.0,
#         "bbox": [377.0, 117.0, 89.0, 14.0],
#         "category_id": 1,
#         "id": 923,
#         "image_id": 1,
#         "iscrowd": 0,
#         "bezier_pts": [377, 117, 405, 117, 434, 117, 463, 117, 465, 130, 436, 130, 407, 130, 378, 130],
#         "rec": [39, 69, 78, 65, 88, 73, 83, 0, 52, 72, 69, 65, 84, 82, 69, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96]
#     }, 
#     {
#         "area": 459.0,
#         "bbox": [493.0, 115.0, 27.0, 17.0],
#         "category_id": 1,
#         "id": 924,
#         "image_id": 1,
#         "iscrowd": 0,
#         "bezier_pts": [493, 115, 501, 115, 510, 115, 519, 115, 519, 131, 510, 131, 501, 131, 493, 131],
#         "rec": [59, 16, 22, 61, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96]
#     }, 
#     {
#         "area": 1200.0,
#         "bbox": [492.0, 151.0, 60.0, 20.0],
#         "category_id": 1,
#         "id": 925,
#         "image_id": 1,
#         "iscrowd": 0,
#         "bezier_pts": [492, 151, 511, 151, 531, 151, 551, 151, 551, 170, 531, 170, 511, 170, 492, 170],
#         "rec": [22, 18, 13, 16, 19, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96]
#     }, 
#     {
#         "area": 705.0,
#         "bbox": [376.0, 198.0, 47.0, 15.0],
#         "category_id": 1,
#         "id": 926,
#         "image_id": 1,
#         "iscrowd": 0,
#         "bezier_pts": [376, 198, 391, 198, 406, 198, 422, 198, 422, 212, 406, 212, 391, 212, 376, 212],
#         "rec": [35, 65, 82, 80, 65, 82, 75, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96]
#     }, 
# }