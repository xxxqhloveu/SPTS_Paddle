import sys
import cv2
import numpy as np
sys.path.append(".")
from reprod_log import ReprodDiffHelper, ReprodLogger
diff_helper = ReprodDiffHelper()

def make_data():
    # np.save(np.random.random((3, 720, 1280)), "./test_diff/save_npy/image.npy")
    image = cv2.imread("../test_images/img_1.jpg")
    image = image.transpose(2, 0, 1)
    # image = np.load("./test_diff/save_npy/image.npy")
    pad_img = np.zeros((3, 896, 1600))
    pad_img[:, :image.shape[1], :image.shape[2]] = image
    
    mask = np.ones(pad_img.shape[1:])   # 无法对齐的情况
    mask[:image.shape[1], :image.shape[2]] = np.zeros(image.shape[1:])
    
    # mask = np.zeros(pad_img.shape[1:])    # 能对齐的情况
    
    seq = np.array([[1098]])
    return pad_img, mask, seq

import torch
@torch.no_grad()
def load_torch_model():
    sys.path.append("../")
    from SPTS.model import build_model
    from SPTS.utils.parser import DefaultParser
    from SPTS.utils.checkpointer import Checkpointer
    from SPTS.utils.nested_tensor import NestedTensor
    parser = DefaultParser()
    args = parser.parse_args()
    # args.resume = "./pretrain/ic15.pth"
    # args.resume = "./pretrain/ctw1500.pth"
    # args.resume = "./pretrain/totaltext.pth"
    args.resume = "./pretrain/ic15.pth"
    args.checkpoint_freq = 2
    args.tfm_pre_norm = True
    args.distributed = False
    
    torch_model = build_model(args)
    checkpointer = Checkpointer(False)
    epoch = checkpointer.load(args.resume, torch_model)
    
    image, mask, seq = make_data()
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    samples = NestedTensor(tensor, mask)
    seq = torch.tensor(seq, dtype=torch.long)

    device = torch.device('cuda')
    torch_model = torch_model.to(device)
    samples = samples.to(device)
    seq = seq.to(device)
    torch_model.eval()
    torch_output, torch_prob = torch_model(samples, seq)
    
    torch_output_np = torch_output.cpu().numpy()
    torch_prob_np = torch_prob.cpu().numpy()
    print(torch_output_np)
    print(torch_prob_np)
    np.save("./test_diff/save_npy/torch_output.npy", torch_output_np)
    np.save("./test_diff/save_npy/torch_prob.npy", torch_prob_np)
    pass

import paddle
@paddle.no_grad()
def load_paddle_model():
    from test_program import preprocess
    # from ppocr.utils.save_load import load_model

    paddle.set_device("gpu")
    config, device, logger, vdl_writer = preprocess(is_train=True)
    from test_main import build_model
    paddle_model = build_model(config['Architecture'])
    # load_model(config, paddle_model)
    # paddle_model_dict = paddle.load("./pretrain/e399_ic15_copy.pdparams")
    # paddle_model_dict = paddle.load("./pretrain/paddle_ctw1500.pdparams")
    # paddle_model_dict = paddle.load("./pretrain/paddle_totaltext.pdparams")
    paddle_model_dict = paddle.load("./pretrain/paddle_ic15.pdparams")
    paddle_model.set_state_dict(paddle_model_dict)

    image, mask, seq = make_data()
    data = {}
    data['image'] = paddle.to_tensor(np.expand_dims(image, axis=0), dtype=paddle.float32)
    data['mask'] = paddle.to_tensor(np.expand_dims(mask, axis=0), dtype=paddle.float32)
    data['sequence'] = paddle.to_tensor(seq, dtype=paddle.int64)
    # paddle.to_tensor(data)
    paddle_model.eval()
    paddle_output, paddle_prob = paddle_model(data)
    paddle_output_np = paddle_output.numpy()
    paddle_prob_np = paddle_prob.numpy()
    print(paddle_output_np)
    print(paddle_prob_np)
    # diff_helper.compare_info({"result_output":paddle_output_np}, {"result_output":np.load("./test_diff/save_npy/torch_output.npy")})
    # diff_helper.report(path="./test_diff/save_npy/ab_diff.log", diff_threshold=1e-5)

    # diff_helper.compare_info({"result_prob":paddle_prob_np}, {"result_prob":np.load("./test_diff/save_npy/torch_prob.npy")})
    # diff_helper.report(path="./test_diff/save_npy/ab_diff.log", diff_threshold=1e-5)

    # np.save("./test_diff/save_npy/torch_output.npy", paddle_output_np)
    # np.save("./test_diff/save_npy/torch_prob.npy", paddle_prob_np)
    pass


if __name__ == "__main__":
    load_paddle_model()
    # load_torch_model()
    pass