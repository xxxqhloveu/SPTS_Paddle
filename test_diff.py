import numpy as np
import torch
import paddle
from reprod_log.reprod_log import ReprodDiffHelper, ReprodLogger
diff_helper = ReprodDiffHelper()

def model_diff():
    torch_info = torch.load("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/output/source_model_param/ic15.pth")
    paddle_info = paddle.load("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/e399_ic15.pdparams")

    for t_k in torch_info['model']:
        if "in_proj" in t_k:
            torch_info['model'].remove(torch_info['model'][t_k])
    for p_k in paddle_info:
        if "q_" in p_k or "k_" in p_k or "v_" in p_k:
            paddle_info.remove(torch_info['model'][p_k])
    diff_helper.compare_info(torch_info['model'], paddle_info)
    diff_helper.report(path="./result/log/forward_diff.log")

def result_diff():
    # load data
    # reprod_logger = ReprodLogger()
    # # paddle_out = paddle_model(paddle.to_tensor(inputs, dtype="float32"))
    # reprod_logger.add("logits", np.load("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/save_test_result/tgt_out/paddle_q.npy"))
    # reprod_logger.save("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/save_test_result/tgt_out/paddle_reprod_q.npy")
    # reprod_logger.add("logits", np.load("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/save_test_result/tgt_out/torch_q.npy"))
    # reprod_logger.save("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/save_test_result/tgt_out/torch_reprod_q.npy")
    
    
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/save_test_result/diff/p_src.npy")
    paddle_info = diff_helper.load_info("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/save_test_result/diff/t_src.npy")

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(
        path="/data/xiaoqihang/myproject/race/paper/SPTS_paddle/save_test_result/forward_diff.log", diff_threshold=1e-5)

def test():
    reprod_logger = ReprodLogger()
    
    paddle_np = np.load("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/save_test_result/tgt_out/paddle_pos_embed.npy")
    paddle_tensor = paddle.to_tensor(paddle_np)
    half_paddle_tensor = paddle.cast(paddle_tensor, dtype="float16")
    reprod_logger.add("logits", half_paddle_tensor.numpy())
    reprod_logger.save("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/save_test_result/tgt_out/paddle_reprod_half_pos.npy")

    torch_np = np.load("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/save_test_result/tgt_out/torch_pos_embed.npy")
    torch_tensor = torch.tensor(torch_np)
    half_torch_tensor = torch_tensor.half()
    reprod_logger.add("logits", half_torch_tensor.detach().numpy())
    reprod_logger.save("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/save_test_result/tgt_out/torch_reprod_half_pos.npy")
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/save_test_result/tgt_out/torch_reprod_half_pos.npy")
    paddle_info = diff_helper.load_info("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/save_test_result/tgt_out/paddle_reprod_half_pos.npy")

    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(
        path="/data/xiaoqihang/myproject/race/paper/SPTS_paddle/save_test_result/forward_diff.log", diff_threshold=1e-5)
    pass

if __name__ == "__main__":
    # model_diff()
    result_diff()
    # test()
    pass