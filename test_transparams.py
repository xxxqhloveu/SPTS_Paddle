import sys
import torch
import torchvision
import paddle
import numpy as np

# paddle_model = paddle.load("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/output/save_model/SPTS.pdparams")

torch_state_dict = torch.load("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/output/source_model_param/ic13.pth")
# torch_state_dict = torch.load("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/output/source_model_param/ic15.pth")
# torch_state_dict = torch.load("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/output/source_model_param/ctw1500.pth")
# torch_state_dict = torch.load("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/output/source_model_param/totaltext.pth")
torch_model_dict = torch_state_dict['model']
# # SPTS_torch_model = torch.load("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/torch_model.pth")

paddle_model = paddle.load("/data/xiaoqihang/myproject/race/paper/SPTS_paddle/output/save_model/SPTS.pdparams")
paddle_dict = paddle_model.keys()
torch_dict = torch_state_dict.keys()
torch_model = torch_state_dict['model']
paddle_torch_match = {}
other_dict = []
import copy
copy_paddle_dict = copy.deepcopy(list(paddle_dict))
copy_torch_dict = copy.deepcopy(list(torch_model))
# print(paddle_model['backbone.layer1.0.conv2.weight'])
copy_paddle_model = copy.deepcopy(paddle_model)
print(len(copy_paddle_dict))
# backbone_num = 0
# transformer_num = 0
# vocab_embed_num = 0
# input_proj_num = 0
# other_num = 0
from collections import OrderedDict

multi_paddle_model = OrderedDict()
multi_torch_model = OrderedDict()
for torch_k in copy_torch_dict:
    copy_torch_k = copy.deepcopy(torch_k)
    # if "encoder" in torch_k and "attn" in torch_k and "weight" in torch_k:
    #     print(torch_k)
    if "backbone" in torch_k:
        torch_k = torch_k.replace("backbone.0.body.", "backbone.")
        torch_k = torch_k.replace("running_mean", "_mean")
        torch_k = torch_k.replace("running_var", "_variance")
        copy_paddle_model[torch_k] = paddle.to_tensor(torch_model[copy_torch_k].detach().cpu().numpy())
    elif "transformer" in torch_k:
        if "embedding" in torch_k:
            copy_paddle_model[torch_k] = paddle.to_tensor(
                torch_model[copy_torch_k].detach().cpu().numpy())
        elif "in_proj"in torch_k:
            q_torch, k_torch, v_torch = torch_model[copy_torch_k].chunk(3, 0)
            if "transformer.encoder.layers.0.self_attn" in torch_k:
                if "weight" in torch_k:
                    q = torch_k.replace("transformer.encoder.layers.0.self_attn.in_proj_weight", "q_proj.weight")
                    k = torch_k.replace("transformer.encoder.layers.0.self_attn.in_proj_weight", "k_proj.weight")
                    v = torch_k.replace("transformer.encoder.layers.0.self_attn.in_proj_weight", "v_proj.weight")
                    multi_paddle_model[q] = paddle.to_tensor(
                        q_torch.transpose(1, 0).detach().numpy())
                    multi_paddle_model[k] = paddle.to_tensor(
                        k_torch.transpose(1, 0).detach().numpy())
                    multi_paddle_model[v] = paddle.to_tensor(
                        v_torch.transpose(1, 0).detach().numpy())
                    multi_torch_k = torch_k.replace("transformer.encoder.layers.0.self_attn.", "")
                    multi_torch_model[multi_torch_k] = torch_model[copy_torch_k]
                    
                elif "bias" in torch_k:
                    q = torch_k.replace("transformer.encoder.layers.0.self_attn.in_proj_bias", "q_proj.bias")
                    k = torch_k.replace("transformer.encoder.layers.0.self_attn.in_proj_bias", "k_proj.bias")
                    v = torch_k.replace("transformer.encoder.layers.0.self_attn.in_proj_bias", "v_proj.bias")
                    multi_paddle_model[q] = paddle.to_tensor(q_torch.detach().numpy())
                    multi_paddle_model[k] = paddle.to_tensor(k_torch.detach().numpy())
                    multi_paddle_model[v] = paddle.to_tensor(v_torch.detach().numpy())
                    multi_torch_k = torch_k.replace("transformer.encoder.layers.0.self_attn.", "")
                    multi_torch_model[multi_torch_k] = torch_model[copy_torch_k]
                    

            if "weight" in torch_k:
                q = torch_k.replace("in_proj_weight", "q_proj.weight")
                k = torch_k.replace("in_proj_weight", "k_proj.weight")
                v = torch_k.replace("in_proj_weight", "v_proj.weight")
                copy_paddle_model[q] = paddle.to_tensor(q_torch.transpose(1, 0).detach().numpy())
                copy_paddle_model[k] = paddle.to_tensor(k_torch.transpose(1, 0).detach().numpy())
                copy_paddle_model[v] = paddle.to_tensor(v_torch.transpose(1, 0).detach().numpy())
            elif "bias" in torch_k:
                q = torch_k.replace("in_proj_bias", "q_proj.bias")
                k = torch_k.replace("in_proj_bias", "k_proj.bias")
                v = torch_k.replace("in_proj_bias", "v_proj.bias")
                copy_paddle_model[q] = paddle.to_tensor(q_torch.detach().numpy())
                copy_paddle_model[k] = paddle.to_tensor(k_torch.detach().numpy())
                copy_paddle_model[v] = paddle.to_tensor(v_torch.detach().numpy())
            else:
                print(torch_k)
            copy_paddle_dict.remove(q)
            copy_paddle_dict.remove(k)
            copy_paddle_dict.remove(v)
        elif "out_proj" in torch_k or "linear" in torch_k:
            if "transformer.encoder.layers.0.self_attn" in torch_k:
                out_k = torch_k.replace("transformer.encoder.layers.0.self_attn.", "")
                if "weight" in torch_k:
                    multi_paddle_model[out_k] = paddle.to_tensor(
                        torch_model[copy_torch_k].transpose(1, 0).detach().cpu().numpy())
                    multi_torch_k = torch_k.replace("transformer.encoder.layers.0.self_attn.", "")
                    multi_torch_model[multi_torch_k] = torch_model[copy_torch_k]
                    
                elif "bias" in torch_k:
                    multi_paddle_model[out_k] = paddle.to_tensor(
                        torch_model[copy_torch_k].detach().cpu().numpy())
                    multi_torch_k = torch_k.replace("transformer.encoder.layers.0.self_attn.", "")
                    multi_torch_model[multi_torch_k] = torch_model[copy_torch_k]
                    
            if "weight" in torch_k:
                copy_paddle_model[torch_k] = paddle.to_tensor(torch_model[copy_torch_k].transpose(1, 0).detach().cpu().numpy())
            elif "bias" in torch_k:
                copy_paddle_model[torch_k] = paddle.to_tensor(torch_model[copy_torch_k].detach().cpu().numpy())
            else:
                print(torch_k)
        else:
            copy_paddle_model[torch_k] = paddle.to_tensor(torch_model[copy_torch_k].detach().cpu().numpy())
    elif "input_proj" in torch_k:
        copy_paddle_model[torch_k] = paddle.to_tensor(torch_model[copy_torch_k].detach().cpu().numpy())
    elif "vocab_embed" in torch_k:
        if "weight" in torch_k:
                copy_paddle_model[torch_k] = paddle.to_tensor(torch_model[copy_torch_k].transpose(1, 0).detach().cpu().numpy())
        elif "bias" in torch_k:
            copy_paddle_model[torch_k] = paddle.to_tensor(torch_model[copy_torch_k].detach().cpu().numpy())
        else:
            print(torch_k)
    else:
        print(torch_k)
    if ("transformer" not in torch_k) or ("in_proj" not in torch_k):
        copy_paddle_dict.remove(torch_k)
# torch.save(multi_torch_model, "/data/xiaoqihang/myproject/race/paper/SPTS_paddle/multi_torch.pth")
# paddle.save(multi_paddle_model, "/data/xiaoqihang/myproject/race/paper/SPTS_paddle/multi_paddle.pdparams")
print(len(copy_paddle_dict)) # 2(en-de)*3(qkv)*6(num_layer)*2(wei-bias)*2(self-multi)=108
# print(copy_paddle_dict)
paddle.save(copy_paddle_model, "/data/xiaoqihang/myproject/race/paper/SPTS_paddle/paddle_ic13.pdparams")
pass



# for torch_k in copy_torch_dict:
#     copy_torch_k = copy.deepcopy(torch_k)
#     if "encoder" in torch_k and "bias" in torch_k:
#         print(torch_k)
#     torch_k = torch_k.replace("backbone.0.body.", "backbone.")
#     torch_k = torch_k.replace("running_mean", "_mean")
#     torch_k = torch_k.replace("running_var", "_variance")
#     if "encoder" in torch_k and "attn" in torch_k and "weight" in torch_k:
#         if "in_proj" in torch_k:
#             in_proj = torch_model[torch_k].transpose(1, 0).detach().cpu().numpy()
#             shape = in_proj.shape[1] // 3
#             if "weight" in torch_k:
#                 q = torch_k.replace("in_proj_weight", "q_proj.weight")
#                 k = torch_k.replace("in_proj_weight", "k_proj.weight")
#                 v = torch_k.replace("in_proj_weight", "v_proj.weight")
#                 copy_paddle_model[q] = in_proj[:, :shape]
#                 copy_paddle_model[k] = in_proj[:, shape:2*shape]
#                 copy_paddle_model[v] = in_proj[:, 2*shape:]
#                 print(q, copy_paddle_model[q].shape)
#                 print(k, copy_paddle_model[k].shape)
#                 print(v, copy_paddle_model[v].shape)
#                 copy_paddle_dict.remove(q)
#                 copy_paddle_dict.remove(k)
#                 copy_paddle_dict.remove(v)
#         elif "out_proj" in torch_k:
#             copy_paddle_model[torch_k] = torch_model[copy_torch_k].transpose(1, 0).detach().cpu().numpy()
#             copy_paddle_dict.remove(torch_k)
#         else:
#             print(torch_k)
#     elif torch_k in copy_paddle_dict:
#         if ("linear" in torch_k or "vocab_embed" in torch_k) and "weight" in torch_k:
#             print(torch_k, torch_model[copy_torch_k].shape)
#             copy_paddle_model[torch_k] = torch_model[copy_torch_k].transpose(1, 0).detach().cpu().numpy()
#         # elif "decoder" in torch_k and "multihead_attn" in torch_k and "weight" in torch_k:
#         #     print(torch_k, torch_model[copy_torch_k].shape)
#         #     copy_paddle_model[torch_k] = torch_model[copy_torch_k].detach().cpu().numpy()
#         elif "attn" in torch_k and "out_proj.weight" in torch_k:
#             print(torch_k, torch_model[copy_torch_k].shape)
#             copy_paddle_model[torch_k] = torch_model[copy_torch_k].detach().cpu().numpy().transpose(1, 0)
#         else:
#             copy_paddle_model[torch_k] = torch_model[copy_torch_k].detach().cpu().numpy()
#         paddle_torch_match[torch_k] = copy_torch_k
#         copy_paddle_dict.remove(torch_k)
#     else:
#         # other_dict.append(torch_k)
#         in_proj = torch_model[torch_k].detach().cpu().numpy()
#         shape = in_proj.shape[0] // 3
#         if "weight" in torch_k:
#             q = torch_k.replace("in_proj_weight", "q_proj.weight")
#             k = torch_k.replace("in_proj_weight", "k_proj.weight")
#             v = torch_k.replace("in_proj_weight", "v_proj.weight")
#             copy_paddle_model[q] = in_proj[:shape, :].transpose(1, 0)
#             copy_paddle_model[k] = in_proj[shape:2*shape, :].transpose(1, 0)
#             copy_paddle_model[v] = in_proj[2*shape:, :].transpose(1, 0)
#             print(q, copy_paddle_model[q].shape)
#             print(k, copy_paddle_model[k].shape)
#             print(v, copy_paddle_model[v].shape)
#         else:
#             q = torch_k.replace("in_proj_bias", "q_proj.bias")
#             k = torch_k.replace("in_proj_bias", "k_proj.bias")
#             v = torch_k.replace("in_proj_bias", "v_proj.bias")
#             copy_paddle_model[q] = in_proj[:shape]
#             copy_paddle_model[k] = in_proj[shape:2*shape]
#             copy_paddle_model[v] = in_proj[2*shape:]
#         copy_paddle_dict.remove(q)
#         copy_paddle_dict.remove(k)
#         copy_paddle_dict.remove(v)

# for torch_k in copy_torch_dict:
#     copy_torch_k = copy.deepcopy(torch_k)
#     if "encoder" in torch_k:
#         print(torch_k)
#     torch_k = torch_k.replace("backbone.0.body.", "backbone.")
#     torch_k = torch_k.replace("running_mean", "_mean")
#     torch_k = torch_k.replace("running_var", "_variance")
#     torch_k = torch_k.replace("in_proj_weight", "in_proj.weight")
#     torch_k = torch_k.replace("in_proj_bias", "in_proj.bias")
#     if torch_k in copy_paddle_dict:
#         if "transformer" in torch_k and "proj.weight" in torch_k:
#             print(torch_k, torch_model[copy_torch_k].shape)
#             copy_paddle_model[torch_k] = torch_model[copy_torch_k].transpose(1, 0).detach().cpu().numpy()
#         elif ("linear" in torch_k or "vocab_embed" in torch_k) and "weight" in torch_k:
#             print(torch_k, torch_model[copy_torch_k].shape)
#             copy_paddle_model[torch_k] = torch_model[copy_torch_k].transpose(1, 0).detach().cpu().numpy()
#         elif "attn" in torch_k and "out_proj.weight" in torch_k:
#             print(torch_k, torch_model[copy_torch_k].shape)
#             copy_paddle_model[torch_k] = torch_model[copy_torch_k].detach().cpu().numpy().transpose(1, 0)
#         else:
#             copy_paddle_model[torch_k] = torch_model[copy_torch_k].detach().cpu().numpy()
#         paddle_torch_match[torch_k] = copy_torch_k
#         copy_paddle_dict.remove(torch_k)
#     else:
#         pass

# for torch_k in copy_torch_dict:
#     copy_torch_k = copy.deepcopy(torch_k)
#     if "encoder" in torch_k and "attn" in torch_k and "0" in torch_k:
#         print(torch_k)
#     torch_k = torch_k.replace("backbone.0.body.", "backbone.")
#     torch_k = torch_k.replace("running_mean", "_mean")
#     torch_k = torch_k.replace("running_var", "_variance")
#     if torch_k in copy_paddle_dict:
#         if ("linear" in torch_k or "vocab_embed" in torch_k) and "weight" in torch_k:
#             print(torch_k, torch_model[copy_torch_k].shape)
#             copy_paddle_model[torch_k] = torch_model[copy_torch_k].transpose(1, 0).detach().cpu().numpy()
#         # elif "decoder" in torch_k and "multihead_attn" in torch_k and "weight" in torch_k:
#         #     print(torch_k, torch_model[copy_torch_k].shape)
#         #     copy_paddle_model[torch_k] = torch_model[copy_torch_k].detach().cpu().numpy()
#         elif "attn" in torch_k and "out_proj.weight" in torch_k:
#             print(torch_k, torch_model[copy_torch_k].shape)
#             copy_paddle_model[torch_k] = torch_model[copy_torch_k].detach().cpu().numpy().transpose(1, 0)
#         else:
#             copy_paddle_model[torch_k] = torch_model[copy_torch_k].detach().cpu().numpy()
#         paddle_torch_match[torch_k] = copy_torch_k
#         copy_paddle_dict.remove(torch_k)
#     else:
#         # other_dict.append(torch_k)
#         in_proj = torch_model[torch_k].detach().cpu().numpy()
#         shape = in_proj.shape[0] // 3
#         if "weight" in torch_k:
#             q = torch_k.replace("in_proj_weight", "q_proj.weight")
#             k = torch_k.replace("in_proj_weight", "k_proj.weight")
#             v = torch_k.replace("in_proj_weight", "v_proj.weight")
#             copy_paddle_model[q] = in_proj[:shape, :].transpose(1, 0)
#             copy_paddle_model[k] = in_proj[shape:2*shape, :].transpose(1, 0)
#             copy_paddle_model[v] = in_proj[2*shape:, :].transpose(1, 0)
#             print(q, copy_paddle_model[q].shape)
#             print(k, copy_paddle_model[k].shape)
#             print(v, copy_paddle_model[v].shape)
#         else:
#             q = torch_k.replace("in_proj_bias", "q_proj.bias")
#             k = torch_k.replace("in_proj_bias", "k_proj.bias")
#             v = torch_k.replace("in_proj_bias", "v_proj.bias")
#             copy_paddle_model[q] = in_proj[:shape]
#             copy_paddle_model[k] = in_proj[shape:2*shape]
#             copy_paddle_model[v] = in_proj[2*shape:]
#         copy_paddle_dict.remove(q)
#         copy_paddle_dict.remove(k)
#         copy_paddle_dict.remove(v)
# print(len(copy_paddle_dict)) # 2(en-de)*3(qkv)*6(num_layer)*2(wei-bias)*2(self-multi)=108
# paddle.save(copy_paddle_model, "/data/xiaoqihang/myproject/race/paper/SPTS_paddle/e399_ic15_copy.pdparams")
# pass