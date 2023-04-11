# for SPTS build model
import sys
import math
import paddle
from paddle import nn
from paddle.vision.models import ResNet
from paddle.vision.models.resnet import BottleneckBlock
import paddle.nn.functional as F
# from ppocr.modeling.backbones import build_backbone
import numpy as np
from test_transformer import Transformer
sys.path.append("/data/xiaoqihang/myproject/SPTS_Paddle")
from reprod_log import ReprodDiffHelper, ReprodLogger
diff_helper = ReprodDiffHelper()
reprod_logger = ReprodLogger()

def build_neck(config):
    support_dict = [
        'SPTS_Neck'
    ]

    module_name = config.pop('name')
    assert module_name in support_dict, Exception('neck only support {}'.format(
        support_dict))

    module_class = eval(module_name)(**config)
    return module_class

def build_head(config):
    support_dict = [
        'SPTS_Head'
    ]

    if config['name'] == 'DRRGHead':
        from .det_drrg_head import DRRGHead
        support_dict.append('DRRGHead')

    #table head

    module_name = config.pop('name')
    assert module_name in support_dict, Exception('head only support {}'.format(
        support_dict))
    module_class = eval(module_name)(**config)
    return module_class


class Position(nn.Layer):
    def __init__(self, config):
        super(Position, self).__init__()
        self.position_embedding = build_position_embedding(config)

    def forward(self, data):
        src, mask = data['image'], data['mask']
        pos, mask = self.position_embedding(src, mask)
        return src, mask, pos

class Joiner(nn.Sequential):
    def __init__(self, backbone):
        super(Joiner, self).__init__(backbone)


class MLP(nn.Layer):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(nn.Linear(n, k) for n, k in zip(
            [input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SPTS(nn.Layer):
    def __init__(self, config):
        super(SPTS, self).__init__()
        self.backbone = ResNet(BottleneckBlock, depth=50, with_pool=False, num_classes=-1)
        self.input_proj = nn.Conv2D(self.backbone.inplanes,
                                    config['Position']['tfm_hidden_dim'],
                                    kernel_size=1)
        self.position = Position(config['Position'])
        self.init_param(config)
        
        # paddle.set_default_dtype("float64")
        self.transformer = Transformer(d_model=self.d_model,
                        nhead=self.nhead,
                        num_encoder_layers=self.num_encoder_layers,
                        num_decoder_layers=self.num_decoder_layers,
                        dim_feedforward=self.dim_feedforward,
                        dropout=self.dropout,
                        normalize_before=self.normalize_before,
                        pad_token_id=self.padding_index,
                        num_classes=self.num_classes,
                        max_position_embeddings=self.max_position_embeddings,
                        return_intermediate_dec=False,
                        num_bins=self.num_bins,
                        eos_index=self.eos_index)
        self.vocab_embed = MLP(self.d_model, self.d_model, self.num_classes, 3)

    def forward(self, data):
        img, seq = data['image'], data['sequence']
        outputs = self.backbone(img)
        
        diff_helper.compare_info({"after_backbone":outputs.numpy()}, {"after_backbone":np.load("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/a_b.npy")})
        diff_helper.report(path="/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/ab_diff.log", diff_threshold=1e-5)

        outputs = self.input_proj(outputs)

        data['image'] = outputs
        src, mask, pos = self.position(data)
        if self.training:
            # sequence = seq[:, 0, :].astype("int")
            sequence = seq[:, 0, :].astype("int")
        else:
            sequence = seq

        outputs = self.transformer(src,
                                    mask,
                                    pos,
                                    sequence,
                                    self.vocab_embed)
        return outputs

    def init_param(self, config):
        self.num_bins = config['Transformer']['num_bins']
        self.max_num_text_ins = config['Transformer']['max_num_text_ins']
        self.chars = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
        num_char_classes = len(self.chars) + 1 # unknown
        recog_pad_index = self.num_bins + num_char_classes
        self.eos_index = recog_pad_index + 1
        self.sos_index = self.eos_index + 1
        self.padding_index = self.sos_index + 1
        self.num_classes = self.padding_index + 1
        self.max_position_embeddings = (2 + 25) * self.max_num_text_ins + 1

        self.d_model = config['Transformer']['tfm_hidden_dim']
        self.nhead = config['Transformer']['nhead']
        self.num_encoder_layers = config['Transformer']['num_encoder_layers']
        self.num_decoder_layers = config['Transformer']['num_decoder_layers']
        self.dim_feedforward = config['Transformer']['dim_feedforward']
        self.dropout = config['Transformer']['dropout']
        self.normalize_before = config['Transformer']['normalize_before']
        self.return_intermediate_dec = config['Transformer']['return_intermediate_dec']

class PositionEmbeddingSine(nn.Layer):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, feature, mask):
        mask = F.interpolate(mask[None].astype("float32"), \
            size=feature.shape[-2:]).astype(bool)[0]
        not_mask = ~mask
        y_embed = not_mask.astype("float32").cumsum(1)
        x_embed = not_mask.astype("float32").cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = paddle.arange(self.num_pos_feats, dtype=paddle.float32)
        dim_t = self.temperature ** (2 * (dim_t / 2).floor() / 
                                     self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = paddle.stack((pos_x[:, :, :, 0::2].sin(), 
                              pos_x[:, :, :, 1::2].cos()), axis=4).flatten(3)
        pos_y = paddle.stack((pos_y[:, :, :, 0::2].sin(), 
                              pos_y[:, :, :, 1::2].cos()), axis=4).flatten(3)
        pos = paddle.concat((pos_y, pos_x), axis=3).transpose([0, 3, 1, 2])
        return pos, mask

def build_position_embedding(config):
    """建立pos_embed"""
    N_steps = config['tfm_hidden_dim'] // 2
    if config['position_embedding'] in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif config['position_embedding'] in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError("not supported \
            {}".format(config['position_embedding']))

    return position_embedding

if __name__ == "__main__":
    # model = resnet50()
    model = ResNet(BottleneckBlock, depth=50, with_pool=False, num_classes=-1)
    x = paddle.rand([1, 3, 896, 1600])
    output = model(x)
    pass


class PositionEmbeddingLearned(nn.Layer):
    """可学习embed"""
    def __init__(self, num_pos_feats=256):
        super().__init__()

    def forward(self, data):
        pass
