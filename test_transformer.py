import copy
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from typing import Optional
from paddle import Tensor
import numpy as np

import sys
sys.path.append("/data/xiaoqihang/myproject/SPTS_Paddle")
from reprod_log import ReprodDiffHelper, ReprodLogger
diff_helper = ReprodDiffHelper()
reprod_logger = ReprodLogger()

weight_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())
bias_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())

class Transformer(nn.Layer):

    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, 
                 dropout, normalize_before, pad_token_id, num_classes, max_position_embeddings, 
                 return_intermediate_dec, num_bins, eos_index, activation="relu"):
        super(Transformer, self).__init__()
        self.embedding = DecoderEmbeddings(num_classes, d_model, pad_token_id, max_position_embeddings, dropout)
        if num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec)
        self.nhead = nhead
        self.d_model = d_model
        self.num_bins = num_bins
        self.eos_index = eos_index
        self.num_encoder_layers = num_encoder_layers
        self.max_position_embeddings = max_position_embeddings

    def forward(self, src, mask, pos_embed, seq, vocab_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).transpose([0, 2, 1])
        
        diff_helper.compare_info({"after_input_proj":src.numpy()}, {"after_input_proj":np.load("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/a_i.npy")})
        diff_helper.report(path="/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/ab_diff.log", diff_threshold=1e-5)

        pos_embed = pos_embed.flatten(2).transpose([0, 2, 1])
        mask = mask.reshape([mask.shape[0], -1])
        
        ###################################
        masks = paddle.zeros(mask.shape)
        masks[mask] = float('-inf')# bool转为inf
        mask = masks
        ###################################
        
        if self.num_encoder_layers > 0:
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed.astype("float16"))
        else:
            memory = src

        diff_helper.compare_info({"after_encoder":memory.numpy()}, {"after_encoder":np.load("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/a_e.npy")})
        diff_helper.report(path="/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/ab_diff.log", diff_threshold=1e-5)


        query_embed = self.embedding.position_embeddings.weight.unsqueeze(0)
        query_embed = paddle.concat([query_embed for _ in range(bs)], axis=1)
        if self.training:
            tgt = self.embedding(seq)
            
            diff_helper.compare_info({"after_decoder_embed":tgt.numpy()}, 
                                        {"after_decoder_embed":np.load("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/a_d_emb.npy")})
            diff_helper.report(path="/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/ab_diff.log", diff_threshold=1e-5)

            hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed[:, :tgt.shape[1], :],
                          tgt_mask=generate_square_subsequent_mask(tgt.shape[1]))
            
            diff_helper.compare_info({"after_decoder":hs.numpy()}, {"after_decoder":np.load("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/a_d.npy")})
            diff_helper.report(path="/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/ab_diff.log", diff_threshold=1e-5)

            return vocab_embed(hs)
        else:
            probs = []
            for i in range(self.max_position_embeddings):
                tgt = self.embedding(seq)

                diff_helper.compare_info({"after_decoder_embed":tgt.numpy()}, 
                                         {"after_decoder_embed":np.load("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/a_d_emb.npy")})
                diff_helper.report(path="/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/ab_diff.log", diff_threshold=1e-5)

                hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed[:, :i+1, :],
                          tgt_mask=generate_square_subsequent_mask(i+1))

                # # print(hs)
                diff_helper.compare_info({"after_decoder":hs.numpy()}, {"after_decoder":np.load("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/a_d.npy")})
                diff_helper.report(path="/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/ab_diff.log", diff_threshold=1e-5)

                out = vocab_embed(hs[:, -1, :])
                out = F.softmax(out)

                # bins chars eos sos padding
                if i % 27 == 0: # coordinate or eos
                    out[:, self.num_bins:self.eos_index] = 0
                    out[:, self.eos_index+1:] = 0
                elif i % 27 == 1: # coordinate
                    out = out[:, :self.num_bins]
                else: # chars
                    out[:, :self.num_bins] = 0
                    out[:, self.eos_index:] = 0

                prob, extra_seq = out.topk(axis=-1, k=1)
                seq = paddle.concat([seq, extra_seq], axis=1)
                probs.append(prob)
                if extra_seq[0] == self.eos_index:
                    break

            seq = seq[:, 1:] # remove start index
            return seq, paddle.concat(probs, axis=-1)

class DecoderEmbeddings(nn.Layer):
    def __init__(self, vocab_size, hidden_dim, pad_token_id, max_position_embeddings, dropout):
        super(DecoderEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id, weight_attr=weight_attr)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim, weight_attr=weight_attr)

        self.LayerNorm = paddle.nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        input_shape = x.shape
        seq_length = input_shape[1]

        position_ids = paddle.arange(seq_length)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (paddle.triu(paddle.ones([sz, sz])) == 1).transpose([1, 0])
    masks = paddle.zeros(mask.shape).astype('float32')
    masks[~mask] = float('-inf')
    return masks


class TransformerEncoder(nn.Layer):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Layer):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return paddle.stack(intermediate)

        return output

class TransformerEncoderLayer(nn.Layer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, nhead, dropout=dropout)
        
        # Implementation of Feedforward model        
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr=weight_attr, bias_attr=bias_attr)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr=weight_attr, bias_attr=bias_attr)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        
        
        diff_helper.compare_info({"bef_en_attn_v":src2.numpy()}, {"bef_en_attn_v":np.load("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/en_bef_attn_value.npy")})
        diff_helper.report(path="/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/ab_diff.log", diff_threshold=1e-5)

        diff_helper.compare_info({"bef_en_attn_q":q.numpy()}, {"bef_en_attn_q":np.load("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/en_bef_attn_query.npy")})
        diff_helper.report(path="/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/ab_diff.log", diff_threshold=1e-5)

        bs, km_len = src_key_padding_mask.shape
        km2attn_mask = paddle.expand(src_key_padding_mask, 
                                                       [self.self_attn.num_heads, bs, km_len])
        src2 = self.self_attn(query=q, key=k, value=src2, attn_mask=src_key_padding_mask)
        # src2 = self.self_attn(query=q, key=k, value=src2, attn_mask=src_mask)
        
        diff_helper.compare_info({"aft_en_attn_v":src2.numpy()}, {"aft_en_attn_v":np.load("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/en_aft_attn_value.npy")})
        diff_helper.report(path="/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/ab_diff.log", diff_threshold=1e-5)

        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear1(src2)
        src2 = self.activation(src2)
        src2 = self.dropout(src2)
        src2 = self.linear2(src2)
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Layer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr=weight_attr, bias_attr=bias_attr)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr=weight_attr, bias_attr=bias_attr)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.multihead_attn = nn.MultiHeadAttention(d_model, nhead, dropout=dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
        # if pos is None:
        #     return tensor
        # else:
        #     # print(tensor)
        #     # print(pos)
        #     return tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        
        diff_helper.compare_info({"bef_de_self_attn_q":q.numpy()}, {"bef_de_self_attn_q":np.load("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/de_bef_self_attn_q.npy")})
        diff_helper.report(path="/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/ab_diff.log", diff_threshold=1e-5)

        diff_helper.compare_info({"bef_de_self_attn_v":tgt2.numpy()}, {"bef_de_self_attn_v":np.load("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/de_bef_self_attn_v.npy")})
        diff_helper.report(path="/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/ab_diff.log", diff_threshold=1e-5)

        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask)
        
        diff_helper.compare_info({"aft_de_self_attn_v":tgt2.numpy()}, {"aft_de_self_attn_v":np.load("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/de_aft_self_attn_v.npy")})
        diff_helper.report(path="/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/ab_diff.log", diff_threshold=1e-5)

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        
        query=self.with_pos_embed(tgt2, query_pos)
        key=self.with_pos_embed(memory, pos)
        value=memory
        attn_mask=memory_key_padding_mask
        
        diff_helper.compare_info({"bef_de_cros_attn_q":query.numpy()}, {"bef_de_cros_attn_q":np.load("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/de_bef_cros_attn_q.npy")})
        diff_helper.report(path="/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/ab_diff.log", diff_threshold=1e-5)

        diff_helper.compare_info({"bef_de_cros_attn_k":key.numpy()}, {"bef_de_cros_attn_k":np.load("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/de_bef_cros_attn_k.npy")})
        diff_helper.report(path="/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/ab_diff.log", diff_threshold=1e-5)

        diff_helper.compare_info({"bef_de_cros_attn_v":value.numpy()}, {"bef_de_cros_attn_v":np.load("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/de_bef_cros_attn_v.npy")})
        diff_helper.report(path="/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/ab_diff.log", diff_threshold=1e-5)
        
        tgt2 = self.multihead_attn(query=query, key=key, value=value, attn_mask=attn_mask)
        
        diff_helper.compare_info({"aft_de_cros_attn_v":tgt2.numpy()}, {"aft_de_cros_attn_v":np.load("/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/de_aft_cros_attn_v.npy")})
        diff_helper.report(path="/data/xiaoqihang/myproject/SPTS_Paddle/test_diff/ab_diff.log", diff_threshold=1e-5)

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

def _get_clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    max_position_embeddings = (2 + 25) * args.max_num_text_ins + 1
    return Transformer(
        d_model=args.tfm_hidden_dim,
        nhead=args.tfm_nheads,
        num_encoder_layers=args.tfm_enc_layers,
        num_decoder_layers=args.tfm_dec_layers,
        dim_feedforward=args.tfm_dim_feedforward,
        dropout=args.tfm_dropout,
        normalize_before=args.tfm_pre_norm,
        pad_token_id=args.padding_index,
        num_classes=args.num_classes,
        max_position_embeddings=max_position_embeddings,
        return_intermediate_dec=False,
        num_bins=args.num_bins,
        eos_index=args.eos_index,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
