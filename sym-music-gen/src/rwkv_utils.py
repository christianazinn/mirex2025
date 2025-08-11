import os
import types


os.environ["RWKV_MY_TESTING"] = "x070"
os.environ["RWKV_CTXLEN"] = "2048"
os.environ["RWKV_HEAD_SIZE_A"] = "64"
os.environ["RWKV_JIT_ON"] = "0"
from rwkv.model import RWKV


def get_rwkv_model(vocab_size):
    args = types.SimpleNamespace()

    args.n_embd = 512
    args.vocab_size = vocab_size
    args.n_layer = 8
    args.my_pos_emb = 0
    args.pre_ffn = 0
    args.my_testing = "0x070"
    args.head_size_a = 64
    args.head_size_divisor = 8
    args.dropout = 0
    args.head_qk = 0
    args.ctx_len = 2048
    args.grad_cp = 0

    model = RWKV(args)

    return model
