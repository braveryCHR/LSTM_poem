import torch as t
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from model import *
from torchnet import meter
import tqdm
from config import *
from test import *


# 给定首句生成诗歌
def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    results = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    input = t.Tensor([word2ix['<START>']]).view(1, 1).long()
    if Config.use_gpu:
        input = input.cuda()
    hidden = None

    # 若有风格前缀，则先用风格前缀生成hidden
    if prefix_words:
        # 第一个input是<START>，后面就是prefix中的汉字
        # 第一个hidden是None，后面就是前面生成的hidden
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    # 开始真正生成诗句，如果没有使用风格前缀，则hidden = None，input = <START>
    # 否则，input就是风格前缀的最后一个词语，hidden也是生成出来的
    for i in range(Config.max_gen_len):
        output, hidden = model(input, hidden)
        # print(output.shape)
        # 如果还在诗句内部，输入就是诗句的字，不取出结果，只为了得到
        # 最后的hidden
        if i < start_words_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        # 否则将output作为下一个input进行
        else:
            # print(output.data[0].topk(1))
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results


# 生成藏头诗
def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
    result = []
    start_words_len = len(start_words)
    input = (t.Tensor([word2ix['<START>']]).view(1, 1).long())
    if Config.use_gpu:
        input = input.cuda()
    # 指示已经生成了几句藏头诗
    index = 0
    pre_word = '<START>'
    hidden = None

    # 存在风格前缀，则生成hidden
    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = (input.data.new([word2ix[word]])).view(1, 1)

    # 开始生成诗句
    for i in range(Config.max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]
        # 说明上个字是句末
        if pre_word in {'。', '，', '?', '！', '<START>'}:
            if index == start_words_len:
                break
            else:
                w = start_words[index]
                index += 1
                # print(w,word2ix[w])
                input = (input.data.new([word2ix[w]])).view(1, 1)
        else:
            input = (input.data.new([top_index])).view(1, 1)
        result.append(w)
        pre_word = w
    return result
