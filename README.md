# pyTorch实现AI写诗

### 前言

这几天主要在熟悉pyTorch，俗话说：“人生苦短，我用pyTorch”，在从TensorFlow转到pytorch之后，除了“爽”以外，我实在找不到其他形容词，简洁的语法，明了的封装，加上大大降低的debug难度，我从此入了pyTorch的坑。

为了熟悉pyTorch，我最近做了几个小项目，今天分享给大家的是一个非常有趣的项目——AI写诗。

该项目参考了《pytorch入门与实践》的教程。



### 先给大家看看效果

#### 第一种模式：给出首句续写全诗

> **春江潮水连海平，**海上洲岛自相惊。
> 日斜漾漾生云色，岸密萧萧带雨青。
> 野宿参差随鹿啸，春船苇岸接楼船。
> 时因弟姪逢三岛，共忆东隣一片经。
> 岚涧寺长僧拾佛，金峰像卷望成经。
> 渔阳有路喧呼唤，渔饮何由解姓铃。
> 大盗既穿宾榻弱，相期不管过漳清。
> 吾君聚月方幽赏，君子无为计与并。
> 自笑才高兴天子，且终朝谒上公卿。

#### 第一种模式：写藏头诗

哈哈哈，这个是撩妹专用

> 浩歌夜坐生光塘，然余坏竹入袁墙。
>
> 最爱林泉多往事，喜逢日月共流光。
>
> 欢讴未暇听雷响，芷壑已惊蛛雁忙。
>
> 若无一年离世曰，宝莲山中有仙郎。

可以看出，神经网络基本学习到了一定的韵律，在意境把握上也比较准确。



### 前期准备

- core i7 的笔记本
- 一个 GTX 1080ti 的显卡
- 装上pytorch的cpu和GPU版本



### 项目目的

使用pytorch实现RNN作诗，可以支持**首句续写**和**藏头诗**两个模式，并实现基本的音律和意境。



### 数据集

整理好的numpy格式数据集，

http://pytorch-1252820389.cosbj.myqcloud.com/tang_199.pth

其中包含唐诗57580首*125字，不足和多余125字的都被补充或者截断。



### 网络定义

```python
class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        # 词向量层，词表大小 * 向量维度
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 网络主要结构
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=Config.num_layers)
        # 进行分类
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        #print(input.shape)
        if hidden is None:
            h_0 = input.data.new(Config.num_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(Config.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # 输入 序列长度 * batch(每个汉字是一个数字下标)，
        # 输出 序列长度 * batch * 向量维度
        embeds = self.embeddings(input)
        # 输出hidden的大小： 序列长度 * batch * hidden_dim
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear(output.view(seq_len * batch_size, -1))
        return output, hidden
```



### 实现细节

1. data是numpy数组，57580首*125字
2. word2ix和ix2word都是字典类型，用于字符和序号的映射
3. nn.Embedding层可以输入为long Tensor型的字的下标（int），输入为同样shape的词向量，下标换成了向量，其余形状不变。最重要的构造参数是num_embeddings, embedding_dim
4. nn.LSTM主要构造参数input_size,hidden_size和num_layers，其中input_size其实就是词向量的维度，forward时输入为input和（h0,c0）,其中input为(seq_len,batch_size,input_size)，h0和c0是(num_layers $*$ num_directions, batch, hidden_size)，而forward的输出为output和(hn,cn)，一般后面一个就叫做hidden，output为(seq_len, batch, num_directions $*$ hidden_size)
5. 在本网络中，从前往后总共经历了这么几个网络，其向量变化如下：
    - input:(seq_len,batch_size)
    - 经过embedding层，embeddings(input)
    - embeds:(seq_len,batch_size,embedding_size)
    - 经过LSTM，lstm(embeds, (h_0, c_0))，输出output，hidden
    - output：(seq_len, batch, num_directions $*$ hidden_size)
    - output view为(seq_len $*$ batch, num_directions $*$ hidden_size)
    - 进过Linear层判别
    - output：(seq_len $*$ batch, vocab_size)
6. 具体训练时的实现方法：
    - 输入的input为(batch_size,seq_len)
    - data_ = data_.long().transpose(1,0).contiguous()将数据转置并且复制了一份，成了(seq_len,batch_size)
    - 通过input_,target = data_[:-1,:],data_[1:,:]将每句话分为前n-1个字作为真正的输入，后n-1个字作为label，size都是(seq_len-1,batch_size)
    - 经过网络，得出output：((seq_len-1) $*$ batch, vocab_size)
    - 通过target.view(-1)将target变成((seq_len-1) $*$ batch)
    - 这里的target不需要是一个one-hot向量，因crossEntropy不需要，直接是下标即可
    - 然后反向传播即可
7. 生成诗句的方法：
    - 首字为<START>，首个hidden自动为空
    - 如果有前缀风格，通过前缀生成hidden
    - 在首句内部时，不使用output，仅仅不断前进得到hidden，直到首句结尾。
    - 进入生成模式后，实际上每次调用model都生成一个字，逐渐生成前n句话。
    - 藏头诗同理，只是在头的部分将诗句首字换掉





### 使用方法

```python
python main.py
```

可以进行训练，result中含有训练过程数据。

```python
python test.py
```

可以测试生成效果。