## FastText

> fastText模型输入一个词的序列(一段文本或一句话)，输出这个词序列属于不同类别的概率
>
> 序列中的词和词组组成特征向量，特征向量通过线性变换映射到中间层，中间层再映射到标签
>
> fastText在预测标签时使用了非线性激活函数，但在中间层不使用非线性激活函数

### 分层分类器

为了改善运行时间，fastText模型使用了层次Softmax技巧，层次Softmax技巧建立在哈夫曼编码的基础上，对标签进行编码，能够极大地缩小模型预测目标的数量

https://blog.csdn.net/john_bh/article/details/79268850

https://blog.csdn.net/yangwohenmai1/article/details/96156497

### Softmax

$$
\LARGE softmax = \frac{e^{z_i}}{\sum_{j=1}^{k} {e^{z_j}}}
$$

softmax将多分类输出转换为概率，分为如下两步

* 分子：通过指数函数，将实数输出映射到零到正无穷
* 分母：将所有结果相加，进行归一化

> 假如模型对一个三分类问题的预测结果为-3、1.5、2.7。我们要用softmax将模型结果转为概率。步骤如下：
>
> 1）将预测结果转化为非负数
>
> y1 = exp(x1) = exp(-3) = 0.05
>
> y2 = exp(x2) = exp(1.5) = 4.48
>
> y3 = exp(x3) = exp(2.7) = 14.88
>
> 2）各种预测结果概率之和等于1
>
> z1 = y1/(y1+y2+y3) = 0.05/(0.05+4.48+14.88) = 0.0026
>
> z2 = y2/(y1+y2+y3) = 4.48/(0.05+4.48+14.88) = 0.2308
>
> z3 = y3/(y1+y2+y3) = 14.88/(0.05+4.48+14.88) = 0.7666
>


$$
\LARGE z_1 = x_1*w_{11}+x_2*w_{12}+x_3*w_{13}
$$

$$
\LARGE s_1 = \frac {e^{z_1}}{e^{z_1}+e^{z_2}+e^{z_3}}
$$

$$
\LARGE Loss_1 = -(y_1*lnS_1+y_2*lnS_2+y_3*lnS_3)
$$



#### 神经网络的输出X传递到损失函数Loss的步骤

* 输入层X与权重矩阵W相乘，得到输出z
* 输出z通过softmax函数转换成概率分布s
* 通过损失函数Loss，计算预测概率分布s和真实概率分布y之间的损失值loss



训练网络过程中的唯一的未知数是权重矩阵W，训练神经网络最后的结果结果就是生成一个或多个参数矩阵W，所以反向传播损失值本质上也就是为了修改权重矩阵W，W就是我们要求导的终极目标，即
$$
\frac {\partial{Loss}}{\partial{W}}
$$


### 链式求导


$$
\frac {\partial{Loss}}{\partial{W}} = \frac {\partial{Loss}}{\partial{s_i}}\cdot\frac {\partial{s_i}}{\partial{z_i}}\cdot\frac {\partial{z_i}}{\partial{w_i}}
$$

$$
Loss = -\sum{y_i}{log_eS_i}
$$

#### 对Loss求导

$$
\large \frac {\partial{Loss}}{\partial{S_i}} = -(y_1lnS_1+y_2lnS_2+...+y_ilnS_i)^{'}
$$

$$
\large \frac {\partial{Loss}}{\partial{S_i}} = -(y_1lnS_1)^{'}-(y_2lnS_2)^{'}-...-(y_ilnS_i)^{'}
$$

对Si求导，实质是对Si中的e^zi求导
$$
\large \frac {\partial{Loss}}{\partial{S_i}} = - \frac {y_1}{S_1}- \frac {y_2}{S_2}-...- \frac {y_i}{S_i}
$$

$$
\large \frac {\partial{Loss}}{\partial{S_i}} = -\sum_{j=1}^{n}y_i\frac {1}{S_j}
$$

#### softmax对输出层求导

完成如下求导
$$
\large \frac {\partial S_j}{\partial z_i} \\
\LARGE s_i = \frac {e^{z_i}}{e^{z_1}+e^{z_2}+...+e^{z_k}} \\
$$
根据导数特性，求导对象是Zi，因此Zi是未知数，Zj为常数，因此
$$
\begin{cases}
\LARGE (e^{z_i})^{'} = e^{z_i}\cdot(z_i)^{'} \\
\LARGE (e^{z_j})^{'} = 0
\end{cases}
$$
求导过程分为i=j，i<>j时两种情况

i = j时
$$
\large \frac {\partial S_j}{\partial z_i} = S_i(1 - S_i)
$$
i <> j时
$$
\large \frac {\partial S_j}{\partial z_i} = -S_jS_i
$$

#### 输出层对权重矩阵求导

$$
\LARGE \frac {\partial z_i}{\partial w_i} \\
\LARGE \frac {\partial z_1}{\partial w_{11}} = (x_1w_{11}+x_2w_{12}+x_3w_{13})^{'} \\
\LARGE \frac {\partial z_1}{\partial w_{11}} = x_1
$$

https://blog.csdn.net/yangwohenmai1/article/details/96741328



### 模型使用

#### 词表示模型

```python
"""
模型调用
"""
import fasttext
# Skipgram model :
model = fasttext.train_unsupervised('data.txt', model='skipgram')
# or, cbow model :
model = fasttext.train_unsupervised('data.txt', model='cbow')

"""
模型内容
"""
print(model.words)   # list of words in dictionary
print(model['king']) # get the vector of the word 'king'

"""
模型保存和加载
"""
#保存
model.save_model("model_filename.bin")
#加载
model = fasttext.load_model("model_filename.bin")
```

#### 文本分类模型

```python
"""
data.train.txt文件为模型训练数据，每行数据为一个样本，标签内容默认以 __label__为前缀
"""
model = fasttext.train_supervised('data.train.txt')
"""
模型内容
"""
print(model.words)
print(model.labels)
"""
模型评估
model.test('test.txt')输出结果为一个含有3个元素tuple，依次为测试数据量，准确率，召回率
"""
def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
print_results(*model.test('test.txt'))

"""
模型样本预测，k为可选参数，k为输出的标签类别个数
预测多样本，可将样本组合为数组[t1,t2]
"""
model.predict(l_text, k=2)
# output  (('__label__0', '__label__1'), array([0.98421693, 0.01580307]))
model.predict(l_text)
# (('__label__0',), array([0.98421693]))

"""
模型量化压缩，尽量保持精度的条件下减小模型
"""
# with the previously trained `model` object, call :
model.quantize(input='data.train.txt', retrain=True)

# then display results and save the new model :
print_results(*model.test(valid_data))
model.save_model("model_filename.ftz")


#有监督模型参数
    input             # training file path (required)
    lr                # learning rate [0.1]
    dim               # size of word vectors [100]
    ws                # size of the context window [5]
    epoch             # number of epochs [5]
    minCount          # minimal number of word occurences [1]
    minCountLabel     # minimal number of label occurences [1]
    minn              # min length of char ngram [0]
    maxn              # max length of char ngram [0]
    neg               # number of negatives sampled [5]
    wordNgrams        # max length of word ngram [1]
    loss              # loss function {ns, hs, softmax, ova} [softmax]
    bucket            # number of buckets [2000000]
    thread            # number of threads [number of cpus]
    lrUpdateRate      # change the rate of updates for the learning rate [100]
    t                 # sampling threshold [0.0001]
    label             # label prefix ['__label__']
    verbose           # verbose [2]
    pretrainedVectors # pretrained word vectors (.vec file) for supervised learning []
#模型方法
    get_dimension           # Get the dimension (size) of a lookup vector (hidden layer).
                            # This is equivalent to `dim` property.
    get_input_vector        # Given an index, get the corresponding vector of the Input Matrix.
    get_input_matrix        # Get a copy of the full input matrix of a Model.
    get_labels              # Get the entire list of labels of the dictionary
                            # This is equivalent to `labels` property.
    get_line                # Split a line of text into words and labels.
    get_output_matrix       # Get a copy of the full output matrix of a Model.
    get_sentence_vector     # Given a string, get a single vector represenation. This function
                            # assumes to be given a single line of text. We split words on
                            # whitespace (space, newline, tab, vertical tab) and the control
                            # characters carriage return, formfeed and the null character.
    get_subword_id          # Given a subword, return the index (within input matrix) it hashes to.
    get_subwords            # Given a word, get the subwords and their indicies.
    get_word_id             # Given a word, get the word id within the dictionary.
    get_word_vector         # Get the vector representation of word.
    get_words               # Get the entire list of words of the dictionary
                            # This is equivalent to `words` property.
    is_quantized            # whether the model has been quantized
    predict                 # Given a string, get a list of labels and a list of corresponding probabilities.
    quantize                # Quantize the model reducing the size of the model and it's memory footprint.
    save_model              # Save the model to the given path
    test                    # Evaluate supervised model using file given by path
    test_label              # Return the precision and recall score for each label.    
```

