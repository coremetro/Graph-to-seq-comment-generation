

该模型包含三个部分，
1 graph construction
将一篇文章构建成一个图，首先通过textrank的方法（也可以用别的方法）提取出关键字（concept）和对应的句子集合（content）
每个concept对应一个content，一般来说有13个concept，其中又一个title，还有一个''表明不属于其他concept的句子
这样每个concept是一个节点，重点是节点之间邻接矩阵adj，这个会在后面的计算中使用，这个部分应该是graph模块，但是具体代码我没有看

2  encoder
论文中称其为vetex encoder,先考虑模块data，用来构建训练集和验证集。
2.1 data 模块，简单来说， batch是一个list，里面的元素是example(代表一篇文章，有几个属性，concept,content,tgt(评论）等等。
2.2 models模块，定义了各种模型，其中graph2gru就是将lstm改为gru的模型,graph2gru_noAtten是去掉global Attention机制的模型，还有论文中
的graph2seq模型，以上几个模型，他们的encoder都是一样的。
encoder实际上是memory_network, menory_network分为两个部分，第一个部分是bert(word_level_model),第二部分是gcn，在gcn中会用到adj(邻接矩阵）
先说memory_network, bert 是有两个multi-attention+前馈模型组成，每个multi-attention有4个head，然后bert的输出和adj一起传入gcn模型中
训练的时候，一个batch是32维，每一维是一篇文章，有13个concepts，每个concept对应一个content，每个content100个单词，
将contents和concepts通过word embedding，将每个单词（整数，vocab中对应的数字）映射成128维，这样contents就是[13,100,128]
concepts就是[13,1,128]，通过positional embeding给每个单词一个位置权重，然后把concept放在对应的content前面，这样就变成了[13,101,128]
将每篇文章输入到bert中,bert是两个multi-head attention+feed forward模型，比较复杂，具体看着篇文章https://arxiv.org/pdf/1906.01231.pdf
也可以看在graph2gru.py和bert.py中的注释。最终bert的输出还是[13,101,128]，然后只取每个content的第一个单词，称其为x，变成了[13,128]，传入gcn模型中，
将x和adj做稀疏矩阵相乘等操作，最后得到encoder的输出，有两部分,contexts是全部的输出，state是题目（title）的输出。

3. decoder
 将encoder的输出和decoder的输入（文章的评论）一起传入decoder中，decoder是一个stackedrnn，两层的lstm（gru），最后会输出每个单词的概率。
这样就能生成一个句子，最大长度是30.在测试中，没有评论，decoder的输入是<EOS>表明开始。可以看看每个模型类的sample方法

使用方法
python train.py -model graph2gru # 改为gru的模型
python train.py -model graph2gru_noAtten  #gru无attention模型

