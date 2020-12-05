### <font color="Purple">LEARNING TO READ AND FOLLOW MUSIC IN COMPLETE SCORE SHEET IMAGES  </font>

> 原文链接  https://arxiv.org/pdf/2007.10736.pdf

> **Abstract**：本文主要解决对未经处理的乐谱图片进行 **score following** 的工作。不同于依赖于**OMR**获得计算机能够读取的乐谱表示（score representation），本文能够在未经处理的整张乐谱图片上进行score following。

> **Related Work**：主要可将**<u>score following</u>**分类为基于计算机读取的乐谱表示（如MusicXML或MIDI），和不需要任何符号表示的方法。
>
> 1. 基于计算机读取的乐谱表示：此类方法使用<u>DTW</u>（dynamic time warping）和HMM（Hidden Markov Models）算法实现准确可靠的追踪结果。缺点是需要机器可识别的乐谱表示，通常这些乐谱表示需要人工制作，耗时费力；或者使用OMR（Optical Music Recognition）自动提取音符，但OMR带来的误差会很大程度上影响跟踪结果。“<u>**Score Following as a Multi-Modal Reinforcement Learning Problem**</u>  ”中也提到，依赖OMR系统提取MIDI乐谱的基于DTW的score following系统在合成的测试集上很难跟踪。
> 2. “**<u>Towards Score Following in Sheet Music Images</u>** ”中提出了多模型深度神经网络预测基于简短音频摘录的乐谱片段的位置。除此之外，“<u>**Learning to Listen, Read, and Follow: Score following as a Reinforcement Learning Game**</u>  ”和“<u>**Score Following as a Multi-Modal Reinforcement Learning Problem**</u> ”还将score following视为强化学习（RL）问题，RL代理的任务是在音乐摘录中调整自己的阅读对乐谱的速度。这种方法需要将乐谱表示为**展开的形式**，五线谱需要在乐谱图片上检测出来，切割出来给到score following系统。

> **本文的方法**
>
> 1. 将乐谱跟踪视作图片分割任务（Image Segmentation Task）：我们想定位输入音频对应乐谱中的位置，我们把音频视为语言输入，乐谱图片视为对应实体。**根据直到当前时间点传入的音乐演奏内容，该模型的任务是针对当前演奏的音乐相对应的给定乐谱预测分割蒙版**——给定乐谱（sheet image）和输入音频，音频将一帧一帧输入到tracker中，追踪系统处理最近40帧（last 40 frames）以预测当前最后一帧（last frame）（Target Frame）在乐谱当前的位置。这个系统针对当前audio对每个像素点预测出一个概率值，最符合当前输入音频的乐谱位置将会高亮。理想情况是乐谱中只有一个位置高亮。
> 2. 挑战：如何处理乐谱图片？——使用条件机制，直接调节处理乐谱图片的特征检测器的活动。我们在循环层顶部使用调节机制处理（提供更长时域信息）任意分辨率的乐谱图片。

<img src="F:\YWM_work\Github_projs\score-following\figs\score.png" style="zoom: 80%;" />

> 3.Feature-wise 线性调制：使用此层的目的是通过调制其特征图来直接干扰所学习的图像表示，从而帮助卷积神经网络仅关注正确分割所需的那些部分。![](F:\YWM_work\Github_projs\score-following\figs\film.png)
>
> 4.模型结构：基于U-net结构（“Towards Full-Pipeline Handwritten OMR with Musical Symbol Detection by U-Nets  ”）可以适应score following任务中将乐谱分割为对应当前audio input的区域，基于前人经验，我们考虑在（**B-H**块）结合从audio input提取conditioning information，仅留下**A和I块**不包含conditioning。由于训练中循环层的出现，我们使用layer Normalization 代替了batch Normalization。![](F:\YWM_work\Github_projs\score-following\figs\audio-encoder.png)
>
> ​	为得到audio input 的 conditioning information，我们测试了两种不同语谱图encoder（上图为encoder结构），第一个encoder使用长度为40 frame 的频谱图片段（对应 2 秒钟音频）与 <font color='#00000'> “Learning Audio-Sheet Music Correspondences for Cross-Modal Retrieval and Piece Identification” </font> 中使用频谱图相似；另一个encoder 使用单帧语谱图作为输入，后接32个单元的 **dense layer**，layer Normalization和ELU激活函数。encoders的输出都会馈入到**LSTM**中,之后LSTM的隐藏层状态作为FiLM层的external输入。![](F:\YWM_work\Github_projs\score-following\figs\Unet.png)如图是Audio-Conditioned U-Net结构，图中每一个块（A-I）都包含两个以ELU为激活函数，并进行layer Normalization的卷积层。**FiLM层**被放在最后一个激活函数的之前，audio 语谱图encoder的输出被用到给FiLM层做调整，途中每一个对称块都有相同数量的滤波器，A有8个滤波器，依次增加到E有128个滤波器。

> **实验**
>
> 1、数据集：**MSMD**（Multi-modal Sheet Music Dataset），包括巴赫、莫扎特和贝多芬的复音钢琴音乐。乐谱文件都是由Lilypond为标准。文中只把单独一张乐谱进行预测，如果一首曲子包含几张谱子，那么会把每张谱子看做一首曲子，同时将原始MIDI信息响应分割。因此我们将原来353首训练集，19首验证集和94首测试集分成945首训练和28首验证集，125首测试集。乐谱图片的分辨率为1181×835 pixels，下采样到393×278 pixels，以此作为U-Net的输入（这样不仅不会影响模型的表现，同时还能加速训练速度）。
>
> 	Ground Truth：使用”Learning Audio–Sheet Music Correspondences for Cross-Modal Retrieval and Piece Identification”文中的音符符头对齐,乐谱图片中符头以（x,y）坐标形式对齐。(本文作者对修正y坐标为对应其标注的音符的中心，以便进行实验。)作者对每个坐标制作了二元 mask，宽度为10像素，高度依赖于实际音符的高度。
>
> 2、现在，U-Net的任务是在给出乐谱图像以及从音频输入获得的条件信息的情况下，推断出一个分割蒙版。理论上，网络应该直接预测出 x 和 y 坐标，但这是一个更困难的任务，而本文还未能实现。
>
> 3、数据处理：
>
> - audio sample：22050 Hz
> - frame rate：20 fps
> - DFT窗长：2048 个采样点
> - semi-logarithmic filterbank：60 Hz - 6kHz, 一共78个log-frequency 频点
> - 每个频点标准化为平均值为0，方差为1

