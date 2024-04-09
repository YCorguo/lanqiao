# AI Note

## Natural Language Processing

### overview of the development timeline of large language models
#### Early Stages: Rule-based and Statistical Models
- **1950s-1980s**: primarily rule-based. relied on expert systems and hardcoded linguistic rules to parse text, with limited effectiveness.
- **1980s-1990s**: Statistical models, notably n-gram models. but still limited to relatively small datasets and simpler tasks.
An n-gram is a contiguous sequence of n items from a given sample of text or speech. The items can be phonemes, syllables, letters, words, or base pairs according to the application. N-grams are used for a variety of purposes in statistical natural language processing and genetic sequence analysis. They serve as the foundation for models that represent the probability of a given sequence of elements.
##### N-gram model
N-gram essentially refers to a set of co-occurring words or characters within a given text. For example:
- A 1-gram (or unigram) is a single unit.
- A 2-gram (or bigram) is a sequence of two units.
- Similarly, a 3-gram (or trigram) consists of three units, and so on.

An n-gram model predicts the probability of a word given the previous \(n-1\) words, making it a type of Markov model that assumes the probability of a word depends only on the previous \(n-1\) words. This simplification makes the model tractable and computationally efficient, but it also means that accuracy may decrease with increasing \(n\), as the specific sequence of \(n-1\) words becomes rare or unseen in the training data.

###### Limitations
- **Sparsity**: As \(n\) increases, the frequency of encountering the exact sequence in the training corpus decreases, leading to sparse data issues.
- **Storage**: Larger \(n\)-grams require significantly more memory and storage, as the number of possible combinations increases exponentially.
- **Generalization**: High-order n-grams may overfit the training data, leading to poor generalization on unseen text.

###### Smoothing Techniques
To address the sparsity problem, smoothing techniques like Laplace smoothing (add-one smoothing) and interpolation are used. These methods adjust the probability distribution to account for unseen n-grams, making the model more robust and less likely to assign zero probability to unseen events.

1. Laplace Smoothing in N-gram model
The basic idea behind Laplace smoothing is to assume that every n-gram has been seen at least once before, by adding a small positive value (typically 1) to the count of every n-gram in the corpus, including those not present in the training set.

    1. How Laplace Smoothing Works
Given an n-gram model, the probability of the next word \(w_n\) given the previous \(n-1\) words (context \(w_{1}^{n-1}\)) can be computed as:
\[ P(w_n | w_{1}^{n-1}) = \frac{C(w_{1}^{n})}{C(w_{1}^{n-1})} \]
where \(C(w_{1}^{n})\) is the count of the n-gram \(w_{1}^{n}\) in the training corpus, and \(C(w_{1}^{n-1})\) is the count of the \(n-1\) words occurring together in the training corpus. Without smoothing, if an n-gram \(w_{1}^{n}\) was not present in the training data, \(C(w_{1}^{n}) = 0\), leading to \(P(w_n | w_{1}^{n-1}) = 0\).
Laplace smoothing adjusts this formula by adding one to the count of all n-grams: \[ P_{\text{Laplace}}(w_n | w_{1}^{n-1}) = \frac{C(w_{1}^{n}) + 1}{C(w_{1}^{n-1}) + V} \] where \(V\) is the number of unique words in the vocabulary. This adjustment ensures that no n-gram has a zero probability.

#### Neural Networks and Word Embeddings
- **Early 2000s**: neural networks. moving away from reliance on manually designed features.
- **2013**: Word2Vec marked the rise of word embedding techniques.
#### Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM)
- **Early 2010s**: RNN and LSTMs, began to be widely used for processing sequential data, including text. These models were capable of capturing long-distance dependencies within sequences, though training them was still challenging due to issues like vanishing and exploding gradients.
#### Attention Mechanism and Transformers
- **2014**: Seq2Seq models and the attention mechanism greatly improved performance on tasks like machine translation and text generation. The attention mechanism allows models to "focus" on different parts of the input sequence while generating each word.
- **2017**: "Attention is All You Need" introduced Transformer model, which relies entirely on attention mechanisms, abandoning traditional RNN architectures. The Transformer model's efficient parallel computation and superior performance laid the groundwork for subsequent large language models.
#### Rise of Pre-trained Models: BERT and GPT
- **2018**: BERT (Bidirectional Encoder Representations from Transformers). It can understand the context of a word in different uses, significantly improving performance on language understanding tasks.
- **2018**: OpenAI released the GPT (Generative Pre-trained Transformer) model, utilizing the Transformer architecture for large-scale text data pre-training, followed by fine-tuning on specific tasks. The GPT model achieved breakthrough progress in various language generation tasks.
##### Differences between RNN, BERT and GPT
###### RNN (Recurrent Neural Network)
- **Mechanism**: RNNs process sequences by maintaining a 'memory' (hidden state) of previous inputs using a loop mechanism. This allows them to transfer information from one step of the sequence to the next.
- **Formula**:
  \[ h_t = f(W_{hh}h_{t-1} + W_{xh}x_t) \]
  Where \(h_t\) is the hidden state at time \(t\), \(x_t\) is the input at time \(t\), \(W_{hh}\) and \(W_{xh}\) are weights, and \(f\) is a non-linear activation function like tanh or ReLU.
- **Limitation**: RNNs struggle with long-term dependencies due to vanishing and exploding gradient problems.

###### BERT (Bidirectional Encoder Representations from Transformers)
- **Mechanism**: BERT uses the Transformer architecture, specifically focusing on the encoder part. It processes entire input sequences simultaneously, allowing it to capture bidirectional contexts by using self-attention mechanisms. BERT is pre-trained on large corpora using tasks like Masked Language Model (MLM) and Next Sentence Prediction (NSP).
- **Formula**:
  The core of BERT's mechanism is the self-attention calculation:
  \[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]
  Where \(Q\), \(K\), and \(V\) are queries, keys, and values matrices derived from input embeddings, \(d_k\) is the dimensionality of the keys, and softmax provides a probability distribution used to weight the values.
- **Distinctive Feature**: BERT's bidirectional training is more comprehensive than traditional left-to-right or right-to-left models, allowing it to understand the context better.

###### GPT (Generative Pre-trained Transformer)
- **Mechanism**: GPT also uses the Transformer architecture but focuses on the decoder. It's trained on a predictive task to generate the next word in a sentence. GPT processes text in a unidirectional (left-to-right) manner but can be adapted for bidirectional contexts in subsequent versions (e.g., GPT-3).
- **Formula**:
  Similar to BERT, GPT relies on the self-attention mechanism, but its training objective and structure are geared towards generation:
  \[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]
  The difference lies in the application of this formula; GPT uses it in a generative context to predict the next token.
- **Distinctive Feature**: GPT is designed for generative tasks and is trained using a causal (autoregressive) language modeling task, predicting each word based on the previous words in a sentence.

##### BERT-Based variants
BERT (Bidirectional Encoder Representations from Transformers) has inspired a multitude of variants, each aiming to enhance or adapt the original model to specific needs. Here's a comparative overview of some prominent BERT variants:

###### 1. RoBERTa (Robustly Optimized BERT Approach)
- **Improvements**: RoBERTa modifies the pre-training procedure of BERT by optimizing hyperparameters, removing the next sentence prediction objective, and training with much larger mini-batches and data.
- **Performance**: Outperforms BERT on many NLP benchmarks by leveraging longer training times and more data.

###### 2. ALBERT (A Lite BERT)
- **Improvements**: Introduces two parameter-reduction techniques to lower memory consumption and increase training speed. ALBERT replaces the next sentence prediction with a sentence-order prediction task and shares parameters across layers.
- **Performance**: Achieves comparable or even superior results to BERT with significantly fewer parameters.

###### 3. DistilBERT (Distilled BERT)
- **Improvements**: Utilizes knowledge distillation during training, where a smaller model (the "student") is trained to reproduce the behavior of a larger model (the "teacher"). 
- **Performance**: Retains 97% of BERT's performance on language understanding benchmarks while being 40% smaller and 60% faster.

###### 4. ERNIE (Enhanced Representation through kNowledge Integration)
- **Improvements**: Developed by Baidu, ERNIE integrates world knowledge into pre-training, processing named entities and phrases as whole units for training.
- **Performance**: Demonstrates superior performance on various Chinese NLP tasks by incorporating structured knowledge.

###### 5. SpanBERT
- **Improvements**: Enhances BERT by pre-training on span selections and predicting the entire content of the spans, rather than predicting words in isolation.
- **Performance**: Shows improvements on span selection tasks and coreference resolution by focusing on predicting spans of text.

###### 6. TinyBERT
- **Improvements**: Focuses on compressing BERT to make it suitable for deploying on devices with limited computational capacity. It employs a two-stage learning process including transformer distillation and data augmentation.
- **Performance**: Despite its smaller size, TinyBERT achieves performance close to its full-sized counterpart on general language understanding tasks.

###### Comparative Analysis
- **Size and Efficiency**: ALBERT and TinyBERT significantly reduce model size and computational requirements, making BERT more accessible for resource-constrained environments.
- **Performance**: RoBERTa and SpanBERT focus on enhancing the model's understanding of context and relationships within text, showing notable performance improvements on several benchmarks.
- **Specialization**: ERNIE incorporates external knowledge into pre-training, providing advantages in tasks that benefit from such information, especially in domain-specific applications.

##### GPT History (Generative Pre-trained Transformer)
###### GPT (Generative Pre-trained Transformer)
- **Release Date**: 2018
- **Features**: The first version introduced by OpenAI, GPT utilizes the Transformer architecture's decoder component and follows a pre-training plus fine-tuning paradigm. With 117 million parameters, GPT was pre-trained in an unsupervised manner on a large dataset and then fine-tuned for specific tasks.
- **Applications**: Even as an early model, GPT demonstrated potential across a range of NLP tasks including text generation, translation, and summarization.

###### GPT-2
- **Release Date**: 2019
- **Features**: GPT-2 expanded the scale of GPT to 1.5 billion parameters. It was pre-trained on a larger dataset, showcasing more refined text generation capabilities, including better coherence and understanding of text.
- **Applications**: GPT-2 made breakthroughs in creative writing, news article generation, and dialogue systems, among other areas.

###### GPT-3
- **Release Date**: 2020
- **Features**: GPT-3 further enlarged the model to 175 billion parameters, making it one of the largest language models of its time. Its innovation lies in its few-shot learning ability, where it can perform a variety of NLP tasks with little to no task-specific data, using just a few examples.
- **Applications**: The applications of GPT-3 are exceedingly broad, ranging from generating programming code and automatic text summarization to language translation and advanced dialogue systems.

###### Instruct GPT
- **Release Date**: 2021
- **Features**: Instruct GPT (also referred to as GPT-3.5) is an optimized version of GPT-3 that further learns from human feedback. It better understands user instructions, generating outputs more aligned with human intentions.
- **Applications**: Instruct GPT shows significant improvements in providing more accurate information, writing high-quality texts, and answering complex questions.

In natural language processing (NLP), the concepts of encoder and decoder are fundamental to understanding many modern architectures, especially those involved in tasks like machine translation, text summarization, and question answering. These components are central to the design of sequence-to-sequence (seq2seq) models, which are used to convert sequences from one domain (e.g., sentences in English) into sequences in another domain (e.g., sentences in French).

#### Encoder & Decoder
###### Encoder

The encoder's role is to process the input sequence and compress all information into a context vector (or a set of vectors in more complex models like Transformers). This context vector aims to capture the essence of the input sequence's information, serving as a comprehensive representation for the decoder to generate the output.

- **Functionality**: In a typical seq2seq model, the encoder processes the input sequence word by word (or token by token) and sequentially updates its internal state. In RNN-based architectures, this involves updating the hidden state at each timestep. The final state of the encoder after the last word of the input sequence has been processed is used as the context vector.
- **Types of Encoders**:
  - **RNN-based Encoders**: Utilize recurrent neural networks, LSTM (Long Short-Term Memory), or GRU (Gated Recurrent Units) to handle sequential data.
  - **Transformer Encoders**: Use self-attention mechanisms to weigh the importance of different words in the input sequence relative to each other, capturing contextual relationships without the sequential processing limitations of RNNs.

###### Decoder

The decoder's task is to take the context vector produced by the encoder and generate the output sequence from it, one token at a time. The decoder learns to generate the output sequence by being trained on the target sequence, effectively learning to translate the context into a new domain.

- **Functionality**: Starting with the context vector, the decoder generates the output sequence token by token. At each step, it considers the context vector (and possibly its previous outputs) to generate the next token. In RNN-based decoders, the generated token is fed back into the model as input for generating the next token.
- **Types of Decoders**:
  - **RNN-based Decoders**: Similar to RNN-based encoders but focused on generating the sequence rather than encoding it. They might also incorporate attention mechanisms to focus on different parts of the input sequence during each step of the generation.
  - **Transformer Decoders**: Leverage self-attention and cross-attention mechanisms, where the latter allows the decoder to focus on different parts of the input sequence as it generates each token of the output sequence.

###### Attention Mechanism
An important enhancement to the encoder-decoder architecture is the attention mechanism, which allows the decoder to focus on different parts of the input sequence during the decoding process. This is particularly useful for longer sequences, where the context vector alone might not be sufficient to capture all the necessary information.
- **How It Works**: The attention mechanism computes a set of attention weights that determine how much focus to put on each part of the input sequence when generating each token of the output sequence. This allows the model to dynamically attend to the most relevant parts of the input as needed, rather than relying solely on the fixed context vector.

##### T5
The T5 (Text-to-Text Transfer Transformer) model, introduced by Google Research in 2019, represents a significant shift in natural language processing (NLP) by framing all NLP tasks as a unified text-to-text problem. This innovative approach enables T5 to handle a wide range of tasks with a single model architecture, from translation and summarization to question answering and classification, by simply changing the input format.

###### Key Features of T5
- **Unified Task Framework**: T5 treats every NLP task as a text-to-text conversion, where both the input and output are sequences of text. This simplifies processing across diverse tasks.
- **Pre-training Objectives**: It uses a pre-training objective called "span corruption," where random contiguous spans of text are masked and the model is trained to predict these masked spans. This is somewhat akin to BERT's masked language model task but operates on spans of text.
- **Scalability**: T5 comes in multiple sizes (Small, Base, Large, 3B, 11B) to cater to different computational resource requirements and applications.

###### Similar Large Language Models

Following T5, several other large language models have been developed, each with unique characteristics but sharing the goal of improving NLP capabilities:

- BERT (Bidirectional Encoder Representations from Transformers)
- GPT Series (Generative Pre-trained Transformer)
- RoBERTa (Robustly Optimized BERT Approach)
- BART (Bidirectional and Auto-Regressive Transformers)
    - Combining the best of BERT and GPT, BART is designed for tasks that require both understanding and generation, like text summarization and translation. It uses a denoising autoencoder setup, where the model learns to reconstruct corrupted text.

###### Comparison and Evolution
- **Task Versatility**: T5's unified text-to-text approach offers a versatile framework for handling a wide range of NLP tasks, a concept that has influenced subsequent models and research.
- **Pre-training Techniques**: While BERT and its optimizations focus on understanding, GPT on generation, and BART combines aspects of both, T5's span corruption pre-training unifies these approaches under a single framework.
- **Model Scalability and Efficiency**: The development of these models has also focused on making them scalable and efficient, with variations like DistilBERT, TinyBERT, and MiniLM aiming to maintain high performance with reduced model sizes for broader application.


#### Scaling Model Sizes
- **2019 to present**: With increased computational power and larger datasets, the size of language models continued to expand. Models like GPT-2 and GPT-3, with billions to hundreds of billions of parameters, further improved the quality and diversity of generated text.



## Reinforcement Learning

### environment of RL

$next\_state = P(\cdot|recent\_state, action\_of\_agent)$

### multi-armed bandit problem

#### problem description
There is one K-armed bandit. Pulling each lever corresponds to a probability distribution of rewards. We start from scratch with the unknown probability distribution of rewards for each lever, aiming to obtain the highest possible cumulative reward after operating $T$ times.

#### format desctiption
$states: <A, R>$
$A$ is the action set(multiset), and the action space is $\{a_1, \ldots, a_K\}$
$R$ each level corresponds to one $R_i(r|a)$
$target:\max\sum_{t=1}^Tr_t,\ s.t.\ r_t\thicksim R(\cdot|a_t)$

#### cumulative regret
For each action $a$, define expected reward as $Q(a)=\Bbb{E}_{r\thicksim R(\cdot|a)}[r]$. Therefore, $\exist Q^* = \max_{a\isin A}Q(a)$. Intuitively, the regret of the action $R(a) = Q^* - Q(a)$, and cumulative regret, for next complete T steps, is $\sigma_R = \sum_{t=1}^TR(a_t)$.

Further, the inference below allow us to dynamically renew expected rewards:
$$\begin{split}Q_k&=\frac{1}{k}\sum\limits_{i=1}^{k}r_i\\
      &=\frac{1}{k}\sum\limits_{i=1}^{k-1}r_i+\frac{r_k}{k}\\
      &=\frac{k-1}{k}(\frac{1}{k-1}\sum\limits_{i=1}^{k-1}r_i)+\frac{r_k}{k}\\
      &=\frac{k-1}{k}Q_{k-1}+\frac{r_k}{k}\\
      &=Q_{k-1}-\frac{1}{k}Q_{k-1}+\frac{r_k}{k}\\
      &=Q_{k-1}+\frac{r_k-Q_{k-1}}{k}\\
\end{split}$$

For each lever, only if we use a counter $N(a)$, updates of $\^Q(a_t)$ could be descripted as:
- for $\forall a \isin A,\ init.\ N(a) = \^Q(a) = 0$
- for $t = 1 \to T$ do
    - choose lever $a_t$
    - get $r_t$
    - update counter: $N(a_t) = N(a_t) + 1$
    - update expected rewards: $\^Q(a_t)=\^Q(a_t)+\frac{1}{N(a_t)}[r_t-\^Q(a_t)]$
    - end for

#### $\epsilon$-Greedy algo.

Optimize the lever choosing strategy, balancing exploration and exploitation. The choosing strategy is:
$$a_t = \left\{\begin{aligned}\arg \max_{a\isin A}\^Q(a), with\ prob.\ 1-\epsilon\\random\ sample\ from\ A, with\ prob.\ \epsilon\\\end{aligned}\right.$$

If set $\epsilon$ as a constant, the cumulative regret will linearly increase. But if set $\epsilon = \frac{1}{t}$, it will be sublinear and obviously better than constant form.

#### Upper Confidence Bound algo.
##### Hoeffding's inequality
Given n i.i.d random variables $X_1, X_2, \ldots, X_n$, with a range of $[0, 1]$, experience expectations is $\overline x_n = \frac{1}{n}\sum_{j=1}^nX_j$, then$$\Bbb{P}\{\Bbb E[X]\ge\overline x_n+u\}\le e^{-2nu^2}$$

##### UCB in MAB for each lever
let $\overline x_t = \^Q_t(a), u = \^U_t(a), p = e^{-2N_t(a)U_t(a)^2}$, then $$\Bbb{P}\{Q_t(a)\ge\^Q_t(a)+\^U_t(a)\}\le e^{-2n\^U^2_t(a)} = p$$
$$1-\Bbb{P}\{Q_t(a)\ge\^Q_t(a)+\^U_t(a)\}\ge 1-p$$
$$\Bbb{P}\{Q_t(a)<\^Q_t(a)+\^U_t(a)\}\ge 1- p$$
when $N_t$ increases, $p$ is decreases. so $Q_t(a) = \^Q_t(a)+\^U_t(a)$, and $\^Q_t(a)+\^U_t(a)$ is expected reward upper bound. Now, we can choose the action with the reward expectation with largest upper bound.

Using $\epsilon$-greedy algo., we could set $p = \epsilon = \frac{1}{t}$, and because $p = e^{-2N_t(a)U_t(a)^2}$, get $\^U_t(a) = \sqrt{\frac{-\log p}{2N_t(a)}}$, and of course, for robustness, $\^U_t(a) = \sqrt{\frac{-\log p}{2(N_t(a)+1)}}$.

At last, we could set a coefficent $c$ to control the weight of uncertainty: $$a = \arg \max _{a \isin A} [\^Q_t(a)+c \cdot \^U_t(a)]$$

#### Thompson Sampling
1. assume that each lever corresponds to one specific distribution.
2. sample on each lever to estimate the specific distribution.
3. choose the action of largest reward.

### Markov decision process
#### Stochastic Process
Stochastic Process describe the variance of random variables according to time. 
#### Markov Property
If one stochastic process is with Markov property, the state of time t+1 depends merely on the state of time t. Formatly:$$P(S_{t+1}|S_t) = P(S_{t+1}|S_1, \ldots, S_t)$$

### two types of RL

#### model-based reinforcement learning

#### model-free reinforcement learning

### classic RL algorithms

#### DQN algo.

#### PPO algo.

##### optimization target of TRPO algo.

$$\max\limits_\theta \Bbb{E}_{s\thicksim\nu^{\pi_{\theta_k}}}\Bbb{E}_{a\thicksim\pi_{\theta_k}}(\cdot|s)[\frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)}A^{\pi_{\theta_k}}(s,a)]\\
s.t.\ \Bbb{E}_{s\thicksim\nu^{\pi_{\theta_k}}}[D_{KL}(\pi_{\theta_k}(\cdot|s), \pi_\theta(\cdot|s))]\le\delta
$$

## Fundamental

### Cross Validation
1. Simple Cross Validation.
Simply split dataset as train set and test set.
2. K-fold Cross Validation
    a. split
    b. enumerate each subset as validset.
    c. use mean value of K times.
3. Leave-one-out Cross Validation
    a. for N samples, use N-1 to train, 1 for evaluate.
4. Usually, if the size is large, simply split it into 10 or 20 parts, else use Sturge's Rule:$$Number\ of\ Bins=1+log_2(N)$$

### Measurements
$$Accuracy = \frac{TP+TN}{TP+TN+FP+FN}$$
$$Precision = \frac{TP}{TP+FP}$$
$$Recall = TPR = \frac{TP}{TP+FN}$$
$$FPR = \frac{FP}{TN+FP}$$
$$F1 = \frac{2*Precision*Recall}{Precision+Recall}$$
$$ROC\ Curve = \left\{
\begin{aligned}
x: FPR \\
y: TPR \\
\end{aligned}
\right.
$$
$$LogLoss = -1.0\times(target\times log(prediction)+(1-target)\times log(1-prediction))$$
$$Macro\ averaged\ precision = \frac{\sum\limits_i Precision_i\times N_i}{\sum\limits_i N_i}$$
$$Micro\ averaged\ precision = \frac{Precision_i}{N_i}$$
$$Weighted\ averaged\ precision = \frac{\sum\limits_i Precision_i\times N_i\times w_i}{\sum\limits_i N_i}$$
$$Confusion\ Matrix:
\begin{matrix} 
\ & class-1 & class-0 \\
class-1 & xx & xx \\
class-0 & xx & xx \\
\end{matrix}$$
$$Error=True Value−Predicted Value$$
$$Absolute Error=Abs(True Value−Predicted Value)$$
$$Squared Error=(TrueValue−Predicted Value)^2$$
$$RMSE=SQRT(MSE)$$
$$Percentage Error=\frac{True Value–Predicted Value}{True Value}\times100$$
$$Coefficient\ of\ determination = R^2 = \frac{\sum_{i=1}^N(y_{t_i}-y_{p_i})^2}{\sum_{i=1}^{N}(y_{t_i}-y_{t_{mean}})}$$
$$MCC=\frac{TP\times TN−FP\times FN}{\sqrt{(TP+FP)\times(FN+TN)\times(FP+TN)\times(TP+FN)}}$$
## references

[动手学强化学习](https://hrl.boyuai.com/)






















