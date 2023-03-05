# --------------------------------------------------------------------------------
# CSCI 6908 ProjectBuilding a QA system
# 
# Part I:   Layers used by the original BiDAF model (Baseline Model).
# Part II:  Layers used by the character-level embeddings.
# Part III: Layers used by Self-attention.
# Part IV:  Layers used by QANet(Combining Local Convolution with Global Self-Attention for Reading Comprehension).
#   
# Refer to : https://github.com/Oceanland-428/Improved-BiDAF-with-Self-Attention
#            https://github.com/umhan35/nlp-squad
#            https://github.com/jerrylzy/SQuAD-QANet
#            Https://github.com/andy840314/QANet-pytorch-
#            https://github.com/inSam/QA-XL
#            https://github.com/datpnguyen/QANet-CS224N
# --------------------------------------------------------------------------------

"""Assortment of layers for use in models.py.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import Parameter
from util import masked_softmax

# --------------------------------------------------------------------------------
#   Part I: Layers used by the original BiDAF model (Baseline Model).
# --------------------------------------------------------------------------------
class Embedding(nn.Module):
    
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
        
    Author:
    Chris Chute (chute@stanford.edu)
    """

    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb

class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # fix cuda warning
        self.rnn.flatten_parameters() 
        
        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


# --------------------------------------------------------------------------------
#   Part II: Layers used by the character-level embeddings.
#
#   Refer to : https://github.com/Oceanland-428/Improved-BiDAF-with-Self-Attention
#              https://github.com/umhan35/nlp-squad
# --------------------------------------------------------------------------------

class EmbeddingWithChar(nn.Module):
    """Embedding layer used by BiDAF, with the character-level component.
    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Initial char vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(EmbeddingWithChar, self).__init__()
        self.word_embed = Embedding(word_vectors, hidden_size//2, drop_prob)
        self.char_embed = Embedding_CNN(char_vectors, hidden_size//2, drop_prob)

    def forward(self, w_idxs, c_idxs):
        word_emb = self.word_embed(w_idxs)   # (batch_size, seq_len, hidden_size/2)
        char_emb = self.char_embed(c_idxs)   # (batch_size, seq_len, hidden_size/2)

        emb = torch.cat([word_emb, char_emb], dim=2)

        return emb


class Embedding_CNN(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, char_vectors, embed_size, drop_prob=0.2,
                       char_embed_size=64, char_limit=16, kernel_size=5):
        """
        It initializes the Embedding_CNN class.
        
        :param char_vectors: a torch.Tensor of shape (num_chars, char_embed_size)
        :param embed_size: the size of the word embedding
        :param drop_prob: dropout rate
        :param char_embed_size: the size of the character embedding, defaults to 64 (optional)
        :param char_limit: the maximum length of a word in characters, defaults to 16 (optional)
        :param kernel_size: 5, defaults to 5 (optional)
        """
        super(Embedding_CNN, self).__init__()

        self.embed_size = embed_size
        self.max_word_len = char_limit
        self.dropout_rate = drop_prob
        self.kernel_size = kernel_size

        self.char_embedding = nn.Embedding.from_pretrained(char_vectors)
        self.char_embed_size = self.char_embedding.embedding_dim

        self.cnn = CombineCNN( char_embed_dim=self.char_embed_size,
                        word_embed_dim=self.embed_size,
                        max_word_length=self.max_word_len,
                        kernel_size=self.kernel_size)

        self.highway = Highway(embed_dim=self.embed_size)

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        
        x: Tensor of integers of shape (sentence_length, batch_size, max_word_length) 
        output: Tensor of shape (sentence_length, batch_size, embed_size), 
        containing the CNN-based embeddings for each word of the sentences in the batch
        """
        # (sentence_length, batch_size, max_word_length)
        x_emb = self.char_embedding(x) # look up char embedding
        sentence_length, batch_size, max_word_length, char_embed_size = x_emb.size()
        # (sentence_length, batch_size, max_word_length, char_embed_size)
        x_reshaped = x_emb.view(sentence_length*batch_size, max_word_length, char_embed_size).permute(0, 2, 1)
        # (sentence_length * batch_size, char_embed_size, max_word_length)
        x_conv = self.cnn(x_reshaped)
        # (sentence_length * batch_size, word_embed_size)
        x_highway = self.highway(x_conv)
        # (sentence_length * batch_size, word_embed_size)
        x_word_emb = self.dropout(x_highway)
        # (sentence_length * batch_size, word_embed_size)
        output = x_word_emb.view(sentence_length, batch_size, -1)
        # (sentence_length, batch_size, word_embed_size)

        return output

class CombineCNN(nn.Module):
    """ To combine the character embeddings """
    def __init__(self, char_embed_dim: int, 
                       word_embed_dim: int, 
                       max_word_length: int=21, 
                       kernel_size: int=5): 
        """
        :param char_embed_dim: the dimension of the character embedding
        :type char_embed_dim: int
        :param word_embed_dim: the dimension of the output word embedding
        :type word_embed_dim: int
        :param max_word_length: The maximum length of a word in the dataset, defaults to 21
        :type max_word_length: int (optional)
        :param kernel_size: The size of the convolutional kernel, defaults to 5
        :type kernel_size: int (optional)
        """
        super(CombineCNN, self).__init__()

        self.conv1d = nn.Conv1d(
            in_channels=char_embed_dim,
            out_channels=word_embed_dim,
            kernel_size=kernel_size,
            bias=True)

        self.maxpool = nn.MaxPool1d(max_word_length - kernel_size + 1)

    def forward(self, x):
        # (batch size, char embedding size, max word length)
        x_conv = self.conv1d(x)
        # (batch size, word embedding size, max_word_length - kernel_size + 1)
        x_conv_out = self.maxpool(torch.relu(x_conv)).squeeze()
        # (batch size, word embedding size)

        return x_conv_out

class Highway(nn.Module):
    # Encode an input dimension using a highway network.
    def __init__(self, embed_dim: int): # word embedding dimension
        super(Highway, self).__init__()
        
        self.conv_out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.gate = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x_conv_out):
        x_proj = torch.relu(self.conv_out_proj(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))

        x = x_gate * x_conv_out + (1 - x_gate) * x_conv_out

        return x 
    
# --------------------------------------------------------------------------------
#   Part III: Layers used by Self-attention.
#      
#   Refer to : https://github.com/Oceanland-428/Improved-BiDAF-with-Self-Attention
#              https://github.com/umhan35/nlp-squad
# --------------------------------------------------------------------------------
class SelfAtt(nn.Module):
    """
        The self attention get the attention score of cotext and context. 
        Refer to : https://allenai.github.io/allennlp-docs/api/allennlp.modules.time_distributed.html?highlight=time%20distributed#module-allennlp.modules.time_distributed 
        and to : Attention is all you need pytroch implementation :https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master/transformer
        Args:
            hidden_size (int): Size of hidden activations.
            drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, hidden_size, drop_prob):
        super(SelfAtt, self).__init__()

        self.drop_prob = drop_prob
        # self.att_wrapper = TimeDistributed(nn.Linear(hidden_size*4, hidden_size))
        self.att_wrapper = nn.Linear(hidden_size*4, hidden_size)
        self.trilinear = TriLinearAttention(hidden_size)
        self.self_att_upsampler = nn.Linear(hidden_size*3, hidden_size*4)
        # self.self_att_upsampler = TimeDistributed(nn.Linear(hidden_size*3, hidden_size*4))
        self.enc = nn.GRU(hidden_size, hidden_size//2, 1,
                           batch_first=True,
                           bidirectional=True)
        self.hidden_size = hidden_size

    
    def forward(self, att, c_mask):
        #(batch_size, c_len, 1600)
        att_copy = att.clone() # To save the original data of attention from pervious layer. 
        #(batch_size * c_len, 1600)
        att_wrapped = self.att_wrapper(att) # unroll the second dimention with the first dimension, and roll it back, change of dimension.
        #non-linearity activation function
        att = F.relu(att_wrapped) #(batch_size * c_len, 1600)
        c_mask = c_mask.unsqueeze(dim=2).float() #(batch_size, c_len, 1)

        drop_att = F.dropout(att, self.drop_prob, self.training) #(batch_size * c_len, hidden_size)

        encoder, _ = self.enc(drop_att)

        self_att = self.trilinear(encoder, encoder) # get the self attention (batch_size, c_len, c_len)

        # to match the shape of the attention matrix 
        mask = (c_mask.view(c_mask.shape[0], c_mask.shape[1], 1) * c_mask.view(c_mask.shape[0], 1, c_mask.shape[1])).cuda()
        identity = torch.eye(c_mask.shape[1], c_mask.shape[1]).view(1, c_mask.shape[1], c_mask.shape[1]).cuda()
        mask = mask * (1 - identity)
        
        #get the self attention vector features
        self_att_softmax = masked_softmax(self_att, mask, log_softmax=False)
        self_att_vector = torch.matmul(self_att_softmax, encoder)

        #concatenate to make the shape (batch, c_len, 1200) 
        conc = torch.cat((self_att_vector, encoder, encoder * self_att_vector), dim=-1)
        
        #To match with the input attention, we have to upsample the hidden_size from 1200 to 1600.
        upsampler = self.self_att_upsampler(conc)
        out = F.relu(upsampler)

        #(batch_size, c_len, 1600)
        att_copy += out

        att = F.dropout(att_copy, self.drop_prob, self.training)
        return att

class TriLinearAttention(nn.Module):
    """
    This function is taken from Allen NLP group, refer to github: 
    https://github.com/chrisc36/allennlp/blob/346e294a5bab1ec0d8f2af962cfe44abc450c369/allennlp/modules/tri_linear_attention.py
    
    TriLinear attention as used by BiDaF, this is less flexible more memory efficient then
    the `linear` implementation since we do not create a massive
    (batch, context_len, question_len, dim) matrix
    """
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self._x_weights = Parameter(torch.Tensor(input_dim, 1))
        self._y_weights = Parameter(torch.Tensor(input_dim, 1))
        self._dot_weights = Parameter(torch.Tensor(1, 1, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self.input_dim*3 + 1))
        self._y_weights.data.uniform_(-std, std)
        self._x_weights.data.uniform_(-std, std)
        self._dot_weights.data.uniform_(-std, std)


    def forward(self, matrix_1, matrix_2):
        # pylint: disable=arguments-differ

        # Each matrix is (batch_size, time_i, input_dim)
        batch_dim = matrix_1.shape[0]
        time_1 = matrix_1.shape[1]
        time_2 = matrix_2.shape[1]

        # (batch * time1, dim) * (dim, 1) -> (batch * time1, 1)
        x_factors = torch.matmul(matrix_1.reshape(batch_dim * time_1, self.input_dim), self._x_weights)
        x_factors = x_factors.contiguous().view(batch_dim, time_1, 1)  # ->  (batch, time1, 1)

        # (batch * time2, dim) * (dim, 1) -> (batch * tim2, 1)
        y_factors = torch.matmul(matrix_2.reshape(batch_dim * time_2, self.input_dim), self._y_weights)
        y_factors = y_factors.contiguous().view(batch_dim, 1, time_2)  # ->  (batch, 1, time2)

        weighted_x = matrix_1 * self._dot_weights  # still (batch, time1, dim)

        matrix_2_t = torch.transpose(matrix_2, 1, 2)  # -> (batch, dim, time2)

        # Batch multiplication,
        # (batch, time1, dim), (batch, dim, time2) -> (batch, time1, time2)
        dot_factors = torch.matmul(weighted_x, matrix_2_t)

        # Broadcasting will correctly repeat the x/y factors as needed,
        # result is (batch, time1, time2)
        return dot_factors + x_factors + y_factors    

# ------------------------------------------------------------------------------------------------------------------
#   Part IV: Layers used by QANet(Combining Local Convolution with Global Self-Attention for Reading Comprehension).
#   
#   Refer to : https://github.com/jerrylzy/SQuAD-QANet
#              https://github.com/andy840314/QANet-pytorch-
#              https://github.com/inSam/QA-XL
#              https://github.com/datpnguyen/QANet-CS224N
# ------------------------------------------------------------------------------------------------------------------

"""
Assortment of layer implementations specified in QANet
Reference: https://arxiv.org/pdf/1804.09541.pdf
"""

class HighwayEncoderForQANET(nn.Module):
    """
    Edits: An dropout layer with p=0.1 was added

    Encode an input sequence using a highway network.
    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).
    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoderForQANET, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, hidden_size, seq_len)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            t = F.dropout(t, p=0.1, training=self.training)
            x = g * t + (1 - g) * x

        return x

    
class Initialized_Conv1d(nn.Module):
    """
    Wrapper Function
    Initializes nn.conv1d and adds a relu output.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, groups=1,
                 relu=False, bias=False):
        super(Initialized_Conv1d, self).__init__()
        
        self.out = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)


class EmbeddingForQANET(nn.Module):
    """
    Embedding layer specified by QANet. 
    Concatenation of 300-dimensional (p1) pre-trained GloVe word vectors and 200-dimensional (p2) trainable char-vectors 
    Char-vectors have a set length of 16 via truncation or padding. Max value is taken by each row/char? 
    To obtain a vector of (p1 + p2) long word vector 
    Uses two-layer highway network (Srivastava 2015) 

    Note: Dropout was used on character_word embeddings and between layers, specified as 0.1 and 0.05 respectively

    Question: Linear/Conv1d layer before or after highway?
    """

    def __init__(self, p1, p2, hidden_size, dropout_w = 0.1, dropout_c = 0.05):
        super(EmbeddingForQANET, self).__init__()
        self.conv2d = nn.Conv2d(p2, hidden_size, kernel_size = (1,5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = Initialized_Conv1d(p1 + hidden_size, hidden_size, bias=False)
        self.high = HighwayEncoderForQANET(2, hidden_size)
        self.dropout_w = dropout_w
        self.dropout_c = dropout_c


    def forward(self, ch_emb, wd_emb, length):
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = F.dropout(ch_emb, p=self.dropout_c, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        # print("character embedding size after conv {}".format(ch_emb.size()))
        ch_emb, _ = torch.max(ch_emb, dim=3)

        wd_emb = F.dropout(wd_emb, p=self.dropout_w, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.conv1d(emb).transpose(1,2)
        #Emb: shape [batch_size * seq_len * hidden_size]
        #print(emb.size())
        emb = self.high(emb).transpose(1,2)
        return emb

class DepthwiseSeperableConv(nn.Module):
    """
    Performs a depthwise seperable convolution
    First you should only convolve over each input channel individually, afterwards you convolve the input channels via inx1x1 to get the number of output channels
    This method conserves memory
    
    For clarification see the following: 
    https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d
    https://arxiv.org/abs/1706.03059


    Args:
         in_channel (int): input channel
         out_channel (int): output channel
         k (int): kernel size

    Question: Padding in depthwise_convolution
    """
    def __init__(self, in_channel, out_channel, k, bias=True):
        super(DepthwiseSeperableConv, self).__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_channel, out_channels=in_channel, kernel_size = k, groups = in_channel, padding = k//2, bias = False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size = 1, bias=bias)

    def forward(self, input):
        return F.relu(self.pointwise_conv(self.depthwise_conv(input)))


def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)


class SelfAttentionForQANET(nn.Module):
    """
    Implements the self-attention mechanism used in QANet. 

    Using the same implementation in "Attention" is all you need, we set value_dim = key_dim = d_model / num_head

    See references here: 
    https://arxiv.org/pdf/1706.03762.pdf
    https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#multi-head-self-attention

    Question: Do I use bias in the linear layers? 
    """
    def __init__(self, d_model, num_head, dropout=0.1):
        super(SelfAttentionForQANET, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.num_head = num_head
        self.kv_conv = Initialized_Conv1d(in_channels=d_model, out_channels=d_model*2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, relu=False, bias=False)

    def forward(self, x, mask):
        kv = self.kv_conv(x)
        query = self.query_conv(x)
        kv = kv.transpose(1,2)
        query = query.transpose(1,2)
        Q = self.split_last_dim(query, self.num_head)
        K, V = [self.split_last_dim(tensor, self.num_head) for tensor in torch.split(kv, self.d_model, dim=2)]

        key_depth_per_head = self.d_model // self.num_head
        Q *= key_depth_per_head**-0.5
        x = self.dot_product_attention(Q, K, V, mask)
        return self.combine_last_two_dim(x.permute(0,2,1,3)).transpose(1, 2)

    def dot_product_attention(self, q, k ,v, mask):
        logits = torch.matmul(q, k.permute(0,1,3,2))
        shapes = [x  if x != None else -1 for x in list(logits.size())]
        mask = mask.view(shapes[0], 1, 1, shapes[-1])
        logits = mask_logits(logits, mask)
        
        weights = F.softmax(logits, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.matmul(weights, v)        
        

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret
        

def PositionEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    """
    Returns the position relative to a sinusoidal wave at varying frequency
    """
    x = x.transpose(1, 2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal.to(x.get_device())).transpose(1, 2)


def get_timing_signal(length, channels,
                      min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal
    
class EncoderForQANET(nn.Module):
    """
    Encoder structure specified in the QANet implementation

    Args:
         num_conv (int): number of depthwise convlutional layers
         d_model (int): size of model embedding
         num_head (int): number of attention-heads
         k (int): kernel size for convolutional layers
         dropout (float): layer dropout probability
    """

    def __init__(self, num_conv, d_model, num_head, k, dropout = 0.1):
        super(EncoderForQANET, self).__init__()
        self.convs = nn.ModuleList([DepthwiseSeperableConv(d_model, d_model, k) for _ in range(num_conv)])
        self.conv_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_conv)])

        self.att = SelfAttentionForQANET(d_model, num_head, dropout = dropout)
        self.FFN_1 = Initialized_Conv1d(d_model, d_model, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(d_model, d_model, bias=True)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.num_conv = num_conv
        self.dropout = dropout

    def forward(self, x, mask, l, blks):
        """
        dropout probability: uses stochastic depth survival probability = 1 - (l/L)*pL, 
        reference here: https://arxiv.org/pdf/1603.09382.pdf 
        Question: uhhh you drop the whole layer apparently, and you apply dropout twice for each other layer?
        """
        total_layers = (self.num_conv + 1) * blks
        out = PositionEncoder(x)
        dropout = self.dropout

        for i, conv in enumerate(self.convs):
            res = out
            out = self.conv_norms[i](out.transpose(1,2)).transpose(1,2)
            if (i) % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = conv(out)
            out = self.res_drop(out, res, dropout*float(l)/total_layers)
            l += 1

        res = out
        out = self.norm_1(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.att(out, mask)
        out = self.res_drop(out, res, dropout*float(l)/total_layers)
        l += 1
        res = out

        out = self.norm_2(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.res_drop(out, res, dropout*float(l)/total_layers)
        return out

        
    def res_drop(self, x, res, drop):
        if self.training == True:
           if torch.empty(1).uniform_(0,1) < drop:
               return res
           else:
               return F.dropout(x, drop, training=self.training) + res
        else:
            return x + res

class CQAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        w4C = torch.empty(d_model, 1)
        w4Q = torch.empty(d_model, 1)
        w4mlu = torch.empty(1, 1, d_model)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)
        self.dropout = dropout

    def forward(self, C, Q, Cmask, Qmask):
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        batch_size_c = C.size()[0]
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        S = self.trilinear_for_attention(C, Q)
        Cmask = Cmask.view(batch_size_c, Lc, 1)
        Qmask = Qmask.view(batch_size_c, 1, Lq)
        S1 = F.softmax(mask_logits(S, Qmask), dim=2)
        S2 = F.softmax(mask_logits(S, Cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out.transpose(1, 2)

    def trilinear_for_attention(self, C, Q):
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        dropout = self.dropout
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1,2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res        

class QAOutput(nn.Module):
    def __init__(self, hidden_size):
        super(QAOutput, self).__init__()
        self.w1 = Initialized_Conv1d(hidden_size*2, 1)
        self.w2 = Initialized_Conv1d(hidden_size*2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = mask_logits(self.w1(X1).squeeze(), mask)
        Y2 = mask_logits(self.w2(X2).squeeze(), mask)
        p1 = F.log_softmax(Y1, dim=1)
        p2 = F.log_softmax(Y2, dim=1)        
        return p1, p2