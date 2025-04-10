import numpy as np
import collections
import pdb
from sklearn.model_selection import train_test_split
# from keras.utils import to_categorical

import numpy as np
import gensim
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical



def read_fasta_file(fasta_file):
    seq_dict = {}
    bag_sen = list()
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        if line[0] == '>':
            name = line[1:]
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()
    for seq in seq_dict.values():
        seq = seq.replace('T', 'U')
        bag_sen.append(seq)

    return np.asarray(bag_sen)


def RNA2Vec(k, s, vector_dim, model, MAX_LEN, pos_sequences, neg_sequences):
    model1 = gensim.models.Doc2Vec.load(model)
    pos_list = seq2ngram(pos_sequences, k, s, model1.wv)
    neg_list = seq2ngram(neg_sequences, k, s, model1.wv)
    seqs = pos_list + neg_list

    X = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
    y = np.array([1] * len(pos_list) + [0] * len(neg_list))
    y = to_categorical(y)

    embedding_matrix = np.zeros((len(model1.wv.vocab), vector_dim))
    for i in range(len(model1.wv.vocab)):
        embedding_vector = model1.wv[model1.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return X, y, embedding_matrix


def seq2ngram(seqs, k, s, wv):
    list01 = []
    for num, line in enumerate(seqs):
        if num < 3000000:
            line = line.strip()
            l = len(line)
            list2 = []
            for i in range(0, l, s):
                if i + k >= l + 1:
                    break
                list2.append(line[i:i + k])
            list01.append(convert_data_to_index(list2, wv))
    return list01


def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data


def get_1_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 1
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        nucle_com.append(ch0)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index


def get_2_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 2
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base
        ch1 = chars[n % base]
        nucle_com.append(ch0 + ch1)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index


def get_3_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 3
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base
        ch1 = chars[n % base]
        n = n // base
        ch2 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index


def get_4_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 4
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base
        ch1 = chars[n % base]
        n = n // base
        ch2 = chars[n % base]
        n = n // base
        ch3 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index


def frequency(seq, kmer, coden_dict):
    Value = []
    k = kmer
    coden_dict = coden_dict
    for i in range(len(seq) - int(k) + 1):
        kmer = seq[i:i + k]
        kmer_value = coden_dict[kmer.replace('T', 'U')]
        Value.append(kmer_value)
    freq_dict = dict(collections.Counter(Value))
    return freq_dict

'''
def coden(seq, kmer, tris):
    coden_dict = tris
    freq_dict = frequency(seq, kmer, coden_dict)
    vectors = np.zeros((101, len(coden_dict.keys())))
    for i in range(len(seq) - int(kmer) + 1):
        value = freq_dict[coden_dict[seq[i:i + kmer].replace('T', 'U')]]
        vectors[i][coden_dict[seq[i:i + kmer].replace('T', 'U')]] = value / 100
    return vectors
'''
def coden(seq,kmer,tris):
    coden_dict = tris
    freq_dict = frequency(seq,kmer,coden_dict)
    vectors = np.zeros((101, len(coden_dict.keys())))
    for i in range(len(seq) - int(kmer) + 1):
        vectors[i][coden_dict[seq[i:i + kmer].replace('T', 'U')]] = 1
    return vectors.tolist()

coden_dict1 = {'GCU': 0, 'GCC': 0, 'GCA': 0, 'GCG': 0,  # alanine<A>
              'UGU': 1, 'UGC': 1,  # systeine<C>
              'GAU': 2, 'GAC': 2,  # aspartic acid<D>
              'GAA': 3, 'GAG': 3,  # glutamic acid<E>
              'UUU': 4, 'UUC': 4,  # phenylanaline<F>
              'GGU': 5, 'GGC': 5, 'GGA': 5, 'GGG': 5,  # glycine<G>
              'CAU': 6, 'CAC': 6,  # histidine<H>
              'AUU': 7, 'AUC': 7, 'AUA': 7,  # isoleucine<I>
              'AAA': 8, 'AAG': 8,  # lycine<K>
              'UUA': 9, 'UUG': 9, 'CUU': 9, 'CUC': 9, 'CUA': 9, 'CUG': 9,  # leucine<L>
              'AUG': 10,  # methionine<M>
              'AAU': 11, 'AAC': 11,  # asparagine<N>
              'CCU': 12, 'CCC': 12, 'CCA': 12, 'CCG': 12,  # proline<P>
              'CAA': 13, 'CAG': 13,  # glutamine<Q>
              'CGU': 14, 'CGC': 14, 'CGA': 14, 'CGG': 14, 'AGA': 14, 'AGG': 14,  # arginine<R>
              'UCU': 15, 'UCC': 15, 'UCA': 15, 'UCG': 15, 'AGU': 15, 'AGC': 15,  # serine<S>
              'ACU': 16, 'ACC': 16, 'ACA': 16, 'ACG': 16,  # threonine<T>
              'GUU': 17, 'GUC': 17, 'GUA': 17, 'GUG': 17,  # valine<V>
              'UGG': 18,  # tryptophan<W>
              'UAU': 19, 'UAC': 19,  # tyrosine(Y)
              'UAA': 20, 'UAG': 20, 'UGA': 20,  # STOP code
              }

def coden1(seq):
    vectors = np.zeros((len(seq), 21))
    for i in range(len(seq) - 2):
        vectors[i][coden_dict1[seq[i:i + 3].replace('T', 'U')]] = 1
    return vectors.tolist()#矩阵转换为列表
def get_RNA_seq_concolutional_array(seq, motif_len=4):
    seq = seq.replace('U', 'T')
    print(seq)
    alpha = 'ACGT'
    row = (len(seq) + 2 * motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len - 1):
        new_array[i] = np.array([0.25] * 4)

    for i in range(row - 3, row):
        new_array[i] = np.array([0.25] * 4)

    # pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len - 1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25] * 4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
    print(new_array)
    return new_array


def processFastaFile(seq):
    phys_dic = {
        'A': [1, 1, 1],
        'U': [0, 0, 1],
        'C': [0, 1, 0],
        'G': [1, 0, 0]}
    seqLength = len(seq)
    sequence_vector = np.zeros([101, 3])
    for i in range(0, seqLength):
        sequence_vector[i, 0:3] = phys_dic[seq[i]]
    for i in range(seqLength, 101):
        sequence_vector[i, -1] = 1
    return sequence_vector


def dpcp(seq):
    phys_dic = {
        # Shift Slide Rise Tilt Roll Twist Stacking_energy Enthalpy Entropy Free_energy Hydrophilicity
        'AA': [-0.08, -1.27, 3.18, -0.8, 7, 31, -13.7, -6.6, -18.4, -0.93, 0.04],
        'AU': [-0.06, -1.36, 3.24, 1.1, 7.1, 33, -15.4, -5.7, -15.5, -1.1, 0.14],
        'AC': [0.23, -1.43, 3.24, 0.8, 4.8, 32, -13.8, -10.2, -26.2, -2.24, 0.14, ],
        'AG': [-0.04, -1.5, 3.3, 0.5, 8.5, 30, -14, -7.6, -19.2, -2.08, 0.08],
        'UA': [0.07, -1.7, 3.38, 1.3, 9.4, 32, -14.2, -13.3, -35.5, -2.35, 0.1],
        'UU': [0.23, -1.43, 3.24, 0.8, 4.8, 32, -13.8, -10.2, -26.2, -2.24, 0.27],
        'UC': [0.07, -1.39, 3.22, 0, 6.1, 35, -16.9, -14.2, -34.9, -3.42, 0.26],
        'UG': [-0.01, -1.78, 3.32, 0.3, 12.1, 32, -11.1, -12.2, -29.7, -3.26, 0.17],
        'CA': [0.11, -1.46, 3.09, 1, 9.9, 31, -14.4, -10.5, -27.8, -2.11, 0.21],
        'CU': [-0.04, -1.5, 3.3, 0.5, 8.5, 30, -14, -7.6, -19.2, -2.08, 0.52],
        'CC': [-0.01, -1.78, 3.32, 0.3, 8.7, 32, -11.1, -12.2, -29.7, -3.26, 0.49],
        'CG': [0.3, -1.89, 3.3, -0.1, 12.1, 27, -15.6, -8, -19.4, -2.36, 0.35],
        'GA': [-0.02, -1.45, 3.26, -0.2, 10.7, 32, -16, -8.1, -22.6, -1.33, 0.21],
        'GU': [-0.08, -1.27, 3.18, -0.8, 7, 31, -13.7, -6.6, -18.4, -0.93, 0.44],
        'GC': [0.07, -1.7, 3.38, 1.3, 9.4, 32, -14.2, -10.2, -26.2, -2.35, 0.48],
        'GG': [0.11, -1.46, 3.09, 1, 9.9, 31, -14.4, -7.6, -19.2, -2.11, 0.34]}

    seqLength = len(seq)
    sequence_vector = np.zeros([101, 11])
    k = 2
    for i in range(0, seqLength - 1):
        sequence_vector[i, 0:11] = phys_dic[seq[i:i + k]]
    return sequence_vector


def nd(seq, seq_length):
    seq = seq.strip()
    nd_list = [None] * seq_length
    for j in range(seq_length):
        # print(seq[0:j])
        if seq[j] == 'A':
            nd_list[j] = round(seq[0:j + 1].count('A') / (j + 1), 3)
        elif seq[j] == 'U':
            nd_list[j] = round(seq[0:j + 1].count('U') / (j + 1), 3)
        elif seq[j] == 'C':
            nd_list[j] = round(seq[0:j + 1].count('C') / (j + 1), 3)
        elif seq[j] == 'G':
            nd_list[j] = round(seq[0:j + 1].count('G') / (j + 1), 3)
    return np.array(nd_list)


def read_fasta_file(fasta_file):
    seq_dict = {}
    bag_sen = list()
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        if line[0] == '>':
            name = line[1:]
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()

    for seq in seq_dict.values():
        seq = seq.replace('T', 'U')
        bag_sen.append(seq)

    return np.asarray(bag_sen)


def dealwithdata(protein):
    seq_length = 101
    tris1 = get_1_trids()
    tris2 = get_2_trids()
    tris3 = get_3_trids()
    tris4 = get_4_trids()
    dataX = []
    Kmers = []
    probMatrs = []
    NDs = []
    DPCPs = []
    dataY = []
    model = '../pre-trained_models/doc2vec/Doc2Vec_model'
    model1 = gensim.models.Doc2Vec.load(model)


    seqpos_path = '../dataset/' + protein + '/positive'
    seqneg_path = '../dataset/' + protein + '/negative'
    seqpos = read_fasta_file(seqpos_path)
    seqneg = read_fasta_file(seqneg_path)

    pos_list = seq2ngram(seqpos, 10, 1, model1.wv)
    neg_list = seq2ngram(seqneg, 10, 1, model1.wv)
    pos_list = pad_sequences(pos_list, maxlen=101, padding='post')
    neg_list = pad_sequences(neg_list, maxlen=101, padding='post')
    # print(len(pos_list))
    # print(len(neg_list))
    seq = np.concatenate((pos_list, neg_list), axis=0)
    # print(len(seq))


    embedding_matrix = np.zeros((len(model1.wv.vocab), 30))
    for i in range(len(model1.wv.vocab)):
        embedding_vector = model1.wv[model1.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # reduce(lambda x, y: [i + j for i in x for j in y], [['A', 'T', 'C', 'G']] * K)
    with open(seqpos_path) as f:
        for line in f:
            if '>' not in line:
                line = line.replace('T', 'U').strip()
                probMatr = processFastaFile(line)
                probMatr_ND = nd(line, seq_length)
                probMatr_NDCP = np.column_stack((probMatr, probMatr_ND))#2个
                probMatr_DPCP = dpcp(line) / 101
                probMatr_NDPCP = np.column_stack((probMatr_NDCP, probMatr_DPCP))#3
                kmer1 = coden(line.strip(), 1, tris1)
                kmer2 = coden(line.strip(), 2, tris2)
                kmer3 = coden1(line.strip())
                Kmer = np.hstack((kmer1, kmer2, kmer3))
                Feature_Encoding = np.column_stack((probMatr_NDPCP, Kmer))#3+1
                dataX.append(Feature_Encoding.tolist())
                Kmers.append(Kmer.tolist())
                probMatrs.append(probMatr.tolist())
                NDs.append(probMatr_ND.tolist())
                DPCPs.append(probMatr_DPCP.tolist())
                dataY.append([0,1])
    with open(seqneg_path) as f:
        for line in f:
            if '>' not in line:
                line = line.replace('T', 'U').strip()
                probMatr = processFastaFile(line)
                probMatr_ND = nd(line, seq_length)
                probMatr_NDCP = np.column_stack((probMatr, probMatr_ND))
                probMatr_DPCP = dpcp(line) / 101
                probMatr_NDPCP = np.column_stack((probMatr_NDCP, probMatr_DPCP))
                kmer1 = coden(line.strip(), 1, tris1)
                kmer2 = coden(line.strip(), 2, tris2)
                kmer3 = coden1(line.strip())
                Kmer = np.hstack((kmer1, kmer2, kmer3))
                Feature_Encoding = np.column_stack((probMatr_NDPCP, Kmer))
                dataX.append(Feature_Encoding.tolist())
                Kmers.append(Kmer.tolist())
                probMatrs.append(probMatr.tolist())
                NDs.append(probMatr_ND.tolist())
                DPCPs.append(probMatr_DPCP.tolist())
                dataY.append([1,0])
    # 加载邻接矩阵文件
    adj_matrices = np.load('../dataset/' + protein + '/' + protein + '.npy')  # 读取保存的邻接矩阵
    indexes = np.random.choice(len(dataY), len(dataY), replace=False)
    dataX = np.array(dataX)[indexes]

    seq = np.array(seq)[indexes]
    dataY = np.array(dataY)[indexes]
    adj_matrices = np.array(adj_matrices)[indexes]  # 按相同的索引打乱邻接矩阵
    train_X, test_X, train_y, test_y = train_test_split(dataX, dataY, test_size=0.2)
    # train_probMatrs,test_probMatrs, train_NDs,test_NDs, train_DPCPs,test_DPCPs, train_Kmers,test_Kmers,train_seq, test_seq,train_adj, test_adj, train_y, test_y = (
    #     train_test_split(probMatrs, NDs, DPCPs, Kmers,seq,adj_matrices, dataY, test_size=0.2))
    # return train_probMatrs,test_probMatrs, train_NDs,test_NDs, train_DPCPs,test_DPCPs, train_Kmers,test_Kmers, train_seq, test_seq,train_adj, test_adj, train_y, test_y, embedding_matrix
    return train_X, test_X, train_y, test_y








