import pahelix.toolkit.linear_rna as linear_rna
import numpy as np
from multiprocessing import Pool, cpu_count


# 读取 fasta 文件内容
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


# 单个序列的邻接矩阵生成函数（多进程中调用）
def generate_adj_matrix(input_sequence):
    # 使用 linear_rna 生成邻接矩阵数据
    data = linear_rna.linear_partition_v(input_sequence, bp_cutoff=0.5)[1]

    # 创建 101x101 的零矩阵
    adj_matrix = np.zeros((101, 101))

    # 填充邻接矩阵（调整索引，避免超界）
    for x, y, prob in data:
        if x - 1 < 101 and y - 1 < 101:
            adj_matrix[x - 1, y - 1] = prob

    return adj_matrix

# 使用多进程生成邻接矩阵
if __name__ == '__main__':
    # 获取 CPU 核心数
    num_workers =  max(cpu_count() - 2, 1)
    print(f"启动 {num_workers} 个进程进行并行处理")
    proteins = ['AUF1', 'DGCR8', 'EWSR1', 'FMRP', 'FXR2', 'HNRNPC', 'HUR', 'IGF2BP1', 'IGF2BP3', 'LIN28A', 'MOV10', 'PTB', 'PUM2', 'TAF15', 'TDP43', 'U2AF65']
    for protein in proteins:
        # 读取正负样本序列
        seqpos_path = './dataset/'+ protein + '/positive'
        seqneg_path = './dataset/'+ protein + '/negative'
        seqpos = read_fasta_file(seqpos_path)
        seqneg = read_fasta_file(seqneg_path)
        # 将所有序列合并为一个列表
        all_sequences = np.concatenate([seqpos, seqneg])
        # 使用进程池进行并行处理
        with Pool(processes=num_workers) as pool:
            # 将每条序列分配到不同的进程中处理，生成所有的邻接矩阵
            adj_matrices = pool.map(generate_adj_matrix, all_sequences)

        # 将生成的邻接矩阵合并为一个三维数组
        adj_matrices = np.array(adj_matrices)

        # 保存三维 numpy 数组为文件
        np.save(protein + '.npy', adj_matrices)
        print(f"邻接矩阵已成功保存为 {protein}.npy，矩阵形状为：{adj_matrices.shape}")
