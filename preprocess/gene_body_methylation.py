import matplotlib.pyplot as plt
import numpy as np
import constants as cntsts
import sys

def get_gene_meth(meth_seq, genes_df,  bin_num, threshold = 0.5, flanking_size = 2000, context=None, context_seq=None):

    genes_avg_p = np.zeros(bin_num, dtype=np.double)
    genes_avg_n = np.zeros(bin_num, dtype=np.double)
    flac_up_avg_p = np.zeros(bin_num, dtype=np.double)
    flac_up_avg_n = np.zeros(bin_num, dtype=np.double)
    flac_down_avg_p = np.zeros(bin_num, dtype=np.double)
    flac_down_avg_n = np.zeros(bin_num, dtype=np.double)

    gene_bins_sum = np.zeros(bin_num)
    flac_up_bins_sum = np.zeros(bin_num)
    flac_down_bins_sum = np.zeros(bin_num)


    for index, row in genes_df.iterrows():
        is_template = row['strand'] == '+'

        prev_gene_end = -1
        next_gene_start = sys.maxsize

        if index > 0 and genes_df.iloc[index-1]['chr'] == row['chr']:
            prev_gene_end = genes_df.iloc[index-1]['end']
        if index < len(genes_df)-1 and genes_df.iloc[index+1]['chr'] == row['chr']:
            next_gene_start = genes_df.iloc[index+1]['start']

        seq_meth = meth_seq[row['chr']][row['start'] - flanking_size: row['end'] + flanking_size]
        if context_seq != None:
            cntx_seq = context_seq[row['chr']][row['start'] - flanking_size: row['end'] + flanking_size]


        flac_bin_size = int(flanking_size/bin_num)
        gene_bin_size = int((row['end'] - row['start'])/bin_num)
        flac_down_num_binsoverlap = min((row['start'] - prev_gene_end)/flac_bin_size, bin_num)
        flac_up_num_binsoverlap = min((next_gene_start - row['end'])/flac_bin_size, bin_num)


        if not is_template:
            seq_meth = seq_meth[::-1]
            if context_seq != None:
                cntx_seq = cntx_seq[::-1]
            flac_down_num_binsoverlap, flac_up_num_binsoverlap = flac_up_num_binsoverlap, flac_down_num_binsoverlap

        for i in range(bin_num):
            cs = None
            if context_seq!=None:
                cs = cntx_seq[i*flac_bin_size: (i+1) * flac_bin_size]
            m_p, m_n = get_meth_percentage(seq_meth[i*flac_bin_size: (i+1) * flac_bin_size], threshold, context=context, cntx_seq=cs)
            if not is_template:
                m_p, m_n = m_n, m_p
            if m_n != None and m_p != None and i > bin_num - flac_down_num_binsoverlap - 1:
                flac_down_avg_p[i] = update_mean(flac_down_avg_p[i], flac_down_bins_sum[i], m_p, flac_bin_size)
                flac_down_avg_n[i] = update_mean(flac_down_avg_n[i], flac_down_bins_sum[i], m_n, flac_bin_size)
                flac_down_bins_sum[i] += flac_bin_size

            if context_seq!=None:
                cs = cntx_seq[i*gene_bin_size + flanking_size: (i+1) * gene_bin_size + flanking_size]
            m_p, m_n = get_meth_percentage(seq_meth[i*gene_bin_size + flanking_size: (i+1) * gene_bin_size + flanking_size], threshold, context=context, cntx_seq=cs)
            if not is_template:
                m_p, m_n = m_n, m_p
            if m_n != None and m_p != None:
                genes_avg_p[i] = update_mean(genes_avg_p[i], gene_bins_sum[i], m_p, gene_bin_size)
                genes_avg_n[i] = update_mean(genes_avg_n[i], gene_bins_sum[i], m_n, gene_bin_size)
                gene_bins_sum[i] += gene_bin_size

            if context_seq!=None:
                cs = cntx_seq[i*flac_bin_size + len(seq_meth) - flanking_size: (i+1) * flac_bin_size + len(seq_meth) - flanking_size]
            m_p, m_n = get_meth_percentage(seq_meth[i*flac_bin_size + len(seq_meth) - flanking_size: (i+1) * flac_bin_size + len(seq_meth) - flanking_size], threshold, context=context, cntx_seq=cs)
            if not is_template:
                m_p, m_n = m_n, m_p
            if m_n != None and m_p != None and  i < flac_up_num_binsoverlap:
                flac_up_avg_p[i] = update_mean(flac_up_avg_p[i], flac_up_bins_sum[i], m_p, flac_bin_size)
                flac_up_avg_n[i] = update_mean(flac_up_avg_n[i], flac_up_bins_sum[i], m_n, flac_bin_size)
                flac_up_bins_sum[i] += flac_bin_size

    return genes_avg_p, genes_avg_n, flac_up_avg_p, flac_up_avg_n, flac_down_avg_p, flac_down_avg_n


def update_mean(mean, sum_weights, new_value, weight):
    return ((mean * sum_weights) + (new_value * weight)) / (sum_weights + weight)


def get_meth_percentage(meth_stat, threshold, context=None, cntx_seq=None):

    countCs_p = 0
    countCs_n = 0
    countMethCs_p = 0
    countMethCs_n = 0

    for i in range(len(meth_stat)):
        if context == None or cntx_seq[i] == cntsts.ContextTypes.cntx_str[context]:
            if float(meth_stat[i]) > 0:
                countCs_p += 1
            elif float(meth_stat[i]) < 0:
                countCs_n += 1
            if float(meth_stat[i]) > threshold:
                countMethCs_p += 1
            elif float(meth_stat[i]) < -1 * threshold:
                countMethCs_n += 1
    if countCs_p != 0 and countCs_n != 0:
        return float(countMethCs_p) / countCs_p, float(countMethCs_n) / countCs_n # QUESTION
    else:
        return None, None



def plot_gene_body_meth(organism_name, meth_seq, genes_df, bin_num, threshold = 0.1, context=None, context_seq = None):

    genes_avg_p, genes_avg_n, flac_up_avg_p, flac_up_avg_n, flac_down_avg_p, flac_down_avg_n = get_gene_meth(meth_seq, genes_df,  bin_num, threshold=threshold, context=context, context_seq=context_seq)

    final_p = np.concatenate((flac_down_avg_p , genes_avg_p , flac_up_avg_p))
    final_n = np.concatenate((flac_down_avg_n , genes_avg_n , flac_up_avg_n))

    plt.switch_backend('agg')
    plt.tick_params(left=False, labelleft=False)
    plt.box(False)
    plt.ylabel("$meC/C$")
    ticks = [0, bin_num, bin_num * 2, bin_num * 3]
    labels = ['      5\' flanking region', '           gene body', '       3\' flanking region', '']
    plt.xticks(ticks, labels, horizontalalignment='left')
    plt.grid(False)
    plt.style.use('seaborn')
    plt.plot(range(0, 3 * bin_num), final_p, color='blue', linewidth=4.0)
    plt.plot(range(0, 3 * bin_num), final_n, color='red', linewidth=4.0)
    plt.axhline(y=0.0, color='black', linestyle='-')
    plt.axvline(x=0.0, color='black', linestyle='-')
    plt.rcParams['axes.facecolor'] = 'white'
    output_root = '/home/ssere004/SalDMR/predictordataprovider/output/'
    if context != None:
        plt.savefig(output_root + organism_name+'/' + 'gene_plots/' + 'genebody_' + str(organism_name) + '_' + context + '.jpg', dpi=2000)
    else:
        plt.savefig(output_root + organism_name+'/' + 'gene_plots/' + 'genebody_' + str(organism_name) + '.jpg', dpi=2000)
    plt.close()
