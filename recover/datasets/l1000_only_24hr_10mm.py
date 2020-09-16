from recover.datasets.l1000_data import L1000
import numpy as np

class L1000TwentyFourHrTenMM(L1000):
    def __init__(self, transform=None, pre_transform=None, fp_bits=1024, fp_radius=4):
        super().__init__(transform, pre_transform, fp_bits, fp_radius, 'L1000_24Hr_10MM')

    def filter_df(self, data_df):
        data_df = data_df[np.isclose(10., data_df['pert_idose_value'])]
        data_df = data_df[np.isclose(24., data_df['pert_itime_value'])]
        data_df = data_df[~data_df['cid'].isna()]

        data_df['cid'] = np.digitize(data_df['cid'], bins=np.sort(data_df['cid'].unique()) + 1)

        return data_df

    def download(self):
        super().download()

    def process(self):
        super().process()

if __name__ == '__main__':
    x = L1000TwentyFourHrTenMM()

    import pdb; pdb.set_trace()
    genes = x.data.gene_expr.flatten()
    import matplotlib.pyplot as plt
    hist, bins = np.histogram(genes, bins = 50)
    #max_val = max(hist)
    #hist = [ float(n)/max_val for n in hist]

    # Plot the resulting histogram
    center = (bins[:-1]+bins[1:])/2
    width = 0.7*(bins[1]-bins[0])
    plt.bar(center, hist, align = 'center', width = width)
    plt.title('Frequency of transcription element values')
    plt.xlabel('Transcription element values')
    plt.ylabel('Number of occurrences')
    plt.show()

    #plt.hist(genes, bins=12)
    #plt.xlim(-6., 6.)
    #plt.xticks(np.arange(-6, 6, step=1))
    #plt.show()
