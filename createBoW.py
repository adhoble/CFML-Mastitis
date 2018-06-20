from data_utils import getDataSetFromMeta, genCodebook, genBoWDataset
import pandas as pd

print ('-----Load FCS data-----')
meta_dir = "milk_patho_meta.csv"
meta = pd.read_csv(meta_dir)
data_dir = "patho_data"
data_list = getDataSetFromMeta(meta, data_dir, as_list=True)

print ('-----Generate Codebook-----')
dl_first = [df[0] for df in data_list]
data_all = pd.concat(dl_first)
print ("Concatenated data shape: {}".format(data_all.shape))
cb = genCodebook (data_all.sample(frac=0.2), verbose=False)
print ("Model shape: {}".format(cb.cluster_centers_.shape))

print ('-----Generate BoW Dataset-----')
bows = genBoWDataset(data_list, cb, sample_size_per_data = 80, save_to_disk = True, save_name="bow_1000_80")
print ('-----All done-----')


