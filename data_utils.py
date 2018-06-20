import pandas as pd 
import numpy as np 
import fcsparser
import os
from os.path import join
from os.path import exists
from os.path import dirname
#from sklearn import preprocessing
from sklearn.externals import joblib
import pickle as pkl

from sklearn.cluster import KMeans


"""
Customized label encoder
"""
class labelEncoder:

	def __init__(self):
		self.count = 0
		self.converter = {}

	def encode (self, label):
		num = self.converter.get(label)
		if num is None:
			self.converter[label] = self.count
			self.count += 1
			return self.count - 1
		else:
			return num

"""
meta: Milk Patho Metadata as pandas DataFrame
data_dir: directory where fcs files are stored
as_list: return the dataset as list of dataFrames, useful for bag-of-word model. Setting this to true will disable concat, shuffle and save
concat: whether to concatenate the cell data
concat_size: if concat, how many row are concatenated together to form a longer row
shuffle: whether to shuffle the data
include_time: whether to include the Time column in the fcs data
save_to_disk: whether to save the data onto disk
save_name: filename to save data onto disk if save_to_disk is True
"""
def getDataSetFromMeta(meta, data_dir, as_list = False, concat=True, concat_size=1000 
	,shuffle=True, include_time = False, save_to_disk = True, save_name="dataset", verbose=False, cb=False):
	
	num_rows = meta.shape[0]
	
	cow_encoder = labelEncoder()
	patho_encoder = labelEncoder()
	qrt_encoder = labelEncoder()

	dataset = None
	dataset_list = []
	for i in range(num_rows):
		# Traverse through metadata
		# Fetch metadata row
		row = meta.iloc[i]
		
		# get filename for all three replicates
		filename = row["FCS_FILENAME1"]
		filename_r1 = filename + "_1.fcs"
		if not exists(join(data_dir, filename_r1)):
			# data file missing for the current row
			# discard the current row
			continue

		filename_r2 = filename + "_2.fcs"
		filename_r3 = filename + "_3.fcs"


		if not cb:
			patho = row["PATHO"]
		else:
			patho = row["HS_PATHO"]
		cow = row["COW"]
		loc = row["LOC"]
		qrt = row["UQTR"]
		lact = row["LACT"]
		intensity = row["INT_PATH"]

		# Load fcs files
		_, data1 = fcsparser.parse(join(data_dir, filename_r1))
		_, data2 = fcsparser.parse(join(data_dir, filename_r2))
		_, data3 = fcsparser.parse(join(data_dir, filename_r3))
		data = pd.concat([data1, data2, data3])

		if not include_time:
			data = data.drop(columns=['Time'])

		if as_list:
			dataset_list.append((data, patho, cow, qrt, loc, lact, intensity))
			continue

		# Turn dataFrame data into numpy matrix
		data_mat = data.as_matrix()
		
		if concat:
			n, d = data_mat.shape
			
			# shuffle data matrix before concatenating
			p = np.random.permutation (n)
			data_mat = data_mat[p]

			num_splits = n // concat_size
			num_rows_cropped = num_splits * concat_size
			data_mat = data_mat[:num_rows_cropped, :]
			data_mat = np.reshape(data_mat, (num_splits, d*concat_size))
		
		cow_num = cow_encoder.encode (cow)
		qrt_num = qrt_encoder.encode (qrt)
		patho_num = patho_encoder.encode (patho)

		one_col = np.ones(data_mat.shape[0]).reshape(data_mat.shape[0], 1)

		data_mat = np.append(data_mat, one_col * patho_num, axis=1)
		data_mat = np.append(data_mat, one_col * cow_num, axis=1)
		data_mat = np.append(data_mat, one_col * qrt_num, axis=1)
		data_mat = np.append(data_mat, one_col * loc, axis=1)
		data_mat = np.append(data_mat, one_col * lact, axis=1)
		data_mat = np.append(data_mat, one_col * intensity, axis=1)

		if dataset is None:
			dataset = data_mat
		else:
			dataset = np.vstack ((dataset, data_mat))
		
		if verbose:
			print ("Processed row {0}/{1}".format(i+1, num_rows))

	if as_list:
		return dataset_list

	# save encoders to disk
	if not exists("Encoders"):
		os.makedirs ("Encoders")
	with open("Encoders/patho.pkl", "wb") as f:
		pkl.dump (patho_encoder, f)
	with open("Encoders/cow.pkl", "wb") as f:
		pkl.dump (cow_encoder, f)
	with open("Encoders/qrt.pkl", "wb") as f:
		pkl.dump (qrt_encoder, f)

	if shuffle:
		perm = np.random.permutation(dataset.shape[0])
		data_mat = dataset[perm]
	
	# save dataset to disk
	if save_to_disk:
		npy_data_dirn = "npy_data"
		if not exists (npy_data_dirn):
			os.makedirs(npy_data_dirn)
		dataset.dump(join(npy_data_dirn, save_name + ".npy"))

	return dataset

"""
Generate codebook for high dim flow cytometry data using KMeans as the clustering algorithm
data: dataFrame of all flow cytometry data
num_clusters: number of codewords the codebook will contain
save_codebook: whether we want to save the codebook model to the disk
save_name: if save_codebook is True, the name of the file that we want to save our model as
"""
def genCodebook (data, num_clusters = 100, save_codebook=True, save_name="codebook_model.pkl", verbose=False):
	# Fit data with kMeans
	cbKMeans = KMeans (n_clusters = num_clusters, n_jobs=1, random_state = 12345).fit (data.as_matrix())
	if verbose:
		print (cbKMeans.cluster_centers_)

	# save model to disk
	if save_codebook:
		if not exists("Codebook"):
			os.makedirs ("Codebook")
		joblib.dump(cbKMeans, join("Codebook", save_name + ".pkl"))

	return cbKMeans


"""
Generate BoW dataset from list of dataFrame and codebook
data_list: list of dataFrame and labels (packed in tuple). Can be generated using getDataSetFromMeta by setting as_list = True
codebook: a KMeans model used as codebook to quantize the data
article_length: length of each article 
sample_size_per_data: number of articles generated from each of the dataFrame
save_to_disk: whether to save the generated dataset to disk
save_name = the name of saved dataset if save_to_disk = True
"""
def genBoWDataset (data_list, codebook, article_length=1000, sample_size_per_data = 50, include_time = False, save_to_disk = False, save_name = "dataset_bow", cb=False):
	dataset = None
	if not cb:
		patho_encoder = labelEncoder()
	cow_encoder = labelEncoder()
	qrt_encoder = labelEncoder()

	n_clusters = codebook.cluster_centers_.shape[0]
	for (data, patho, cow, qrt, loc, lact, path_int) in data_list:
		for _ in range(sample_size_per_data):
			# sample articles from current dataFrame
			sample = data.sample(n = article_length)
			sample_mat = sample.as_matrix()

			# Quantize result using codebook
			quantize_result = codebook.predict(sample_mat)

			# Sum up the number of occurance of each of the words in the sample, producing bag of word
			bow = np.array([np.sum(quantize_result == i) for i in range(n_clusters)])
			
			# Encode the labels
			if not cb:
				lpatho = patho_encoder.encode (patho)
			else:
				lpatho = patho
			lcow = cow_encoder.encode (cow)
			lqrt = qrt_encoder.encode (qrt)

			# Generated labeled BoW dataset row
			bow_l = np.append(bow, [lpatho, lcow, lqrt, loc, lact, path_int])

			# stack dataset
			if dataset is None:
				dataset = bow_l
			else:
				dataset = np.vstack((dataset, bow_l))

	# save encoders to disk
	if not exists("EncodersBow"):
		os.makedirs ("EncodersBow")
	if not cb:
		with open("EncodersBow/patho.pkl", "wb") as f:
			pkl.dump (patho_encoder, f)
	with open("EncodersBow/cow.pkl", "wb") as f:
		pkl.dump (cow_encoder, f)
	with open("EncodersBow/qrt.pkl", "wb") as f:
		pkl.dump (qrt_encoder, f)

	# save dataset to disk
	if save_to_disk:
		npy_data_dirn = "npy_data"
		if not exists (npy_data_dirn):
			os.makedirs(npy_data_dirn)
		dataset.dump(join(npy_data_dirn, save_name + ".npy"))

	return dataset
		