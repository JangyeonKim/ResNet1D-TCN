import os
import glob
import torch
import random
import librosa
import torchaudio
import numpy as np
import sys
import csv
from lipreading.utils import read_txt_lines


class MyDataset(object):
    def __init__(self, modality, data_partition, data_dir, label_fp, #annonation_direc=None,
        preprocessing_func=None, data_suffix='.npz'):
        assert os.path.isdir( data_dir ), "File path provided for the labels does not exist. Path iput: {}".format(data_dir)
        self._data_partition = data_partition
        self._data_dir = data_dir
        self._data_suffix = data_suffix

        self._label_fp = label_fp
        #self._annonation_direc = annonation_direc

        #self.fps = 25 if modality == "video" else 16000
        #self.is_var_length = True
        #self.label_idx = -3
        
        self.preprocessing_func = preprocessing_func

        self._data_dir = glob.glob(os.path.join(self._data_dir, self._data_partition,"*.csv"))

        self._data_files = []

        #self.list = self.load_list(self._data_dir)

        self.maxlen = 10
        self.load_dataset()

    def load_dataset(self):

        # -- read the labels file
        self._labels = read_txt_lines(self._label_fp)


        # -- add examples to self._data_files
        self._load_list()


        # -- from self._data_files to self.list
        self.list = dict()
        #self.instance_ids = dict()
        for i, x in enumerate(self._data_files):
            label = x[1]
            self.list[i] = [ x[0], self._labels.index( label ) ]
            #print( self._labels.index( label ))
            #self.instance_ids[i] = self._get_instance_id_from_path( x )

        print('Partition {} loaded'.format(self._data_partition))

    def _load_list(self):
    
        for path_count_label in open(self._data_dir[0]).read().splitlines():
            rel_path, label = path_count_label.split(",")[:2]
            self._data_files.append(
                (
                    rel_path,
                    label
                )
            )


    def load_data(self, filename):

        try:
            if filename.endswith('npz'):
                return np.load(filename)['data']
            elif filename.endswith('mp4'):
                return librosa.load(filename, sr=16000)[0][-19456:]
            elif filename.endswith('wav'):
                waveform = librosa.load(os.path.join(filename), sr=16000)[0]
            
                if len(waveform) > (self.maxlen * 16000):
                    waveform = waveform[:self.maxlen * 16000]

                return waveform
            else:
                return np.load(filename)
        except IOError:
            print( "Error when reading file: {}".format(filename) )
            sys.exit()

    def __getitem__(self, idx):

        raw_data = self.load_data(self.list[idx][0])

        data = raw_data

        preprocess_data = self.preprocessing_func(data)
        label = self.list[idx][1]
        return preprocess_data, label

    def __len__(self):
        return len(self.list)


def pad_packed_collate(batch):
    if len(batch) == 1:
        data, lengths, labels_np, = zip(*[(a, a.shape[0], b) for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])
        data = torch.FloatTensor(np.array(data))
        lengths = [data.size(1)]

    if len(batch) > 1:
        data_list, lengths, labels_np = zip(*[(a, a.shape[0], b) for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])

        if data_list[0].ndim == 3:
            max_len, h, w = data_list[0].shape  # since it is sorted, the longest video is the first one
            data_np = np.zeros(( len(data_list), max_len, h, w))
        elif data_list[0].ndim == 1:
            max_len = data_list[0].shape[0]
            data_np = np.zeros( (len(data_list), max_len))
        for idx in range( len(data_np)):
            data_np[idx][:data_list[idx].shape[0]] = data_list[idx]
        data = torch.FloatTensor(np.array(data_np))
    labels = torch.LongTensor(labels_np)
    return data, lengths, labels
