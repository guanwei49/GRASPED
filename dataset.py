
import os
from collections import defaultdict

import numpy as np


from pm4py.objects.log.importer.xes import importer as xes_importer
from sklearn.preprocessing import LabelEncoder


class Dataset(object):
    def __init__(self, dataset_Path,attr_keys):
        # Public properties
        self.dataset_name = dataset_Path

        start_symbol = '▶'
        end_symbol = '■'

        ######hospital test#######
        logPath = os.path.join( dataset_Path)
        log = xes_importer.apply(logPath)

        self.case_lens = []
        feature_columns = defaultdict(list)
        for trace in log:
            self.case_lens.append(len(trace) + 2)
            for attr_key in attr_keys:
                feature_columns[attr_key].append(start_symbol)
            for event in trace:
                for attr_key in attr_keys:
                    feature_columns[attr_key].append(event[attr_key])
            for attr_key in attr_keys:
                feature_columns[attr_key].append(end_symbol)

        # print(feature_columns)

        for key in feature_columns.keys():
            encoder = LabelEncoder()
            feature_columns[key] = encoder.fit_transform(feature_columns[key]) + 1

        # Transform back into sequences
        case_lens = np.array(self.case_lens)
        offsets = np.concatenate(([0], np.cumsum(case_lens)[:-1]))
        self.features = [np.zeros((case_lens.shape[0], case_lens.max())) for _ in range(len(feature_columns))]
        for i, (offset, case_len) in enumerate(zip(offsets, case_lens)):
            for k, key in enumerate(feature_columns):
                x = feature_columns[key]
                self.features[k][i, :case_len] = x[offset: offset + case_len]

    @property
    def num_cases(self):
        """Return number of cases in the event log, i.e., the number of examples in the dataset."""
        return len(self.features[0])

    @property
    def num_events(self):
        """Return the total number of events in the event log."""
        return sum(self.case_lens)

    @property
    def max_len(self):
        """Return the length of the case with the most events."""
        return self.features[0].shape[1]


    @property
    def attribute_dims(self):
        return  np.asarray([f.max()  for f in self.features])

    @property
    def num_attributes(self):
        """Return the number of attributes in the event log."""
        return len(self.features)