import os
import numpy as np
from sklearn.model_selection import train_test_split
from absl import logging

class ConfigsGeneration:
    def __init__(self,
                 dataset_name,
                 dataset_path):
        self._dataset_name = dataset_name
        self._dataset_path = dataset_path
        self._train_percentage = 0.90

        if self._dataset_name == 'video_recognition':
            self.cls_to_id = {
                'no_action': 0,
                'pass': 1,
                'shot': 2}
        elif self._dataset_name == 'expected_goals':
            self.cls_to_id = {
                '0': 0,
                '1': 1}

        os.makedirs(os.path.join(self._dataset_path, 'configs'), exist_ok=True)

    def _write_csv(self, X, y, mode):
        with open(os.path.join(self._dataset_path, 'configs', mode) + '.csv', 'w') as f:
            for X_elem, y_elem in zip(X, y):
                f.write('{} {}\n'.format(X_elem, y_elem))

            for cls, cls_id in self.cls_to_id.items():
                logging.info('{} {} num examples: {}'.format(mode.title(), cls, np.sum(y == cls_id)))

    def _write_class_weights(self, y_train):
        """
        Formula from https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html. (see class_weight)

        n_samples / (n_classes * np.bincount(y))
        """
        with open(os.path.join(self._dataset_path, 'configs', 'class_weights.txt'), 'w') as f:
            n_samples = y_train.shape[0]
            n_classes = np.amax(y_train) + 1

            class_weights = n_samples / (n_classes * np.bincount(y_train))
            class_weights = [str(round(class_weight, 2)) for class_weight in class_weights]
            f.write('[{}]'.format(','.join(class_weights)))

    def generate(self):
        X = []
        y = []

        classes = sorted([cls for cls in os.listdir(self._dataset_path) if cls not in ['configs', 'observations']])

        for cls in classes:
            cls_path = os.path.join(self._dataset_path, cls)
            for vid in sorted(os.listdir(cls_path)):
                vid_path = os.path.join(cls, vid)

                X.append(vid_path)
                y.append(self.cls_to_id[cls])

        X = np.array(X)
        y = np.array(y)

        X_train, X_valtest, y_train, y_valtest = train_test_split(X, y,
                                                                  train_size=self._train_percentage,
                                                                  random_state=42,
                                                                  stratify=y)

        X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest,
                                                        test_size=0.5,
                                                        random_state=42,
                                                        stratify=y_valtest)

        self._write_class_weights(y_train)

        self._write_csv(X_train, y_train, mode='train')
        self._write_csv(X_val, y_val, mode='val')
        self._write_csv(X_test, y_test, mode='test')

        logging.info('Done!')
