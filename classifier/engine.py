import os
import torch
import itertools
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from model import Model
from dataset import FootballDataset

class Engine:
    def __init__(self,
                 phase,
                 run_name,
                 model_path,
                 lr,
                 momentum,
                 weight_decay,
                 class_weights,
                 batch_size,
                 train_data_cfg,
                 val_data_cfg,
                 test_data_cfg,
                 epochs,
                 data_path,
                 logdir):
        self._phase = phase
        self._run_name = run_name
        self._model_path = model_path
        self._lr = lr
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._class_weights = torch.FloatTensor([float(class_weight) for class_weight in class_weights]).cuda()
        self._batch_size = batch_size
        self._train_data_cfg = train_data_cfg
        self._val_data_cfg = val_data_cfg
        self._test_data_cfg = test_data_cfg
        self._epochs = epochs
        self._data_path = data_path
        self._logdir = logdir

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self._phase == 'train':
            os.makedirs(self._logdir, exist_ok=True)

            experiments = os.listdir(self._logdir)
            experiments = [experiment for experiment in experiments
                                if '-'.join(experiment.split('-')[1:]) == self._run_name]
            self._run_path = os.path.join(self._logdir, '{}-{}'.format(len(experiments), self._run_name))
            os.makedirs(self._run_path)
            os.makedirs(os.path.join(self._run_path, 'models'))
        elif self._phase == 'test':
            self._run_path = '/'.join(self._model_path.split('/')[:-2])

        writer_logdir = os.path.join(self._run_path, 'summaries_{}'.format(self._phase))
        os.makedirs(writer_logdir, exist_ok=True)
        self._writer = SummaryWriter(log_dir=writer_logdir)

        self._model = Model().build_model()
        self._model.to(self._device)
        if self._phase == 'test':
            self._model = Model().load_model(self._model, self._model_path)

        self._criterion = nn.CrossEntropyLoss(weight=self._class_weights)
        criterion_params = [p for p in self._criterion.parameters() if p.requires_grad]
        params = [p for p in self._model.parameters() if p.requires_grad]
        params.extend(criterion_params)
        self._optimizer = torch.optim.SGD(params, lr=self._lr, momentum=self._momentum,
                                          weight_decay=self._weight_decay)

        if self._phase == 'train':
            self._train_data_loader = torch.utils.data.DataLoader(FootballDataset(dataset_name='train',
                                                                                  data_path=self._data_path,
                                                                                  data_cfg=self._train_data_cfg),
                                                                  batch_size=self._batch_size,
                                                                  shuffle=True,
                                                                  num_workers=4)
            self._val_data_loader = torch.utils.data.DataLoader(FootballDataset(dataset_name='val',
                                                                                data_path=self._data_path,
                                                                                data_cfg=self._val_data_cfg),
                                                                batch_size=self._batch_size,
                                                                shuffle=False,
                                                                num_workers=4)
        elif self._phase == 'test':
            self._test_data_loader = torch.utils.data.DataLoader(FootballDataset(dataset_name='test',
                                                                                 data_path=self._data_path,
                                                                                 data_cfg=self._test_data_cfg),
                                                                 batch_size=self._batch_size,
                                                                 shuffle=False,
                                                                 num_workers=4)

    def _train_one_epoch(self, epoch):
        print('Train')
        self._model.train()
        loss_accumulated = 0
        total_num_images = 0
        correct_preds = 0

        for idx, (images, labels, _) in enumerate(self._train_data_loader):
            images = images.to(self._device)
            labels = labels.to(self._device)

            total_num_images += images.size()[0]

            output = self._model(images)

            y_pred = torch.softmax(output, dim=1)
            loss = self._criterion(output, labels)

            loss_accumulated += loss * len(output[:, 0])
            total_loss = loss_accumulated / total_num_images

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            correct_preds += np.sum(labels.cpu().detach().numpy() == torch.argmax(y_pred, dim=1).cpu().detach().numpy())
            acc = correct_preds / total_num_images

            if idx % 10 == 0:
                print('Ep: {}, It: {}/{}, Lr: {}, Loss: {}, Acc: {}'
                            .format(epoch,
                                    idx + 1, len(self._train_data_loader),
                                    self._optimizer.param_groups[0]["lr"],
                                    total_loss,
                                    acc))

        self._writer.add_scalar(tag='Train/Loss', scalar_value=total_loss, global_step=epoch)
        self._writer.add_scalar(tag='Train/Accuracy', scalar_value=acc, global_step=epoch)
        self._writer.add_scalar(tag='Train/LR', scalar_value=self._optimizer.param_groups[0]["lr"], global_step=epoch)

    def _plot_confusion_matrix(self, cmtx, num_classes, class_names=None, figsize=None):
        """
        Function from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/visualization/utils.py#L48
        """
        if class_names is None or type(class_names) != list:
            class_names = [str(i) for i in range(num_classes)]

        figure = plt.figure(figsize=figsize)
        plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Use white text if squares are dark; otherwise black.
        threshold = cmtx.max() / 2.0
        for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
            color = "white" if cmtx[i, j] > threshold else "black"
            plt.text(
                j,
                i,
                format(cmtx[i, j], ".2f") if cmtx[i, j] != 0 else ".",
                horizontalalignment="center",
                color=color,
            )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        return figure

    def _validation_test(self, data_loader, epoch, header='Val'):
        print(header)
        self._model.eval()

        with torch.no_grad():
            loss_accumulated = 0
            total_num_images = 0
            correct_preds = 0

            all_preds = []
            all_labels = []

            all_goal_probs = []

            for idx, (images, labels, _) in enumerate(data_loader):
                images = images.to(self._device)
                labels = labels.to(self._device)

                total_num_images += images.size()[0]

                output = self._model(images)

                y_pred = torch.softmax(output, dim=1)
                loss = self._criterion(output, labels)

                loss_accumulated += loss * len(output[:, 0])
                total_loss = loss_accumulated / total_num_images

                correct_preds += np.sum(labels.cpu().detach().numpy() == torch.argmax(y_pred, dim=1).cpu().detach().numpy())
                acc = correct_preds / total_num_images

                all_labels.extend(labels.cpu().detach().numpy().tolist())
                all_preds.extend(torch.argmax(y_pred, dim=1).cpu().detach().numpy().tolist())

                all_goal_probs.extend(y_pred[:, 1].cpu().detach().numpy().tolist())

                if idx % 10 == 0:
                    print('Ep: {}, It: {}/{}, Loss: {}, Acc: {}'
                                .format(epoch,
                                        idx + 1, len(data_loader),
                                        total_loss,
                                        acc))

            self._writer.add_scalar(tag='{}/Loss'.format(header), scalar_value=total_loss, global_step=epoch)
            self._writer.add_scalar(tag='{}/Accuracy'.format(header), scalar_value=acc, global_step=epoch)

            self._writer.add_scalar(tag='{}/AG (Goals)'.format(header), scalar_value=sum(all_labels), global_step=epoch)
            self._writer.add_scalar(tag='{}/xG (Expected Goals)'.format(header), scalar_value=sum(all_goal_probs), global_step=epoch)

            cm = confusion_matrix(all_labels, all_preds)
            cm_normalized = confusion_matrix(all_labels, all_preds, normalize='true')

            figure = self._plot_confusion_matrix(cm, 2, ['no_goal', 'goal'])
            figure_normalized = self._plot_confusion_matrix(cm_normalized, 2, ['no_goal', 'goal'])
            self._writer.add_figure(tag='{}/Confusion Matrix Unnormalized'.format(header), figure=figure, global_step=epoch)
            self._writer.add_figure(tag='{}/Confusion Matrix'.format(header), figure=figure_normalized, global_step=epoch)

            tp = cm[1, 1]
            fp = cm[0, 1]
            fn = cm[1, 0]

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * ((precision * recall) / (precision + recall))
            self._writer.add_scalar(tag='{}/Precision'.format(header), scalar_value=precision, global_step=epoch)
            self._writer.add_scalar(tag='{}/Recall'.format(header), scalar_value=recall, global_step=epoch)
            self._writer.add_scalar(tag='{}/F1'.format(header), scalar_value=f1, global_step=epoch)

    def train(self):
        for epoch in range(self._epochs):
            self._train_one_epoch(epoch)
            self._validation_test(self._val_data_loader, epoch)

            Model().save_model(self._model, '{}/models/model_{}.pt'.format(self._run_path, epoch))
            self._writer.flush()

        self._writer.close()

    def test(self):
        epoch = int(self._model_path.split('/')[-1].split('_')[-1][:-3])
        self._validation_test(self._test_data_loader, epoch, header='Test')

        self._writer.close()
