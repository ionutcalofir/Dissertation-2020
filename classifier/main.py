from absl import flags
from absl import app
from absl import logging

from engine import Engine

FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    flags.DEFINE_string('phase', None, 'Phase to run.')
    flags.DEFINE_string('run_name', 'default', 'The name of the current run')
    flags.DEFINE_string('model_path', None, 'Path to the model')
    flags.DEFINE_float('lr', 0.0001, 'LR')
    flags.DEFINE_float('momentum', 0.9, 'Momentum')
    flags.DEFINE_float('weight_decay', 0.0005, 'Weight decay')
    flags.DEFINE_list('class_weights', None, 'Class weights')
    flags.DEFINE_integer('batch_size', 32, 'Batch size')
    flags.DEFINE_string('train_data_cfg', '/mnt/storage1/dissertation_dataset/expected_goals/configs/train.csv', 'Path to train cfg')
    flags.DEFINE_string('val_data_cfg', '/mnt/storage1/dissertation_dataset/expected_goals/configs/val.csv', 'Path to val cfg')
    flags.DEFINE_string('test_data_cfg', '/mnt/storage1/dissertation_dataset/expected_goals/configs/test.csv', 'Path to test cfg')
    flags.DEFINE_integer('epochs', 100, 'Number of epochs')
    flags.DEFINE_string('data_path', '/mnt/storage1/dissertation_dataset/expected_goals', 'Path to test cfg')
    flags.DEFINE_string('logdir', None, 'Where to save the experiments.')

def main(_):
    if FLAGS.phase is None:
        logging.error('Please choose a phase option!')
        raise Exception('"phase" flag is none!')

    ob = Engine(FLAGS.phase,
                FLAGS.run_name,
                FLAGS.model_path,
                FLAGS.lr,
                FLAGS.momentum,
                FLAGS.weight_decay,
                FLAGS.class_weights,
                FLAGS.batch_size,
                FLAGS.train_data_cfg,
                FLAGS.val_data_cfg,
                FLAGS.test_data_cfg,
                FLAGS.epochs,
                FLAGS.data_path,
                FLAGS.logdir)
    if FLAGS.phase == 'train':
        ob.train()
    elif FLAGS.phase == 'test':
        ob.test()

if __name__ == '__main__':
    app.run(main)
