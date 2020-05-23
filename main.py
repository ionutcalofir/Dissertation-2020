from absl import flags
from absl import app
from absl import logging

from football_utils.dataset_generation import DatasetGeneration

FLAGS = flags.FLAGS

if not FLAGS.is_parsed():
    flags.DEFINE_enum('phase', None, ['generate_dataset'], 'Phase to run.')

    # DATASET GENERATION
    flags.DEFINE_enum('dataset_generation_name', None, ['pass', 'shot_goal', 'heatmap'], 'What dataset to generate.')
    flags.DEFINE_string('dataset_generation_path', './datasets/raw_dataset', 'Path to the raw dataset.')
    flags.DEFINE_string('dataset_generation_output_path', './datasets/pass_dataset', 'Path to the output dataset.')
    flags.DEFINE_bool('dataset_generation_downscale_videos', True, 'Wether to downscale the videos.')

def main(_):
    if FLAGS.phase is None:
        logging.error('Please choose a phase option!')
        raise Exception('"phase" flag is none!')

    if FLAGS.phase == 'generate_dataset':
        if FLAGS.dataset_generation_name is None:
            logging.error('Please select what dataset you want to generate.')
            raise Exception('"dataset_name" flag is none!')

        dataset_generation = DatasetGeneration(FLAGS.dataset_generation_name,
                                               FLAGS.dataset_generation_path,
                                               FLAGS.dataset_generation_output_path,
                                               FLAGS.dataset_generation_downscale_videos)
        dataset_generation.generate()

    logging.info('Done!')

if __name__ == '__main__':
    app.run(main)
