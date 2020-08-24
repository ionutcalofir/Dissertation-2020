from absl import flags
from absl import app
from absl import logging

from football_utils.dataset_generation import DatasetGeneration
from football_utils.configs_generation import ConfigsGeneration
from football_utils.game_generation import GameGeneration
from football_utils.highlight_detection import HighlightDetection

FLAGS = flags.FLAGS

if not FLAGS.is_parsed():
    flags.DEFINE_enum('phase', None, ['generate_dataset', 'generate_configs', 'generate_game', 'highlight_detection'], 'Phase to run.')

    # DATASET GENERATION
    flags.DEFINE_enum('dataset_generation_name', None, ['video_recognition', 'expected_goals'], 'What dataset to generate.')
    flags.DEFINE_list('dataset_generation_video_recognition_classes', ['pass', 'shot'], 'What classes to generate for video recognition.')
    flags.DEFINE_string('dataset_generation_path', '/mnt/storage1/football_dataset/raw_dataset', 'Path to the raw dataset.')
    flags.DEFINE_string('dataset_generation_output_path', None, 'Path to the output dataset.')
    flags.DEFINE_bool('dataset_generation_downscale_videos', True, 'Whether to downscale the videos.')

    # CONFIGS GENERATION
    flags.DEFINE_string('configs_generation_dataset_path', None, 'Path to classes examples.')

    # HIGHLIGHT DETECTION
    flags.DEFINE_string('highlight_detection_game_path', None, 'Path to the game.')

def main(_):
    if FLAGS.phase is None:
        logging.error('Please choose a phase option!')
        raise Exception('"phase" flag is none!')

    if FLAGS.phase == 'generate_dataset':
        if FLAGS.dataset_generation_name is None:
            logging.error('Please select what dataset you want to generate.')
            raise Exception('"dataset_generation_name" flag is none!')
        if FLAGS.dataset_generation_output_path is None:
            logging.error('Please select where to save the generated dataset.')
            raise Exception('"dataset_generation_output_path" flag is none!')

        dataset_generation = DatasetGeneration(FLAGS.dataset_generation_name,
                                               FLAGS.dataset_generation_path,
                                               FLAGS.dataset_generation_output_path,
                                               FLAGS.dataset_generation_video_recognition_classes,
                                               FLAGS.dataset_generation_downscale_videos)
        dataset_generation.generate()
    elif FLAGS.phase == 'generate_configs':
        if FLAGS.configs_generation_dataset_path is None:
            logging.error('Please select the path to the dataset.')
            raise Exception('"configs_generation_dataset_path" flag is none!')
        configs_generation = ConfigsGeneration(FLAGS.dataset_generation_name,
                                               FLAGS.configs_generation_dataset_path)

        configs_generation.generate()
    elif FLAGS.phase == 'generate_game':
        game_generation = GameGeneration(FLAGS.dataset_generation_path,
                                         FLAGS.dataset_generation_output_path,
                                         FLAGS.dataset_generation_downscale_videos)
        game_generation.generate()
    elif FLAGS.phase == 'highlight_detection':
        ob = HighlightDetection(FLAGS.highlight_detection_game_path)
        ob.highlight()

    logging.info('Done!')

if __name__ == '__main__':
    app.run(main)
