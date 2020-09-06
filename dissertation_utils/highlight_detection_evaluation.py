import os
import random

games_path = '/mnt/storage1/dissertation_dataset/games'
configs_path = '/mnt/storage1/dissertation_dataset/games/configs'

def do_test(test_games):
    f_test = open(os.path.join(configs_path, 'test.txt'), 'w')
    f_test_vr = open(os.path.join(configs_path, 'test_vr.txt'), 'w')
    f_test_eg = open(os.path.join(configs_path, 'test_eg.txt'), 'w')

    for game in test_games:
        game_path = os.path.join(games_path, game)
        sliding_window_path = os.path.join(game_path, 'sliding_window_videos')

        d_name = 'sliding_window_videos_information_length_{}'.format(15) # best hyperparam
        d_path = os.path.join(sliding_window_path, d_name)
        f_test_vr.write('{} {}\n'.format(game_path, d_path))

        d_name = 'sliding_window_videos_information_expected_goals'
        d_path = os.path.join(sliding_window_path, d_name)
        f_test_eg.write('{} {}\n'.format(game_path, d_path))

        d_path = os.path.join(game_path, 'gt_videos_observations')
        f_test_eg.write('{} {}\n'.format(game_path, d_path))

        f_test.write('{}\n'.format(game))

    f_test.close()
    f_test_vr.close()
    f_test_eg.close()

def do_val(val_games):
    f_val = open(os.path.join(configs_path, 'validation.txt'), 'w')
    f_val_vr = open(os.path.join(configs_path, 'validation_vr.txt'), 'w')

    for game in val_games:
        game_path = os.path.join(games_path, game)
        sliding_window_path = os.path.join(game_path, 'sliding_window_videos')
        for l in [10, 15, 20]:
            d_name = 'sliding_window_videos_information_length_{}'.format(l)
            d_path = os.path.join(sliding_window_path, d_name)

            f_val_vr.write('{} {}\n'.format(game_path, d_path))

        f_val.write('{}\n'.format(game))

    f_val.close()
    f_val_vr.close()

if __name__ == '__main__':
    random.seed(42)
    all_games = sorted([game_path for game_path in os.listdir(games_path) if game_path != 'configs'])
    random.shuffle(all_games)

    val_games = all_games[:30]
    test_games = all_games[30:60]

    do_val(val_games)
    do_test(test_games)
