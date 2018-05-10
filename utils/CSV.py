from utils.config import process_config
import os


def StoreAsCSV(option_log_list):
    # option_log_list format -> [selected_tactical, cur_step, reward, self.episode_count]

    config = process_config("../configs/test_config.json")
    filename = os.path.join("..", "experiments", config.exp_name, config.map_name + "_" + config.test_id + ".csv")

    for o in option_log_list:
        with open(filename, 'a+') as f:
            data = "%d,%d,%d,%d\n" % (o[0], o[1], o[2], o[3])
            f.write(data)
