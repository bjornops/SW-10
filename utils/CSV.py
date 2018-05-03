from utils.config import process_config


def StoreAsCSV(option_log_list):
    config = process_config("../configs/test_config.json")
    filename = config.exp_name + "_" + config.map_name + "_" + config.test_id

    for o in option_log_list:
        with open(filename, 'a+') as f:
            data = "%d,%d,%d\n" % (o[0], o[1], o[2])
            f.write(data)
