from utils.config import process_config


def StoreAsCSV(option):
    config = process_config("../configs/test_config.json")
    filename = config.exp_name + "_" + config.map_name + "_" + config.test_id

    with open(filename, 'w') as f:
        data = "%d,%d,%d\n" % (option[0], option[1], option[2])
        f.write(data)
