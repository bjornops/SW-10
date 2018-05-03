from utils.config import process_config


def StoreAsCSV(option):
    config = process_config("../configs/test_config.json")
    filename = config.exp_name + "_" + config.map_name + "_" + config.test_id

    with open(filename, 'w') as f:
        for o in option:
            data = "option:%d,timesteps:%d,reward:%d\n" % (o[0], o[1], o[2])
            f.write(data)
