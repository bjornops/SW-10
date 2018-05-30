import numpy as np
import tensorboard.backend.event_processing.event_accumulator as event_accumulator
import glob, os

def main():
    exp_name = "cold_env_reset"
    exp_dir = summary_path(exp_name)

    print("Generating statistics for experiment '" + exp_name + "'")
    print("From directory: '" + exp_dir + "'")

    event_dirs = get_event_dirs(exp_dir)
    for dir in event_dirs:
        event = get_event(dir)
        event_reward_scalar = get_scalar_array(event, "Reward")
        event_reward_avg = np.average(event_reward_scalar)
        print("Average reward for " + os.path.basename(dir) + " " + str(event_reward_avg))

def calc_event_reward_avg(event_file):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    scalar_str = "Reward"
    lc = np.stack(
        [np.asarray([scalar.step, scalar.value])
         for scalar in ea.Scalars(scalar_str)])
    return np.average(lc)

def get_event_dirs(exp_dir):
    non_empty_dirs = []
    for entry in os.listdir(exp_dir):
        entry_path = os.path.join(exp_dir, entry)
        if os.path.isdir(entry_path) and (len(os.listdir(entry_path)) > 0):
            non_empty_dirs.append(entry_path)
    return non_empty_dirs

def summary_path(exp_name):
    cur_dir = os.getcwd()
    base_dir = os.path.dirname(cur_dir)
    exp_summary_dir = os.path.join(base_dir, "experiments", exp_name, "summary")
    if os.path.isdir(exp_summary_dir):
        return exp_summary_dir
    else:
        raise Exception("'" + exp_summary_dir + "' is not a directory.")

def get_event(dir_path):
    return max(
        glob.glob('{}/*'.format(dir_path)),
        key=os.path.getctime)

def get_scalar_array(event_file, scalar_str):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    lc = np.stack(
        [np.asarray([scalar.step, scalar.value])
         for scalar in ea.Scalars(scalar_str)])
    return(lc)

if __name__ == '__main__':
    main()

