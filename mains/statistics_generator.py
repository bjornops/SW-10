import numpy as np
import tensorboard.backend.event_processing.event_accumulator as event_accumulator

def main():
    event_file_path = ""
    event_avg = calc_event_reward_avg(event_file_path)
    print(str(event_avg))

def calc_event_reward_avg(event_file):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    scalar_str = "Reward"
    lc = np.stack(
        [np.asarray([scalar.step, scalar.value])
         for scalar in ea.Scalars(scalar_str)])
    return np.average(lc)

if __name__ == '__main__':
    main()