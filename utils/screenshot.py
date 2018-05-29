import pyautogui as pag
from utils.config import process_config
import os
import sys

def grap_screenshot(worker_id, episode_number):

    config = process_config("../configs/test_config.json")

    # Take screenshot
    pic = pag.screenshot()

    # Set save location
    my_path = "C:\\Users\\Bjørn\\Documents\\GitHub\\bjornops\\SW-10\\experiments\\cold_env_reset\\screenshot"
    name = "%s_%s_%s.png" % (config.map_name, worker_id, str(episode_number))
    path = os.path.join(my_path, name)

    # Save the image
    pic.save(path)

