import numpy as np
import cv2 as cv

import utils

from scipy import stats


def sustraction(img1: np.array, img2: np.array, th: int):
    return np.abs(img1 - img2) > th


def consecutive_sustraction(frames: list, nframes: int, threslhold: int, d: int):
    if len(frames) == nframes:
        return (255 * sustraction(frames[-1], frames[-d-1], threslhold)).astype(np.uint8)
    return np.zeros_like(frames[-1])


def last_n_images_median_sustraction(frames: list, nframes: int):
    if len(frames) == nframes:
        return np.median(np.asarray(frames), axis=0).astype(np.uint8)
    return np.zeros_like(frames[-1])


def last_n_images_mode_sustraction(frames: list, nframes: int):
    if len(frames) == nframes:
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=np.asarray(frames)).astype(np.uint8)
    return np.zeros_like(frames[-1])


background = None
def moving_average_sustraction(frames: list, nframes: int, factor: float):
    global background
    if background is None:
        background = np.copy(frames[-1])

    background = (factor * frames[-1].astype(np.uint8)) + ((1 - factor) * background.astype(np.uint8)).astype(np.uint8)
    return background


# Static sustraction
'''img = cv.imread('./bs/fr00001.png', 0).astype(np.int16)
bg = cv.imread('./bs/bg/fr00004.png', 0).astype(np.int16)
result = 255 * sustraction(img, bg, 25)
utils.show_image(result)'''

# Consecutive sustraction
'''utils.process_video('./bs2/fr0010%d.png', consecutive_sustraction, 5, {'threslhold': 25, 'd': 4}, fps=2)'''

# Last n images median
'''utils.process_video('./bs/fr00%d01.png', last_n_images_median_sustraction, 5, {}, fps=0.2)'''

# Last n images mode
'''utils.process_video('./bs/fr00%d01.png', last_n_images_mode_sustraction, 5, {}, fps=2)'''

# Moving average
'''video_path = './video/videoplayback.mp4'
utils.process_video(video_path, moving_average_sustraction, 1, {'factor': 0.05}, fps=30)'''



