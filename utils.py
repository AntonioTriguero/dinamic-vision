import argparse
import cv2 as cv
import time
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from multiprocessing.pool import Pool
from concurrent.futures.thread import ThreadPoolExecutor


def get_args_values(args: list):
    ap = argparse.ArgumentParser()
    for arg in args:
        ap.add_argument(arg[0], nargs='+', required=arg[1])
    values = list(vars(ap.parse_args()).values())
    for index, value in enumerate(values):
        if value and len(value) == 1:
            values[index] = value[0]
    return values


def process_video(path: str, frame_processor: callable, verbose: bool = True, fps: int = None, output_path: str = None,
                  images_sequence: bool = False):
    if images_sequence:
        path += "/image_%08d_0.png"
    cap = cv.VideoCapture(path)

    out = None
    if output_path:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out_fps = 10
        if fps:
            out_fps = fps
        out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), out_fps, (frame_width, frame_height))

    progress = tqdm(total=int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
    while True:
        _, frame = cap.read()
        if frame is None:
            break

        frame = frame_processor(frame)

        if fps:
            time.sleep(1 / fps)

        if verbose:
            cv.imshow('Frame', frame)

        if out:
            out.write(frame)

        progress.update()

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv.destroyAllWindows()


def process_iterable(pool, iter: list, func: callable, workers: int = 2):
    with pool(workers) as executor:
        if isinstance(executor, Pool):
            execute = executor.imap
        elif isinstance(executor, ThreadPoolExecutor):
            execute = executor.map
        return list(tqdm(execute(lambda args: func(*args), iter), total=len(iter)))


def show_image(img: np.array, ax: plt.Subplot = None, title: str = ''):
    if ax:
        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.imshow(img)
    else:
        plt.imshow(img)
