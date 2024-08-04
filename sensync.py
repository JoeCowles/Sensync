# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
import os
import pandas as pd
import math
import threading


class Sensync:
    def sensor_motion(self, sensor_path: str, frames: int):
        df = pd.read_csv(sensor_path)
        # Calculate magnitude of acceleration
        df["magnitude"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)

        # Create a time index based on the timestamps
        df["time_index"] = pd.to_datetime(df["seconds_elapsed"], unit="s")

        df = df.set_index("time_index")
        df_resampled = df["magnitude"].resample("33.33ms").mean()

        return df_resampled

    def video_motion(self, video_paths: list):
        def process_video(video_path):
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise IOError(f"Could not open video file: {video_path}")
                prev_frame = None
                local_frame_differences = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_frame is not None:
                        frame_diff = cv2.absdiff(prev_frame, frame_gray)
                        total_diff = frame_diff.sum()
                        local_frame_differences.append(total_diff)
                    prev_frame = frame_gray
                frame_differences.extend(local_frame_differences)
            except Exception as e:
                print(f"Error processing video {video_path}: {str(e)}")

        frame_differences = []
        threads = []
        for video_path in video_paths:
            thread = threading.Thread(target=process_video, args=(video_path,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return frame_differences

    def sync(
        self,
        video_path: str,
        fps: int,
        sensor_path: str,
        window=30,
        export_path="",
        export=False,
    ):
        if export and export_path == "":
            raise ValueError("export_path must be valid!")

        vid_motion = self.video_motion([video_path])
        sensor_motion = self.sensor_motion(sensor_path, fps)

        bestOffset = 0
        bestCorr = -1  # Initialize best correlation to -1 (worst possible correlation)

        for offset in range(-fps * window, fps * window):
            # Calculate correlation for current offset
            try:
                corr = np.corrcoef(
                    vid_motion[offset : offset + 2 * fps * window],
                    sensor_motion[offset : offset + 2 * fps * window],
                )[0, 1]
                if corr > bestCorr:
                    bestCorr = corr
                    bestOffset = offset
            except IndexError:
                continue  # Skip offsets that are out of range

        return bestOffset
