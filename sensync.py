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

class Sensync():
    
    def sensor_motion(self, sensor_path:str, frames:int):
        df = pd.read_csv(sensor_path)
        # Calculate magnitude of acceleration
        df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
        
        # Create a time index based on the timestamps
        df['time_index'] = pd.to_datetime(df['seconds_elapsed'], unit='s')
        
        df = df.set_index('time_index')
        df_resampled = df['magnitude'].resample('33.33ms').mean()

        return df_resampled
        
    def video_motion(self, video_path:str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file")
            raise Exception('Could not open the video file specified, please check the path and try again')
        # Initialize variables
        prev_frame = None
        frame_differences = []
        
        while True:
        
            ret, frame = cap.read()
            if not ret:
                break
        
            # Convert the frame to grayscale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            # If this isn't the first frame...
            if prev_frame is not None:
                frame_diff = cv2.absdiff(prev_frame, frame_gray)
                total_diff = frame_diff.sum()
                frame_differences.append(total_diff)
            prev_frame = frame_gray
            
        return frame_differences
    
    def sync(self, video_path:str, 
             frames:int,
             sensor_path:str, 
             export_path='',
             export=False,
             ):
        if export == True and export_path == '':
            raise Exception('export_path must be valid!')
        offset = 0
        
        vid_motion = self.video_motion(video_path)
        sensor_motion = self.sensor_motion(sensor_path, frames)
        
        
        
        return offset
    
    