# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 23:13:44 2023

@author: joewc
"""
import tkinter as tk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from tkVideoPlayer import TkinterVideo
import numpy as np

class Application(tk.Frame):
    
    frame_differences = []
    df_resampled = pd.DataFrame()
    
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.place(relwidth=1, relheight=1)
        self.create_widgets()
        root.geometry("800x700")

    def create_widgets(self):
        self.select_video_button = tk.Button(self)
        self.select_video_button["text"] = "Select Video"
        self.select_video_button["command"] = self.select_video
        self.select_video_button.place(x=300, y=0, width=100, height=50)

        self.select_csv_button = tk.Button(self)
        self.select_csv_button["text"] = "Select CSV"
        self.select_csv_button["command"] = self.select_csv
        self.select_csv_button.place(x=400, y=0, width=100, height=50)

        self.videoPlayer = TkinterVideo(master=root, scaled=True)
        self.videoPlayer.set_size((900, 300), True)
        self.videoPlayer.place(x=0, y=50, relwidth=1, height=300)

        self.fig, self.ax = plt.subplots(3, 1)
        self.ax[0].set_xticklabels([])
        self.ax[1].set_xticklabels([])
        self.ax[2].set_xticklabels([])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().place(x=0, y=350, relwidth=1, relheight=0.50)
        


    def select_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if not video_path:
            print("No video file selected")
            return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video file: {video_path}")
            return
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame from video file: {video_path}")
            return
        
        prev_frame = None
        
        while True:
        
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            # If this isn't the first frame...
            if prev_frame is not None:
                # Compute the absolute difference with the previous frame
                frame_diff = cv2.absdiff(prev_frame, frame_gray)
        
                # Sum the differences to get a measure of motion in the frame
                total_diff = frame_diff.sum()
        
                # Store the result
                self.frame_differences.append(total_diff)
    
            # The current frame becomes the previous frame for the next iteration
            prev_frame = frame_gray
            
        self.videoPlayer.load(video_path)
        self.videoPlayer.play()
        self.ax[0].plot(self.frame_differences)
        self.canvas.draw()
        self.calc_correlation()
        

    def select_csv(self):
        csv_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not csv_path:
            print("No CSV file selected")
            return
        df = pd.read_csv(csv_path)
        df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

        # Create a time index based on the timestamps
        df['time_index'] = pd.to_datetime(df['seconds_elapsed'], unit='s')

        df = df.set_index('time_index')
        self.df_resampled = df['magnitude'].resample('33.33ms').mean()


        self.ax[1].plot(self.df_resampled.index, self.df_resampled.values)
        plt.show()
        self.canvas.draw()
        self.calc_correlation()
        
    def calc_correlation(self):
        if len(self.df_resampled) == 0 or len(self.frame_differences) ==0 :
            return
        frame_diff_series = pd.Series(self.frame_differences, index=pd.date_range(start=self.df_resampled.index[0], periods=len(self.frame_differences), freq='33.33ms'))
        cross_corr = np.correlate(self.df_resampled.fillna(0), frame_diff_series.fillna(0), mode='full')
        best_offset_index = cross_corr.argmax()
        
        best_offset_secs = best_offset_index / 30  # since we're dealing with 30Hz data
        
        print('Best offset is', best_offset_secs, 'seconds')
        
        offsets = np.arange(-len(self.df_resampled) + 1, len(frame_diff_series))
        
        self.ax[2].plot(offsets / 30, cross_corr)  # Dividing by 30 to convert offsets to seconds
        plt.xlabel('Best offset')
        self.canvas.draw()
        plt.show()
        
        
root = tk.Tk()
app = Application(master=root)
app.mainloop()
