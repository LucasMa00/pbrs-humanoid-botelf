'''
Author: shuaikangma shuaikangma@qq.com
Date: 2025-03-21 22:48:30
LastEditors: shuaikangma shuaikangma@qq.com
LastEditTime: 2025-03-21 22:48:34
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import torch
import numpy as np
import matplotlib.pyplot as plt

class WaveVisualizer:
    def __init__(self, phase_freq=1.0, eps=0.01):
        self.phase_freq = phase_freq
        self.eps = eps

    def smooth_sqr_wave(self, phase):
        p = 2. * torch.pi * phase * self.phase_freq
        return torch.sin(p) / (2 * torch.sqrt(torch.sin(p)**2. + self.eps**2.)) + 1./2.

    def plot_wave(self):
        phases = torch.linspace(0, 2, 500)  # 从0到2的周期范围
        values = self.smooth_sqr_wave(phases).numpy()

        plt.figure(figsize=(8, 4))
        plt.plot(phases.numpy(), values, label="Smooth Square Wave", color='b')
        plt.xlabel("Phase")
        plt.ylabel("Wave Value")
        plt.title("Smooth Square Wave Function")
        plt.legend()
        plt.grid()
        plt.show()

# 创建实例并绘制曲线
visualizer = WaveVisualizer(phase_freq=1.0, eps=0.2)
visualizer.plot_wave()
