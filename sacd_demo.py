import numpy as np
import scipy.fft as fft
from scipy import ndimage
import tifffile
import matplotlib.pyplot as plt
from skimage.restoration import richardson_lucy
import os
from scipy.special import erf

# ==========================================
# 1. 基础工具函数
# ==========================================

def generate_rsf(gama, n=None):
    if n is None:
        n = int(np.ceil(gama / np.sqrt(8 * np.log(2)) * np.sqrt(-2 * np.log(0.0002))) + 1)
    sigma = gama / np.sqrt(8 * np.log(2))
    kernelRadius = min(int(np.ceil(sigma * np.sqrt(-2 * np.log(0.0002))) + 1), int(np.floor(n / 2)))
    ii = np.arange(-kernelRadius, kernelRadius + 1)
    rsf_x = 0.5 * (erf((ii + 0.5) / (np.sqrt(2) * sigma)) - erf((ii - 0.5) / (np.sqrt(2) * sigma)))
    kernel = np.outer(rsf_x, rsf_x)
    return kernel / np.sum(kernel)

# ==========================================
# 2. 核心算法函数
# ==========================================

def fourier_interpolation(img, itp_fac):
    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        frames, h, w = img.shape
    
    new_h, new_w = int(h * itp_fac), int(w * itp_fac)
    im_f = fft.fft2(img, axes=(-2, -1))
    im_f_shifted = fft.fftshift(im_f, axes=(-2, -1))
    
    pad_h, pad_w = (new_h - h) // 2, (new_w - w) // 2
    if img.ndim == 3:
        pad_width = ((0, 0), (pad_h, pad_h), (pad_w, pad_w))
    else:
        pad_width = ((pad_h, pad_h), (pad_w, pad_w))
        
    im_f_padded = np.pad(im_f_shifted, pad_width, mode='constant', constant_values=0)
    out = fft.ifft2(fft.ifftshift(im_f_padded, axes=(-2, -1)), axes=(-2, -1))
    return np.real(out) * (itp_fac ** 2)

def cumulant(im, n):
    if n == 2:
        term = im[:-1, :, :] * im[1:, :, :]
        return np.mean(term, axis=0)
    raise NotImplementedError("Currently only supports 2nd order")

def rl_core(data, kernel, iterations):
    if data.ndim == 3:
        deconvolved = np.zeros_like(data)
        for i in range(data.shape[0]):
             deconvolved[i] = richardson_lucy(data[i], kernel, num_iter=iterations, clip=False)
        return deconvolved
    return richardson_lucy(data, kernel, num_iter=iterations, clip=False)

# ==========================================
# 3. SACDm 主流程
# ==========================================

def SACDp_main(imgstack, pixel_size=65, NA=1.3, wavelength=561, mag=2, iter1=7, iter2=8, subfactor=0.8):
    print(f"开始 SACD 重建... 数据维度: {imgstack.shape}")
    resolution = 0.61 * wavelength / NA 
    sigma_pixel = resolution / pixel_size
    psf = generate_rsf(sigma_pixel, n=min(imgstack.shape[1], imgstack.shape[2]))
    psf_v2 = generate_rsf(mag * sigma_pixel, n=min(imgstack.shape[1], imgstack.shape[2])*mag)
    
    stack = imgstack.astype(np.float32) - np.min(imgstack)
    print("执行预反卷积...")
    datadecon = rl_core(stack, psf, iterations=iter1)
    print("执行傅里叶插值...")
    datadeconl = fourier_interpolation(datadecon, mag)
    datadeconl[datadeconl < 0] = 0
    
    print("计算累积量...")
    mean_img = np.mean(datadeconl, axis=0)
    fluctuation_stack = np.abs(datadeconl - subfactor * mean_img)
    cum_img = np.abs(cumulant(fluctuation_stack, 2))
    
    print("执行后反卷积...")
    psf_final = (psf_v2 ** 2) / np.sum(psf_v2 ** 2)
    return rl_core(cum_img, psf_final, iterations=iter2)

# ==========================================
# 4. 运行配置 
# ==========================================

if __name__ == "__main__":
    # --- 1. 实验数据路径配置 ---
    DISK_NAME = "MyDisk"
    # 注意：路径中如果有空格，Python 的 os.path 会自动处理，不需要额外加引号
    DATA_FOLDER = f"/Volumes/{DISK_NAME}/example data for up samped PSF/Experimental_data/TetraSpeck_680nm_-1um_1um_50nm_330nm_ground_12"
    FILE_NAME = "TetraSpeck_680nm_-1um_1um_50nm_330nm_ground_12_MMStack_Pos0.ome.tif"
    
    FULL_PATH = os.path.join(DATA_FOLDER, FILE_NAME)
    
    # --- 2. 读取数据逻辑 ---
    imgstack = None
    if os.path.exists(FULL_PATH):
        print(f"✅ 正在从硬盘读取实验数据: {FILE_NAME}")
        try:
            # 读取 .ome.tif 文件
            imgstack = tifffile.imread(FULL_PATH)
            print(f"原始数据维度: {imgstack.shape}")
            
            # --- 3. 维度自动对齐 (非常重要) ---
            # 算法要求维度是 (Frames, Height, Width)
            # 如果读进来是 (H, W, Frames)，比如 (512, 512, 100)
            if imgstack.ndim == 3:
                # 检查最后一维是否是帧数（通常帧数比像素尺寸小，或者根据经验判断）
                if imgstack.shape[2] < imgstack.shape[0] and imgstack.shape[2] < imgstack.shape[1]:
                    print("检测到维度为 (H, W, Frames)，正在转置为 (Frames, H, W)...")
                    imgstack = np.transpose(imgstack, (2, 0, 1))
            elif imgstack.ndim == 2:
                print("检测到单帧图像，将其扩展为单帧序列...")
                imgstack = imgstack[np.newaxis, :, :]
            
            # 如果数据太大（比如几千帧），建议先截取一部分跑通流程
            if imgstack.shape[0] > 100:
                print(f"数据量较大 ({imgstack.shape[0]} 帧)，为节省时间，Demo仅取前 100 帧进行处理...")
                imgstack = imgstack[:100, :, :]

            # --- 4. 运行算法 ---
            # 注意：这里的参数 (pixel_size, NA, wavelength) 应根据你的实验实际参数修改
            # 默认：pixel_size=65, NA=1.3, wavelength=561
            sr_img = SACDp_main(imgstack, pixel_size=65, NA=1.3, wavelength=561)
            
            # --- 5. 可视化结果 ---
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Experimental Raw (Avg)")
            plt.imshow(np.mean(imgstack, axis=0), cmap='hot')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.title("SACD Reconstructed")
            plt.imshow(sr_img**0.5, cmap='hot') # 使用0.5次幂增加对比度
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"❌ 读取出错: {e}")
    else:
        print(f"❌ 路径不存在，请检查硬盘是否插好。查找路径为: {FULL_PATH}")