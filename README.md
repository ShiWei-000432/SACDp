# SACDp: Super-resolution via Analysis of Cumulants and Deconvolution

SACDp 是一种结合了 **SOFI (超分辨光涨落成像)** 和 **Richardson-Lucy 反卷积** 的超分辨率图像处理算法。它通过分析荧光信号的时间涨落和数学去模糊，实现了超越衍射极限的成像效果。

## 🌟 核心特性
- **傅里叶插值**: 实现无损上采样，为高分辨率重建提供画布。
- **二阶累积量 (Cumulant)**: 利用时间相关性去除背景噪声，并提升分辨率。
- **双重反卷积**: 预处理去模糊与后期锐化相结合，进一步优化点扩散函数 (PSF)。

## 🛠 安装与使用

1. **克隆仓库**:
   ```bash
   git clone https://github.com/ShiWei-000432/SACDp.git
   cd SACDp
   ```

2. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

3. **运行 Demo**:
   ```bash
   python sacd_demo.py
   ```

## 🔬 算法逻辑
1. **Pre-RL**: 使用 `rl_core` 进行初始去模糊。
2. **Fourier Interpolation**: 使用 `fourier_interpolation` 进行空间网格加密。
3. **Cumulant Analysis**: 通过 `cumulant` 提取时间涨落信号，压制噪声。
4. **Post-RL**: 使用针对超分辨校准后的 PSF 进行最终锐化。

## 📄 许可证
MIT License
