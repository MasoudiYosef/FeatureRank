# GPU Kurulum Talimatları

Bu dokümanda GPU desteğiyle TensorFlow'u kurma adımları açıklanmıştır.

## 📋 Sistem Gereksinimleri

### Hardware:
- **NVIDIA GPU** (Compute Capability 3.5 ve üstü)
- GPU VRAM: En az 2GB (önerilen: 4GB+)

### Software:
- **CUDA Toolkit**: 12.3 veya 12.4
- **cuDNN**: 9.1 veya 9.2
- **TensorFlow**: 2.21.0

## 🔧 Adım-Adım Kurulum

### 1. NVIDIA CUDA Toolkit Kurulumu

#### macOS:
```bash
# macOS'ta GPU desteği sınırlıdır. Metal Performance Shaders kullanılır.
# CUDA kurulması gerekli değildir, ancak TensorFlow GPU desteği sınırlıdır.

# GPU kontrolü:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### Windows:
```bash
# NVIDIA CUDA Toolkit 12.4 indirin
# https://developer.nvidia.com/cuda-12-4-0-download-center

# Kurulum sırasında:
# - "Custom" seçin
# - CUDA Toolkit seçin
# - Tüm varsayılanları kabul edin
# - Kurulum bittiğinde sistemi yeniden başlatın

# Kurulumu doğrulayın:
nvcc --version
```

#### Linux (Ubuntu/Debian):
```bash
# CUDA 12.4 resmi repo'dan:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-12-4

# Kurulumu doğrulayın:
nvcc --version
```

### 2. cuDNN Kurulumu

#### Tüm Platformlar:

1. **cuDNN indir** (NVIDIA hesabı gerekir):
   - https://developer.nvidia.com/cudnn
   - **cuDNN 9.2 for CUDA 12.x** indir

2. **Dosyaları uygun yerlere kopy et**:

   **Windows:**
   ```bash
   # cuDNN zip dosyasını aç
   # CUDA_PATH genellikle: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
   
   # Kopyala:
   # cuDNN\bin\cudnn*.dll → CUDA_PATH\bin\
   # cuDNN\include\cudnn*.h → CUDA_PATH\include\
   # cuDNN\lib\x64\cudnn*.lib → CUDA_PATH\lib\x64\
   ```

   **Linux:**
   ```bash
   tar -xvf cudnn-linux-x86_64-9.2.0.tar.xz
   sudo cp cudnn-linux-x86_64-9.2.0/include/cudnn.h /usr/local/cuda-12.4/include/
   sudo cp cudnn-linux-x86_64-9.2.0/lib/libcudnn* /usr/local/cuda-12.4/lib64/
   sudo chmod a+r /usr/local/cuda-12.4/lib64/libcudnn*
   ```

   **macOS:**
   ```bash
   # macOS Metal Performance Shaders kullandığı için cuDNN zorunlu değildir
   # Ancak TensorFlow GPU desteği kısıtlıdır
   ```

### 3. NVIDIA Driver Kurulumu

```bash
# Windows: NVIDIA GeForce Experience veya Driver Download sayfasından kur
# https://www.nvidia.com/Download/index.aspx

# Linux:
sudo apt-get install nvidia-driver-555

# Doğrulayın:
nvidia-smi
```

### 4. Python Ortamı Kurulması

```bash
# 1. conda ortamı oluştur (Python 3.10 veya 3.11 önerilir)
conda create -n tf-gpu python=3.11 -y
conda activate tf-gpu

# 2. GPU requirements kurulumu
pip install -r requirements-gpu.txt
```

### 5. GPU Desteğini Doğrulayın

```bash
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('GPU Devices:', tf.config.list_physical_devices('GPU'))
if tf.config.list_physical_devices('GPU'):
    print('✅ GPU HAZIR!')
else:
    print('❌ GPU BULUNAMADI')
"
```

## 🚀 Kullanım

### Scripti GPU ile çalıştırma:

```bash
# Varsayılan (otomatik GPU algıla):
python scripts/run_autoencoder.py --dataset-name breast_cancer_data.csv

# Açıkça GPU kullan:
python scripts/run_autoencoder.py --device gpu --dataset-name breast_cancer_data.csv

# Açıkça CPU kullan (GPU yok sayarak):
python scripts/run_autoencoder.py --device cpu --dataset-name breast_cancer_data.csv
```

## 🔍 Sorun Giderme

### Problem: "Could not load dynamic library 'libcudart.so.12'"

**Çözüm:**
```bash
# LD_LIBRARY_PATH ayarla (Linux):
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Windows PATH'i kontrol et:
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64
```

### Problem: "GPU devices not found but GPU option selected"

**Çözüm:**
```bash
# 1. NVIDIA Driver kontrol et:
nvidia-smi

# 2. CUDA kurulumunu kontrol et:
nvcc --version

# 3. cuDNN kurulumunu kontrol et:
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"

# 4. TensorFlow CUDA desteği kontrol et:
python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"
```

### Problem: "Out of GPU memory"

**Çözüm:**
```bash
# Script'te batch size küçült:
python scripts/run_autoencoder.py --batch-size 8

# Veya diğer işlemler kapatıp GPU belleğini temizle:
# Windows Task Manager veya Linux: killall python
```

## 📊 Performance Karşılaştırması

Tipik breast_cancer_data.csv için:
- **CPU (i7 8-cores)**: ~45 saniye/epoch
- **GPU (NVIDIA RTX 3080)**: ~2 saniye/epoch
- **Speedup**: ~22x hızlı

## ℹ️ Notlar

- TensorFlow GPU desteği **macOS'ta sınırlıdır** (Metal Performance Shaders)
- CUDA/cuDNN sürümleri uyumlu olmalıdır (bkz: Sistem Gereksinimleri)
- GPU bellegi optimizasyonu otomatik yapılır (`memory_growth=True`)

## 🔗 Kaynaklar

- [TensorFlow GPU Guide](https://www.tensorflow.org/install/gpu)
- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- [cuDNN Download](https://developer.nvidia.com/cudnn)
- [NVIDIA Driver Download](https://www.nvidia.com/Download/index.aspx)
