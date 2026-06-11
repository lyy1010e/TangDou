# 糖豆视频下载工具

自动下载糖豆 APP 收藏的视频，并裁剪片头和片尾广告。

## 安装

安装 Python 依赖：

```powershell
pip install -r requirements.txt
```

安装 FFmpeg：

```powershell
winget install Gyan.FFmpeg
```

确认 FFmpeg 可用：

```powershell
ffmpeg -version
```

如果要使用 `realesrgan`，下载 Real-ESRGAN ncnn-vulkan 绿色版，并在 `config.ini` 的 `[realesrgan]` 中配置路径。

## 使用

先在手机上抓包收藏的数据，搜索 mod=fav，把接口 URL 放到 `urls.txt`，每行一个 URL：

```text
https://example.com/api...
https://example.com/api...
```

运行：

```powershell
python .\download_app_video.py
```

输出示例：

```text
download/1_视频标题.mp4
download/enhanced/1_视频标题_720p.mp4
```

## 增强方式

默认使用 FFmpeg：

```ini
[upscale]
enhance_engine = ffmpeg
```

FFmpeg 增强速度快，主要做分辨率放大和轻锐化，不会真正恢复原视频中不存在的细节。

切换到 Real-ESRGAN：

```ini
[upscale]
enhance_engine = realesrgan
```

Real-ESRGAN 会逐帧处理视频，耗时和临时空间占用都明显更高。建议先用短片段测试效果，再处理完整视频。



仅供学习交流使用，严禁用于商业用途及非法活动，一切法律责任由使用者自行承担。
