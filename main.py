#!/usr/bin/env python
"""视频实时插帧播放器 — 入口。

用法:
    python main.py                 # 启动后打开文件
    python main.py 视频.mp4        # 直接播放指定视频
"""

import sys

from player.app import main

if __name__ == "__main__":
    sys.exit(main(sys.argv))
