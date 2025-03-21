import pygame
# 音频播放模块
def monkey():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load("./猴哥.mp3")
        pygame.mixer.music.play()
        print("\n🎵 任务完成音效播放中...")
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"\n⚠️ 音频播放异常: {str(e)}")

