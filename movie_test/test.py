import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

def make_frame(t):
    # 座標(t,t)に長方形を描画
    fig, ax = plt.subplots(figsize=(6,6))
    ax.add_patch(Rectangle((t, t), 1,2))
    ax.set_xlim(0,10)
    ax.set_ylim(0,10)
    # bitmapに変換. bmp.shape==(432,432,3)
    bmp = mplfig_to_npimage(fig)
    plt.close()
    return bmp

fname = "output/out2.mp4"
# 10秒, 30fps の計300フレームからなる動画を生成
animation = VideoClip(make_frame, duration = 10)
animation.write_videofile(fname, fps = 30)