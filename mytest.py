import tensorflow as tf
import os
from glob import glob

eval_low_name = []
eval_low_data_name = glob('./results/eval_result/*')
eval_low_data_name.sort()

eval_high_name = []
eval_high_data_name = glob('./LOLdataset/eval15/high/*')
eval_high_data_name.sort()
PSNR = 0
SSIM = 0
for idx in range(len(eval_high_data_name)):
    [_, name] = os.path.split(eval_high_data_name[idx])
    suffix = name[name.find('.') + 1:]
    name = name[:name.find('.')]
    eval_low_name.append("./results/eval_result/" + name + "_kindle.png")
    eval_high_name.append("./LOLdataset/eval15/high/" + name + ".png")
    img1 = tf.image.decode_image(tf.io.read_file(eval_low_name[idx]))
    img2 = tf.image.decode_image(tf.io.read_file(eval_high_name[idx]))
    img1 = tf.image.convert_image_dtype(img1, tf.float32)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)
    _SSIM_ = tf.image.ssim(img1, img2, max_val=1.0)
    _PSNR_ = tf.image.psnr(img1, img2, max_val=1.0)
    with tf.Session() as sess:
        SSIM += sess.run(_SSIM_)
        PSNR += sess.run(_PSNR_)
print("SSIM：", (float(SSIM) / 15) * 100)
print("PSNR：", (float(PSNR) / 15))

