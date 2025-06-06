import os
import cv2
import numpy as np

def dark_channel(img, size=15):
    r, g, b = cv2.split(img)
    min_img = cv2.min(r, cv2.min(g, b))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dc_img = cv2.erode(min_img, kernel)
    return dc_img

def get_atmo(img, percent=0.001):
    dark = dark_channel(img)
    flat_img = img.reshape(-1, 3)
    flat_dark = dark.reshape(-1)
    num = int(max(img.shape[0] * img.shape[1] * percent, 1))
    indices = flat_dark.argsort()[-num:]
    atmo = np.mean(flat_img[indices], axis=0)
    return atmo

def get_trans(img, atom, w=0.95):
    normed = img / atom
    t = 1 - w * dark_channel(normed)
    return t

def refine_transmission(img, trans, r=40, eps=1e-3):
    # 引导滤波细化透射率
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    trans = trans.astype(np.float32)
    refined = cv2.ximgproc.guidedFilter(guide=gray, src=trans, radius=r, eps=eps)
    return refined

def dehaze_img(img):
    img = img.astype('float64') / 255
    atom = get_atmo(img)
    trans = get_trans(img, atom)
    trans = np.clip(trans, 0.1, 1)
    # 引导滤波细化
    try:
        import cv2.ximgproc
        trans = refine_transmission(img, trans)
        trans = np.clip(trans, 0.1, 1)
    except Exception as e:
        # 如果没有ximgproc模块则跳过细化
        pass
    result = np.empty_like(img)
    for i in range(3):
        result[:, :, i] = (img[:, :, i] - atom[i]) / trans + atom[i]
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)

def batch_dehaze_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        save_dir = os.path.join(output_dir, rel_path)
        os.makedirs(save_dir, exist_ok=True)
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, fname)
                img = cv2.imread(img_path)
                if img is not None:
                    dehazed = dehaze_img(img)
                    cv2.imwrite(os.path.join(save_dir, fname), dehazed)

if __name__ == '__main__':
    # 训练集和验证集图片去雾
    batch_dehaze_images('cityscapes/leftImg8bit/train', 'cityscapes/leftImg8bit_dehaze/train')
    batch_dehaze_images('cityscapes/leftImg8bit/val', 'cityscapes/leftImg8bit_dehaze/val')