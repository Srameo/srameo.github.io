import rawpy
import numpy as np
import os
import glob
import cv2
import exifread


Sony_A7S2_CCM = np.array([[ 1.9712269,-0.6789218, -0.29230508],
                          [-0.29104823, 1.748401 , -0.45735288],
                          [ 0.02051281,-0.5380369,  1.5175241 ]],
                         dtype='float32')


def pack_raw_bayer(raw: np.ndarray, raw_pattern: np.ndarray):
    #pack Bayer image to 4 channels
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)

    raw = raw.astype(np.uint16)
    H, W = raw.shape
    if H % 2 == 1:
        raw = raw[:-1]
    if W % 2 == 1:
        raw = raw[:, :-1]
    out = np.stack((raw[R[0][0]::2,  R[1][0]::2], #RGBG
                    raw[G1[0][0]::2, G1[1][0]::2],
                    raw[B[0][0]::2,  B[1][0]::2],
                    raw[G2[0][0]::2, G2[1][0]::2]), axis=0).astype(np.uint16)

    return out


def depack_raw_bayer(raw: np.ndarray, raw_pattern: np.ndarray):
    _, H, W = raw.shape
    raw = raw.astype(np.uint16)

    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)

    raw_flatten = np.zeros((H * 2, W * 2))
    raw_flatten[R[0][0]::2,  R[1][0]::2] = raw[0]
    raw_flatten[G1[0][0]::2,  G1[1][0]::2] = raw[1]
    raw_flatten[B[0][0]::2,  B[1][0]::2] = raw[2]
    raw_flatten[G2[0][0]::2,  G2[1][0]::2] = raw[3]

    raw_flatten = raw_flatten.astype(np.uint16)
    return raw_flatten


def metainfo(rawpath):
    with open(rawpath, 'rb') as f:
        tags = exifread.process_file(f)
        _, suffix = os.path.splitext(os.path.basename(rawpath))

        if suffix == '.dng':
            expo = eval(str(tags['Image ExposureTime']))
            iso = eval(str(tags['Image ISOSpeedRatings']))
        else:
            expo = eval(str(tags['EXIF ExposureTime']))
            iso = eval(str(tags['EXIF ISOSpeedRatings']))

        # print('ISO: {}, ExposureTime: {}'.format(iso, expo))
    return iso, expo

def read_img(raw_path):
    raw = rawpy.imread(raw_path)
    raw_vis = raw.raw_image_visible.copy()
    raw_pattern = raw.raw_pattern
    black_level = np.array(raw.black_level_per_channel, dtype=np.float32).reshape(1, 4, 1, 1)
    white_level_data = raw.camera_white_level_per_channel
    if white_level_data is None:
        white_level_data = [raw.white_level]
    
    white_level = np.array(white_level_data, dtype=np.float32)
    if white_level.size == 1:
        white_level = white_level.repeat(4, 0)
    white_level = white_level.reshape(1, 4, 1, 1)
    raw_packed = np.array(pack_raw_bayer(raw_vis, raw_pattern), dtype=np.float32)[np.newaxis]
    return raw, raw_pattern, raw_packed, black_level, white_level


def postprocess(raw, raw_pattern, im, bl, wl, output_bps=8):
    im = im * (wl - bl) + bl
    im = im[0]
    im = depack_raw_bayer(im, raw_pattern)
    H, W = im.shape
    raw.raw_image_visible[:H, :W] = im
    rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=output_bps)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb

if __name__ == "__main__":
    SAVE_PATH = '/Users/jinxin/workspace/homepage/srameo.github.io/projects/ultraled/assets/image_demo'
    im_paths = ['/Users/jinxin/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/781dc9afb0e65f36be7fc847a73f76a4/Message/MessageTemp/c33fcc0815207bf3145cf63597419fea/File/pic_for_demo/DSC01176.ARW']
    # im_paths = glob.glob('/Users/jinxin/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/781dc9afb0e65f36be7fc847a73f76a4/Message/MessageTemp/c33fcc0815207bf3145cf63597419fea/File/pic_for_demo/*.ARW')
    for im_path in im_paths:
        im_name = im_path.split("/")[-1].split(".")[0]
        print(im_name)
        ratio = 40.0
        raw_path = im_path
        raw, raw_pattern, raw_packed, black_level, white_level = read_img(raw_path)
        raw_packed = (raw_packed - black_level) / (white_level - black_level)
        rgb = postprocess(raw, raw_pattern, raw_packed, black_level, white_level)
        rgb = cv2.resize(rgb, (1757, 1172), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f'{SAVE_PATH}/{im_name}_in.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 75])
        raw_packed = (raw_packed * ratio).clip(0, 1)
        rgb = postprocess(raw, raw_pattern, raw_packed, black_level, white_level)
        rgb = cv2.resize(rgb, (1757, 1172), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f'{SAVE_PATH}/{im_name}_in_noisy.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 75])

        im_path = im_path + '.png'
        rgb = cv2.imread(im_path)
        rgb = cv2.resize(rgb, (1757, 1172), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f'{SAVE_PATH}/{im_name}_out.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])