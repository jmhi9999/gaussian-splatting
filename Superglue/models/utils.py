# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import time
from collections import OrderedDict
from threading import Thread
import numpy as np
import cv2
import torch
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
import random
from os import path
matplotlib.use('Agg')


class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1./total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()


class VideoStreamer:
    """ Class to help process image streams. Four types of possible inputs:"
        1.) USB Webcam.
        2.) An IP camera
        3.) A directory of images (files in directory matching 'image_glob').
        4.) A video file, such as an .mp4 or .avi file.
    """
    def __init__(self, basedir, resize, skip, image_glob, max_length=1000000):
        self._ip_grabbed = False
        self._ip_running = False
        self._ip_camera = False
        self._ip_image = None
        self._ip_index = 0
        self.cap = []
        self.camera = True
        self.video_file = False
        self.listing = []
        self.resize = resize
        self.interp = cv2.INTER_AREA
        self.i = 0
        self.skip = skip
        self.max_length = max_length
        if isinstance(basedir, int) or basedir.isdigit():
            print('==> Processing USB webcam input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(int(basedir))
            self.listing = range(0, self.max_length)
        elif basedir.startswith(('http', 'rtsp')):
            print('==> Processing IP camera input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.start_ip_camera_thread()
            self._ip_camera = True
            self.listing = range(0, self.max_length)
        elif Path(basedir).is_dir():
            print('==> Processing image directory input: {}'.format(basedir))
            self.listing = list(Path(basedir).glob(image_glob[0]))
            for j in range(1, len(image_glob)):
                image_path = list(Path(basedir).glob(image_glob[j]))
                self.listing = self.listing + image_path
            self.listing.sort()
            self.listing = self.listing[::self.skip]
            self.max_length = np.min([self.max_length, len(self.listing)])
            if self.max_length == 0:
                raise IOError('No images found (maybe bad \'image_glob\' ?)')
            self.listing = self.listing[:self.max_length]
            self.camera = False
        elif Path(basedir).exists():
            print('==> Processing video input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.listing = range(0, num_frames)
            self.listing = self.listing[::self.skip]
            self.video_file = True
            self.max_length = np.min([self.max_length, len(self.listing)])
            self.listing = self.listing[:self.max_length]
        else:
            raise ValueError('VideoStreamer input \"{}\" not recognized.'.format(basedir))
        if self.camera and not self.cap.isOpened():
            raise IOError('Could not read camera')

    def load_image(self, impath):
        """ Read image as grayscale and resize to img_size.
        Inputs
            impath: Path to input image.
        Returns
            grayim: uint8 numpy array sized H x W.
        """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        w, h = grayim.shape[1], grayim.shape[0]
        w_new, h_new = process_resize(w, h, self.resize)
        grayim = cv2.resize(
            grayim, (w_new, h_new), interpolation=self.interp)
        return grayim,(w,h)

    def next_frame(self):
        """ Return the next frame, and increment internal counter.
        Returns
             image: Next H x W image.
             status: True or False depending whether image was loaded.
        """

        if self.i == self.max_length:
            return (None, False)
        if self.camera:

            if self._ip_camera:
                #Wait for first image, making sure we haven't exited
                while self._ip_grabbed is False and self._ip_exited is False:
                    time.sleep(.001)

                ret, image = self._ip_grabbed, self._ip_image.copy()
                if ret is False:
                    self._ip_running = False
            else:
                ret, image = self.cap.read()
            if ret is False:
                print('VideoStreamer: Cannot get image from camera')
                return (None, False)
            w, h = image.shape[1], image.shape[0]
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])

            w_new, h_new = process_resize(w, h, self.resize)
            image = cv2.resize(image, (w_new, h_new),
                               interpolation=self.interp)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_file = str(self.listing[self.i])
            image,(w,h) = self.load_image(image_file)
        self.i = self.i + 1
        return (image, True,(w,h))

    def start_ip_camera_thread(self):
        self._ip_thread = Thread(target=self.update_ip_camera, args=())
        self._ip_running = True
        self._ip_thread.start()
        self._ip_exited = False
        return self

    def update_ip_camera(self):
        while self._ip_running:
            ret, img = self.cap.read()
            if ret is False:
                self._ip_running = False
                self._ip_exited = True
                self._ip_grabbed = False
                return

            self._ip_image = img
            self._ip_grabbed = ret
            self._ip_index += 1
            #print('IPCAMERA THREAD got frame {}'.format(self._ip_index))


    def cleanup(self):
        self._ip_running = False

# --- PREPROCESSING ---

def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def read_image(path, device, resize, rotation, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales


# --- GEOMETRY ---


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    assert rot <= 3
    h, w = image_shape[:2][::-1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array([[fy, 0., cy],
                         [0., fx, w-1-cx],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 2:
        return np.array([[fx, 0., w-1-cx],
                         [0., fy, h-1-cy],
                         [0., 0., 1.]], dtype=K.dtype)
    else:  # if rot == 3:
        return np.array([[fy, 0., h-1-cy],
                         [0., fx, cx],
                         [0., 0., 1.]], dtype=K.dtype)


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array([[np.cos(r), -np.sin(r), 0., 0.],
                  [np.sin(r), np.cos(r), 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float32)
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    scales = np.diag([1./scales[0], 1./scales[1], 1.])
    return np.dot(scales, K)


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2)
                    + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))
    return d


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs


# --- VISUALIZATION ---


def plot_image_pair(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size*n, size*3/4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw)
                 for i in range(len(kpts0))]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                       color, text, path, show_keypoints=False,
                       fast_viz=False, opencv_display=False,
                       opencv_title='matches', small_text=[]):

    if fast_viz:
        make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                                color, text, path, show_keypoints, 10,
                                opencv_display, opencv_title, small_text)
        return

    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()


def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)
    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


def error_colormap(x):
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1)
class Cell():
    def __init__(self,image0_idx_kpt):
        self.l = list()
        self.image0_idx_kpt = image0_idx_kpt
        #self.image1_idx_kpt = image1_idx_kpt
    def append(self,item):
        self.l.append(item)
def sortCell(val):
    return val['total_score']
def calcScores(tris,scores,valid_indices):
    all_new_scores = list()
    all_new_scores_sh = list()
    for t,s,v in zip(tris,scores,valid_indices):
        new_scores = torch.full_like(s,0.0)
        new_scores_sh = torch.full_like(s,0.0)
        for i,match in enumerate(t):    
            if v[i] == -1:
                continue
            for triangle in match.l:
                avg_score = triangle['score_da_as']
                avg_score_sh = triangle['score_da_as_sh']
                image_d_kpt_idx = triangle['image_d_kpt_idx']
                orig_score = triangle['score_sd']
                new_scores[0,i,image_d_kpt_idx] = avg_score
                new_scores_sh[0,i,image_d_kpt_idx] = avg_score_sh
        all_new_scores.append(new_scores)
        all_new_scores_sh.append(new_scores_sh)
    return all_new_scores,all_new_scores_sh
def create_triangles(image_s,   #source image
                    image_d,    #destination image
                    image_a,    #aux image
                    matching_sd,
                    matching_da,
                    matching_as,
                    margin=10):
    LINES = 5
    H2, W2 = 0,0
    DEBUG_PRINT = False
    H0, W0 = image_s.shape
    H1, W1 = image_d.shape
    H2, W2 = image_a.shape
    H, W = max(H0,H1)+H2, W0 + W1 + margin
    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image_s
    out[:H1, W0+margin:] = image_d
    H2_margin_w = int((W-W2)/2)
    out[max(H1,H2):,H2_margin_w:H2_margin_w+W2] = image_a
    out = np.stack([out]*3, -1)
    scores_sd_orig = matching_sd['full_scores']
    scores_as_orig = matching_as['full_scores']
    scores_da_orig = matching_da['full_scores']
    scores_sd_wo_sinkhorn_orig = matching_sd['full_scores_wo_sinkhorn']
    scores_da_wo_sinkhorn_orig = matching_da['full_scores_wo_sinkhorn']
    scores_as_wo_sinkhorn_orig = matching_as['full_scores_wo_sinkhorn']
    kpts_s = matching_sd['kpts_s']
    kpts_d = matching_sd['kpts_d']
    kpts_a = matching_as['kpts_a']
    
    COLOR_FACTOR = 2**(-4)
    
    kpts_s, kpts_d = np.round(kpts_s).astype(int), np.round(kpts_d).astype(int)
    kpts_a = np.round(kpts_a).astype(int)
    cnt = 0
    triangles_per_match = np.full((kpts_s.shape[0]),None)
    match_total_score = list()
    for START_KEY_POINT in range(len(kpts_s)):
        matches_lr = list()
        #01
        (x0, y0) = kpts_s[START_KEY_POINT]
        red = (0, 30, 250)
        #cv2.circle(out, (x0, y0), 3, red, -1, lineType=cv2.LINE_AA)
        scores_sd = scores_sd_orig[0,START_KEY_POINT,:]   #score_sd for enire START_KEY_POINT row
        scores_sd_wo_sinkhorn = scores_sd_wo_sinkhorn_orig[0,START_KEY_POINT,:]
        index_sorted_scores_sd = scores_sd.argsort()
        t = index_sorted_scores_sd.numpy()
        index_sorted_scores_sd = torch.from_numpy(t)[-1-LINES+1:]    #choosing the last #LINES scores 
        sorted_scores_sd = scores_sd[index_sorted_scores_sd]
        sorted_scores_sd_wo_sinkhorn = scores_sd_wo_sinkhorn[index_sorted_scores_sd]
        sorted_kpts_d = kpts_d[index_sorted_scores_sd]
        for i,(score,score_wo_sinkhorn) in enumerate( zip(sorted_scores_sd,sorted_scores_sd_wo_sinkhorn) ):
            if LINES == 1:
                (x1,y1) = sorted_kpts_d
            else:
                (x1, y1) = sorted_kpts_d[i]
            matches_lr.append({'image_s_kpt':(x0,y0),'score_sd':score,
            'image_d_kpt':(x1, y1),'image_s_kpt_idx':START_KEY_POINT,
            'image_d_kpt_idx':index_sorted_scores_sd[i],
            'score_sd_wo_skinhorn':score_wo_sinkhorn})
        #12
        top_matches = Cell(START_KEY_POINT)
        for match in matches_lr:
            max_score = 0.0
            idx_kpt_image_d = match['image_d_kpt_idx']
            idx_kpt_image_s = match['image_s_kpt_idx']
            for kpt_idx_image_a,kpt_image_a in enumerate(kpts_a):
                #score_da
                score_da = scores_da_orig[0,idx_kpt_image_d,kpt_idx_image_a]
                score_da_sh = scores_da_wo_sinkhorn_orig[0,idx_kpt_image_d,kpt_idx_image_a]
                #score_as
                score_as = scores_as_orig[0,kpt_idx_image_a,idx_kpt_image_s]
                score_as_sh = scores_as_wo_sinkhorn_orig[0,kpt_idx_image_a,idx_kpt_image_s]
                if max_score < score_da*score_as:
                    max_score = score_da*score_as
                    max_score_sh = score_da_sh*score_as_sh
                    max_score_da = score_da
                    max_score_da_sh = score_da_sh
                    max_kpt_idx_image_a = kpt_idx_image_a
                    max_score_as = score_as
                    max_score_as_sh = score_as_sh
            match['image_a_kpt'] = kpts_a[max_kpt_idx_image_a]
            match['score_da'] = max_score_da
            match['score_as'] = max_score_as
            match['score_da_as'] = np.sqrt(np.sqrt(max_score_da*max_score_as)*score)
            match['score_da_as_sh'] = max_score_da_sh*max_score_as_sh
            match['score_da_sh'] = max_score_da_sh
            match['score_as_sh'] = max_score_as_sh
            #save largest score
            top_matches.append(match)
        triangles_per_match[match['image_s_kpt_idx']] = top_matches
        if DEBUG_PRINT:
            cnt+=1
            print(f'{cnt}: image_s_kpt:{match["image_s_kpt"]} image_d_kpt:{match["image_d_kpt"]} image_a_kpt:{match["image_a_kpt"]}')
        if DEBUG_PRINT:
            cnt = 0
            print('-'*20)
    return triangles_per_match
def avg_dist(triangles,warped_kpts,valid_indices):
    dist = 0.0
    cnt = 0
    for i,routs in enumerate(triangles):
        if valid_indices[i] == -1:
            continue
        best_match = routs.l[-1]  #last one ist the best
        kpt_d = best_match['image_d_kpt']
        dist +=  np.sqrt(np.dot(kpt_d-warped_kpts[i],kpt_d-warped_kpts[i]))
        cnt+=1
    return dist,cnt
def write_warped_kpts(kpts,warped_kpts,file_dest):
    with open(file_dest,'w',encoding='utf-8') as f:
        for kpt,warped_kpt in zip(kpts,warped_kpts):
            f.write(f'({kpt[0]},{kpt[1]}),({warped_kpt[0]},{warped_kpt[1]})')
def write_to_file(text_list,file_dest):
    if not path.exists(file_dest):
        with open(file_dest,'a',encoding='utf-8') as f:
            #title
            f.write('index , score, avg, score_wo_skinhorn, image0 kpts_x,image0 kpts_y, image1 kpts_x,image1 kpts_y, warped image1 kpts_x,warped image1 kpts_y\n')    
    with open(file_dest,'a',encoding='utf-8') as f:
        for d in text_list:
            f.write(f"{d['idx']}, ")
            f.write(f"{d['score_sd']}, ")
            f.write(f"{d['avg']}, ")
            f.write(f"{d['score_sd_wo_skinhorn']},  ")
            f.write(f"{d['image_s_kpt'][0]},{d['image_s_kpt'][1]}, ")
            f.write(f"{d['image_d_kpt'][0]},{d['image_d_kpt'][1]}, ")
            f.write(f"{d['warped_image_d_kpt'][0]},{d['warped_image_d_kpt'][1]}\n")
def load_H(file_name):
    with open(file_name,'r',encoding='utf-8') as f:
        H = np.empty((0,3),dtype=float)
        for line in f:
            row_txt = line.split(' ')
            row = [float(item) for item in row_txt]
            row = np.array(row)
            row = np.reshape(row,[1,3])
            H = np.append(H,row,axis=0)
    return H
def draw_match(image0,image1,orig_match,imp_match,warped_kpt):
    image0_kpt = orig_match['kpts_s']
    orig_image1_kpt = orig_match['kpts_d']
    imp_image1_kpt = imp_match['kpts_d']
    H, W = image0.shape
    out = 255*np.ones((H, W*2), np.uint8)
    out[:H, :W] = image0
    out[:H, W:] = image1
    out = np.stack([out]*3, -1)

    red = (0, 30, 250)
    blue = (250,10,10)
    green = (10,250,10)
    dot_size = 2
    (x0,y0) = image0_kpt
    (warped_x1,warped_y1) = int(warped_kpt[0]),int(warped_kpt[1])
    cv2.circle(out, (x0, y0), dot_size, red, -1, lineType=cv2.LINE_AA)
    (x1,y1) = int(orig_image1_kpt[0]),int(orig_image1_kpt[1])
    cv2.circle(out, (x1+W, y1), dot_size, blue, -1, lineType=cv2.LINE_AA)
    cv2.line(out, (x0, y0), (x1+W, y1),
                color=green, thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(out, (warped_x1+W, warped_y1), dot_size, green, -1, lineType=cv2.LINE_AA)
    (x1,y1) = int(imp_image1_kpt[0]),int(imp_image1_kpt[1])
    cv2.circle(out, (x1+W, y1), dot_size, red, -1, lineType=cv2.LINE_AA)
    cv2.line(out, (x0, y0), (x1+W, y1),
                color=green, thickness=1, lineType=cv2.LINE_AA)
    return out
def draw_triangles(tris,warped_kpts,kpt_idx,image_s,image_d,image_a,margin=10):
    KEY_POINT = kpt_idx
    cell = tris[KEY_POINT]
    matches = cell.l
    H2, W2 = 0,0
    H0, W0 = image_s.shape
    H1, W1 = image_d.shape
    H2, W2 = image_a.shape
    H, W = max(H0,H1)+H2, W0 + W1 + margin
    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image_s
    out[:H1, W0+margin:] = image_d
    H2_margin_w = int((W-W2)/2)
    out[max(H1,H2):,H2_margin_w:H2_margin_w+W2] = image_a
    out = np.stack([out]*3, -1)
    ls = np.linspace(0.1,0.9,5)
    text_matches = list()
    for match_idx,match in enumerate(matches):
        avg_score = match['score_da_as']
        score_sd_rounded = round(np.asscalar(match['score_sd'].numpy()),2)
        avg_score_rounded = round(np.asscalar(avg_score.numpy()),2)
        score_wo_skinhon_rounded = round(np.asscalar(match['score_sd_wo_skinhorn'].numpy()),2)
        idx_kpt = match['image_s_kpt_idx']
        colors = cm.jet(ls)
        color = colors[match_idx]
        c = (np.array(color)*255).astype(int)
        c = c.tolist()
        (x0,y0) = match['image_s_kpt']
        (x1,y1) = match['image_d_kpt']
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                color=c, thickness=1, lineType=cv2.LINE_AA)
        #print('-'*20)
        #print(f'image_s_kpt:({x0},{y0}) image_d_kpt:({x1},{y1}) score_sd:{score_sd_rounded}')
        warped_x1 = int(warped_kpts[idx_kpt,0])
        warped_y1 = int(warped_kpts[idx_kpt,1])
        #print(f'warp: image_s_kpt:({x0},{y0}) idx:{KEY_POINT} image_d_kpt:({warped_x1},{warped_y1})')
        d = {'image_s_kpt':(x0,y0),'image_d_kpt':(x1,y1),
        'idx':KEY_POINT,'warped_image_d_kpt':(warped_x1,warped_y1),
        'score_sd':score_sd_rounded,
        'avg':avg_score_rounded,
        'score_sd_wo_skinhorn':score_wo_skinhon_rounded}
        text_matches.append(d)
        red = (0, 30, 250)
        blue = (250,10,10)
        green = (10,250,10)
        dot_size = 2
        cv2.circle(out, (x0, y0), dot_size, red, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), dot_size, blue, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (warped_x1 + margin + W0, warped_y1), dot_size, green, -1, lineType=cv2.LINE_AA)
        (x0,y0) = match['image_d_kpt']
        (x1,y1) = match['image_a_kpt']
        cv2.line(out, (x0 + margin + W0, y0), (x1+H2_margin_w, y1+max(H0,H1)),
                color=c, thickness=1, lineType=cv2.LINE_AA)
        (x0,y0) = match['image_a_kpt']
        (x1,y1) = match['image_s_kpt']
        cv2.line(out, (x0+H2_margin_w, y0+max(H0,H1)), (x1, y1),
                color=c, thickness=1, lineType=cv2.LINE_AA)
        # Scale factor for consistent visualization across scales.
        sc = min(H / 800., 2.0)

        # Big text.
        Ht = int(25 * sc)  # text height
        txt_color_fg = c
        txt_color_bg = (0, 0, 0)
        C = 200
        avg_score_text = str(avg_score_rounded)
        orig_score_text = str(score_sd_rounded)
        wo_skinhorn_score_text = str(score_wo_skinhon_rounded)
        for text_idx, t in enumerate(orig_score_text):
            cv2.putText(out, t, (int(13*(sc+text_idx)), C+Ht*(match_idx+3)), cv2.FONT_HERSHEY_DUPLEX,
                        1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
            cv2.putText(out, t, (int(13*(sc+text_idx)), C+Ht*(match_idx+3)), cv2.FONT_HERSHEY_DUPLEX,
                        1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

        for text_idx, t in enumerate(avg_score_text):
            cv2.putText(out, t, (int(13*((sc+12)+sc+text_idx)), C+Ht*(match_idx+3)), cv2.FONT_HERSHEY_DUPLEX,
                        1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
            cv2.putText(out, t, (int(13*((sc+12)+sc+text_idx)), C+Ht*(match_idx+3)), cv2.FONT_HERSHEY_DUPLEX,
                        1.0*sc, txt_color_fg, 1, cv2.LINE_AA)
        #before skinhorn
        for text_idx, t in enumerate(wo_skinhorn_score_text):
            cv2.putText(out, t, (int(+12*((sc+24)+sc+text_idx)), C+Ht*(match_idx+3)), cv2.FONT_HERSHEY_DUPLEX,
                        1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
            cv2.putText(out, t, (int(12*((sc+24)+sc+text_idx)), C+Ht*(match_idx+3)), cv2.FONT_HERSHEY_DUPLEX,
                        1.0*sc, txt_color_fg, 1, cv2.LINE_AA)
        
        #title
        title_text = 'original | averaged'
        txt_color_fg = (200, 200, 50)
        for text_idx, t in enumerate(list(title_text)):
            cv2.putText(out, t, (int(12*(sc+text_idx)), C+Ht), cv2.FONT_HERSHEY_DUPLEX,
                        1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
            cv2.putText(out, t, (int(12*(sc+text_idx)), C+Ht), cv2.FONT_HERSHEY_DUPLEX,
                        1.0*sc, txt_color_fg, 1, cv2.LINE_AA)
    return out,text_matches

def warp(source_image_kpts,H):
    #H_t = np.transpose(H,(1,0))
    row_size = source_image_kpts.shape[0]
    source_image_kpts_homogenous = np.append(source_image_kpts,np.ones((row_size,1)),axis=1)
    source_image_kpts_homogenous_t = np.transpose(source_image_kpts_homogenous,(1,0))
    homogen = lambda x: x[:-1]/x[-1]
    out = np.matmul(H,source_image_kpts_homogenous_t)
    warped_kpts = np.transpose(out,(1,0))
    warped_kpts = np.array([homogen(x) for x in warped_kpts])
    
    return warped_kpts

def make_matching_plot_one_to_many(image0,
                            image1,
                            image2,
                            matching01,
                            matching12,
                            matching20,
                            margin=10,
                            path=None):
    START_KEY_POINT = 115
    LINES = 10
    H2, W2 = 0,0
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H2, W2 = image2.shape
    H, W = max(H0,H1)+H2, W0 + W1 + margin
    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    H2_margin_w = int((W-W2)/2)
    out[max(H1,H2):,H2_margin_w:H2_margin_w+W2] = image2
    out = np.stack([out]*3, -1)
    scores01 = matching01['full_scores']
    scores20 = matching20['full_scores']
    scores12 = matching12['full_scores']
    kpts01_0 = matching01['kpts_s']
    kpts01_1 = matching01['kpts_d']
    kpts20_0 = matching20['kpts_d']
    kpts20_2 = matching20['kpts_s']
    kpts12_1 = matching12['kpts_s']
    kpts12_2 = matching12['kpts_d']

    COLOR_FACTOR = 2**(-4)
    #color01 = cm.jet(scores01[0,KEY_POINT,:]**COLOR_FACTOR)
    #color20 = cm.Reds(scores20[0,KEY_POINT,:]**COLOR_FACTOR)
    #color12 = cm.Reds(scores12[0,KEY_POINT,:]**COLOR_FACTOR)
    
    #color01 = (np.array(color01)*255).astype(int)[:, ::-1]
    #color20 = (np.array(color20)*255).astype(int)[:, ::-1]
    #color12 = (np.array(color12)*255).astype(int)[:, ::-1]
    kpts01_0, kpts01_1 = np.round(kpts01_0).astype(int), np.round(kpts01_1).astype(int)
    kpts20_0, kpts20_2 = np.round(kpts20_0).astype(int), np.round(kpts20_2).astype(int)
    kpts12_1, kpts12_2 = np.round(kpts12_1).astype(int), np.round(kpts12_2).astype(int)
    #01
    (x0, y0) = kpts01_0[START_KEY_POINT]
    scores01 = scores01[0,START_KEY_POINT,:]
    index_sorted_scores01 = scores01.argsort()
    t = index_sorted_scores01.numpy()
    t = t[::-1].copy()
    index_sorted_scores01 = torch.from_numpy(t)[:LINES]
    sorted_scores01 = scores01[index_sorted_scores01]
    color01 = cm.jet(sorted_scores01**COLOR_FACTOR)
    color01 = (np.array(color01)*255).astype(int)
    sorted_kpts01_1 = kpts01_1[index_sorted_scores01]
    red = (0, 30, 250)
    cv2.circle(out, (x0, y0), 3, red, -1, lineType=cv2.LINE_AA)
    for i,(_,c) in enumerate(zip(sorted_scores01,color01)):
        if LINES == 1:
            (x1,y1) = sorted_kpts01_1
        else:
            (x1, y1) = sorted_kpts01_1[i]
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        print(f'from image0 ({x0},{y0}) to image1 ({x1},{y1})')
    #12
    print('-'*20)
    index_kpts_dest = list()
    for idx in index_sorted_scores01[:LINES]:
        kpts1 = kpts01_1[idx]
        scores12_kpt = scores12[0][idx,:]
        index_sorted_scores12 = scores12_kpt.argsort()
        t = index_sorted_scores12.numpy()
        t = t[::-1][:LINES].copy()
        index_sorted_scores12 = torch.from_numpy(t)
        scores12_sorted = scores12_kpt[index_sorted_scores12]
        color12 = cm.jet(scores12_sorted**COLOR_FACTOR)
        color12 = (np.array(color12)*255).astype(int)
        sorted_kpts12_2 = kpts12_2[index_sorted_scores12]
        index_kpts_dest.append(index_sorted_scores12[0])
        (x0, y0) = kpts1
        for i,(_,c) in enumerate(zip(scores12_sorted,color12)):
            if LINES == 1:
                (x1, y1) = sorted_kpts12_2
            else:    
                (x1, y1) = sorted_kpts12_2[i]
            c = c.tolist()
            cv2.line(out, (x0 + margin + W0, y0), (x1+H2_margin_w, y1+max(H0,H1)),
                    color=c, thickness=1, lineType=cv2.LINE_AA)
            print(f'from image1 ({x0},{y0}) to image2 ({x1},{y1})')
            break
    #20
    print('-'*20)
    for idx in index_kpts_dest:
        kpts2 = kpts12_2[idx]
        scores20_kpt = scores20[0][idx,:]
        index_scores20_sorted = scores20_kpt.argsort()
        t = index_scores20_sorted.numpy()
        t = t[::-1][:LINES].copy()
        index_scores20_sorted = torch.from_numpy(t)
        scores20_sorted = scores20_kpt[index_scores20_sorted]
        color20 = cm.jet(scores20_sorted**COLOR_FACTOR)
        color20 = (np.array(color20)*255).astype(int)
        sorted_kpts20_0 = kpts20_0[index_scores20_sorted]
        (x0, y0) = kpts2
        for i,(_,c) in enumerate(zip(scores20_sorted,color20)):
            if LINES == 1:
                (x1, y1) = sorted_kpts20_0
            else:
                (x1, y1) = sorted_kpts20_0[i]
            c = c.tolist()
            cv2.line(out, (x0+H2_margin_w, y0+max(H0,H1)), (x1, y1),
                    color=c, thickness=1, lineType=cv2.LINE_AA)
            print(f'from image2 ({x0},{y0}) to image0 ({x1},{y1})')
            break
    if path is not None:
        cv2.imwrite(str(path), out)
    opencv_display = False
    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out