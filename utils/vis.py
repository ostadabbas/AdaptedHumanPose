'''
for key points visualization. Also visualizer for visdom class.
'''

import os
import os.path as osp
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import re

import sys
import ntpath
import time
from . import utils_tool, html
from subprocess import Popen, PIPE
# from scipy.misc import imresize
from collections import OrderedDict
from skimage.transform import resize # misc deprecated e
import glob
import datetime
from tqdm import tqdm
from scipy.ndimage import zoom
from sklearn.manifold import TSNE
import seaborn as sns
sns.set_style('whitegrid')
import json
import utils.utils_tool as ut_t

def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    '''

    :param img:
    :param kps: 3 * n_jts
    :param kps_lines:
    :param kp_thresh:
    :param alpha:
    :return:
    '''
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None, input_shape=(256, 256), if_dsFmt=True):
    # worked mainly for ds format with range set properly
    # vis with x, z , -y
    # plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[0], c[1], c[2])) for c in colors]     #  array list

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=[colors[l]], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=[colors[l]], marker='o')

    x_r = np.array([0, input_shape[1]], dtype=np.float32)
    y_r = np.array([0, input_shape[0]], dtype=np.float32)
    z_r = np.array([0, 1], dtype=np.float32)
    
    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    if if_dsFmt:        # if ds format , then form it this way
        ax.set_xlim([0, input_shape[1]])
        ax.set_ylim([0,1])
        ax.set_zlim([-input_shape[0],0])
    # ax.legend()

    plt.show()
    cv2.waitKey(0)

def vis_entry(entry_dict):
    '''
    from the entry dict plot the images
    :param entry_dict:
    :return:
    '''

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk. Also to webpage

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = utils_tool.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = resize(im, (h, int(w * aspect_ratio)))
        if aspect_ratio < 1.0:
            im = resize(im, (int(h / aspect_ratio), w))
        utils_tool.save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    display_id -> loss; +1 -> images +2-> text +3 metrics
    """

    def __init__(self, opts):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opts = opts  # cache the option
        self.display_id = opts.display_id
        self.use_html = opts.use_html       #
        self.win_size = opts.display_winsize
        self.name = opts.name
        self.port = opts.display_port
        self.saved = False
        self.clipMode = opts.clipMode  # 01 or 11
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = opts.display_ncols
            self.vis = visdom.Visdom(server=opts.display_server, port=opts.display_port, env=opts.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = opts.web_dir
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            utils_tool.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opts.web_dir, 'loss_log.txt')  # put this loss in result at this time to avoid overlapping
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def load(self, epoch):
        if osp.exists(osp.join(self.opts.vis_dir, 'vis_{}.npy'.format(epoch))):
            attr_dict = np.load(osp.join(self.opts.vis_dir, 'vis_{}.npy'.format(epoch)), allow_pickle=True).item()
            for key in attr_dict:
                if attr_dict[key]:
                    setattr(self, key, attr_dict[key])
        else:
            print('loading visualizer {} failed, start from scratch'.format(epoch))

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result, if_bchTs=False):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols  #controlled to 4 columns so more not show here
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]   # create iterator then goes to next
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    if if_bchTs:
                        image_numpy = utils_tool.tensor2im(image, clipMod=self.clipMode) # 1st in batch
                    else:
                        image_numpy = image     # directly use current
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1])) # channel first
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    # self.vis.text(table_css + label_html, win=self.display_id + 2, opts=dict(title=title + ' labels'))        # not useful
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        if if_bchTs:
                            image_numpy = utils_tool.tensor2im(image, clipMod = self.clipMode)
                        else:
                            image_numpy = image
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = utils_tool.tensor2im(image, clipMod=self.clipMode)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                utils_tool.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = utils_tool.tensor2im(image, clipMod=self.clipMode)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}      # legend from very beginning
        self.plot_data['X'].append(epoch + counter_ratio)
        # check if input has same dim as previosu  othervise, fill the last value to it.
        if self.plot_data['Y']:     # if there is any data
            if len(self.plot_data['Y'][-1]) > len(losses):     # more losses before only decrese case, increase no done yet
                appd_Y = self.plot_data['Y'][-1]
                lgd = self.plot_data['legend']
                for k in losses:
                    appd_Y[lgd.index(k)] = losses[k]        # fill the missing  Y
            else:   # same length append directly
                appd_Y = [losses[k] for k in self.plot_data['legend']]
        else:
            appd_Y = [losses[k] for k in self.plot_data['legend']]      # give full losses list

        self.plot_data['Y'].append(appd_Y)   # plotdata{Y: [ [l1] [l2];  ]  } each column
        try:
            if len(self.plot_data['legend']) < 2:
                # X = np.expand_dims(np.array(self.plot_data['X']), axis=1)
                X = np.array(self.plot_data['X'])
                Y = np.array(self.plot_data['Y'])
                if Y.size>1:
                    Y = Y.squeeze()
            else:
                X = np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1)
                Y = np.array(self.plot_data['Y'])
            self.vis.line(
                # X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                X=X,
                Y=Y,
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()
    def plot_metrics(self, epoch, evals):  # at the end of each epoch plot metrics
        """display the current metrics. use display_id + 3

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            evals (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'evals'):
            self.evals = {'X': [], 'Y': [], 'legend': list(evals.keys())}
        self.evals['X'].append(epoch)
        self.evals['Y'].append([evals[k] for k in self.evals['legend']])
        try:
            if len(self.evals['legend']) < 2:
                # X = np.expand_dims(np.array(self.plot_data['X']), axis=1)
                X = np.array(self.evals['X'])
                Y = np.array(self.evals['Y'])
                if Y.size>1:
                    Y = Y.squeeze()
            else:
                X = np.stack([np.array(self.evals['X'])] * len(self.evals['legend']), 1)
                Y = np.array(self.evals['Y'])
            self.vis.line(
                X=X,
                Y=Y,
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.evals['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'evals'},
                win=self.display_id+3)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message


def tsne_plot(folder, output_file_path, ds_li, max_num_samples_per_dataset=30, scaling=[0.5, 0.5, 0.5], sz_font=20, if_prof=False):
    '''
    history: 08.25.20, add li_ds to control,  all G_fts number floor(len(G_file)/maxnumber) to sample it.
    next: save t-sne out, ft_tsne xN, label xN ,  f_nm xN
    folder: "vis/train"
    output_file_path: /path/to/plot.png
    max_num_samples_per_dataset:
    scaling: scaling factors.
        example. if features are (2048, 8, 8) and scaling is (0.5, 0.75, 0.25),
        then features become (1024, 6, 2)
    :param if_prof: not really run, just check the the number in each ds .
    '''
    # print(
    #     '==== [{}] IMPORTANT. GENERATING TEST PLOT TO {} TO VERIFY VALID DESTINATION BEFORE GOING THROUGH COMPUTATIONS'.format(
    #         datetime.datetime.now(), output_file_path))
    # sns.scatterplot(x=[1, 2], y=[1, 2]).get_figure().savefig(output_file_path)
    # print('==== [{}] Output figure path validated. Continuing with calculations.'.format(datetime.datetime.now()))

    # datasets = os.listdir(folder)  # ['Human36M', ...]
    datasets = ds_li

    # Load data
    all_G = []
    labels = []
    idxs = []

    print('==== [{}] Loading files from {} datasets'.format(datetime.datetime.now(), len(datasets)))

    for dataset in datasets: #
        feature_folder = os.path.join(folder, dataset, "G_fts_raw")
        numpy_files = glob.glob(os.path.join(feature_folder, "*npy"))
        # np.random.shuffle(numpy_files)    # no use shuffle to fix performance
        n_file = len(numpy_files)
        print('{} holds {} files'.format(dataset, n_file))
        step = int(n_file/float(max_num_samples_per_dataset))     # floor it
        if not if_prof:  # if not for profiling
            for file in tqdm(numpy_files[:step*max_num_samples_per_dataset:step], desc=dataset):
                x = np.load(file)
                assert x.shape == (2048, 8, 8)
                all_G.append(x)     # the G features
                # keep the file name to another list, replace p
                if '_p' in dataset:
                    labels.append(dataset[:-3]) # get rid of -p2 thing
                else:
                    labels.append(dataset)
            str_idx = int(file.split('/')[-1][:-4])        # get file name
            idxs.append(str_idx)   # keep idx
    if not if_prof: # if not for profiling
        print('==== [{}] Done loading files. Loaded {} samples.'.format(datetime.datetime.now(), len(all_G)))

        # Reshape
        print('==== [{}] Downsampling features'.format(datetime.datetime.now()))
        all_G = zoom(all_G, (1,) + tuple(scaling))
        print('==== [{}] Done downsampling. Current shape: {}'.format(datetime.datetime.now(), np.shape(all_G)))

        print('==== [{}] Reshaping feature array'.format(datetime.datetime.now()))
        new_shape = (len(all_G), np.prod(np.shape(all_G)[1:]))  # N x n_fts
        all_G = np.reshape(all_G, new_shape).astype(float)
        print('==== [{}] Done reshaping. Current shape: {}'.format(datetime.datetime.now(), np.shape(all_G)))

        # Run t-SNE
        print('==== [{}] Running t-SNE'.format(datetime.datetime.now()))
        model = TSNE(n_components=2)
        output = model.fit_transform(all_G)

        # Plot
        print('==== [{}] Plotting and saving figure'.format(datetime.datetime.now()))
        snsplot = sns.scatterplot(x=output[:, 0], y=output[:, 1], hue=labels, alpha=0.7)
        plt.setp(snsplot.get_legend().get_texts(), fontsize=str(sz_font))  # increase size
        snsplot.get_figure().savefig(output_file_path, dpi=300)
        plt.cla()
        print('==== [{}] Figure saved to {}.'.format(datetime.datetime.now(), output_file_path))

        rst = OrderedDict()
        rst['fts_tsne'] = output.tolist()       # the translated tsne features
        rst['labels'] = labels  # string        # idx of 4 sets
        rst['idxs'] = idxs  # st        # the idx number of the image
        # rst_fd = osp.join(folder, 'tsne_rst')
        # if not osp.exists(rst_fd):
        #     os.makedirs(rst_fd)
        pth_tsne = osp.join(folder, 'tsne_rst.json')
        print('==== [{}] tsne saved to {}.'.format(datetime.datetime.now(), pth_tsne))
        with open(osp.join(folder, 'tsne_rst.json'), 'w') as f:       # can be reploting with rename
            json.dump(rst, f)
            f.close()

def cmb2d3d(set_fd, nm_2d='2d', nm_3d='3d_hm'):
    '''
    combine the 2d and 3d files
    :param vis_fd:
    :return:
    '''
    cmb_fd = osp.join(set_fd, '2d3d')
    ut_t.make_folder(cmb_fd)
    fd_2d = osp.join(set_fd, '2d')
    f_nms = os.listdir(fd_2d)
    tar_size = (256, 256)
    for nm in tqdm(f_nms, desc='combining {}'.format(osp.basename(set_fd))):
        img_pth = osp.join(set_fd, '2d', nm)
        img_2d = cv2.imread(img_pth)
        img_pth = osp.join(set_fd, '3d_hm', nm)
        img_3d = cv2.imread(img_pth)
        img_2d = cv2.resize(img_2d, tar_size)
        img_3d = cv2.resize(img_3d, tar_size)
        img_cmb = np.concatenate([img_2d, img_3d], axis=1)
        cv2.imwrite(osp.join(cmb_fd, nm), img_cmb)


def genVid(fd, nm=None, fps=30, svFd='output/vid'):
    '''
    from the target folder, generate the video with given fps to folder
    svFd with name of the fd last name.
    :param fd:
    :param svFd:
    :param fps:
    :return:
    '''
    if not os.path.exists(svFd):
        os.makedirs(svFd)
    if not nm:
        nm = os.path.basename(fd)
    f_li = os.listdir(fd)
    f_li.sort(key=lambda f: int(re.sub('\D', '', f)))
    if not f_li:
        print('no images found in target dir')
        return
    img = cv2.imread(os.path.join(os.path.join(fd, f_li[0])))
    # make vid handle
    sz = (img.shape[1], img.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(os.path.join(svFd, nm + '.mp4'), fourcc, fps, sz)
    for nm in f_li:
        fname = os.path.join(os.path.join(fd, nm))
        img = cv2.imread(fname)
        video.write(img)
    video.release()