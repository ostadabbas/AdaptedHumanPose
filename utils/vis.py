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
import sys
import ntpath
import time
from . import utils_tool, html
from subprocess import Popen, PIPE
# from scipy.misc import imresize
from skimage.transform import resize # misc deprecated e


def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):

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
