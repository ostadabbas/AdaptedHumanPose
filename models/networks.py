import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opts):
    """Return a learning rate scheduler
    can't use last epoch for initalize as 'last_epoch=opts.start_epoch - 1' maybe pytorch's bug .
    Parameters:
        optimizer          -- the optimizer of the network
        opts (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opts.lr_policy == 'linear':       # not working now. If need, add decay args
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opts.epoch_decay_st - opts.niter_decay) / float(opts.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_iters, gamma=0.1)
    elif opts.lr_policy == 'plateau':    # metric not updated no use
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opts.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.niter, eta_min=0)
    elif opts.lr_policy == 'multi_step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, opts.lr_dec_epoch)    # gamma default one is 0.1
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opts.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function,
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):   # element layer has weight parameter or just linear
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain) # init should be std here.
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if len(gpu_ids)>1:
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs only when more GPUs there
    init_weights(net, init_type, init_gain=init_gain)   # use different init types
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128 |
        n_phy -- how many physical parameters
        if_gated -- if use gated physical parameters
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def pm_G(input_nc, output_nc, ngf, n_stg=3, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], n_phy=1, phyMode='concat',
         n_gateLayers=1, actiType=None, sz_std=256, if_posInit = 'w'):
    '''
    build net then initialize it.  a wrapper for net generation and initialization.
    :param input_nc:
    :param output_nc:
    :param ngf:
    :param n_stg:   how many stages in the net.
    :param norm:
    :param use_dropout:
    :param init_type:
    :param init_gain:
    :param gpu_ids:
    :param n_phy:
    :param phyMode:
    :param n_gateLayers:
    :param actiType:
    :param sz_std:
    :param if_posInit:
    :return:
    '''
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    ## single version
    # net = UnetPmGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer,
    #                       use_dropout=use_dropout, n_phy=n_phy, phyMode=phyMode, n_gateLayers=n_gateLayers, actiType=actiType, sz_std=sz_std)
    ## stacked version
    net = StkPmGenerator(input_nc, output_nc, 8, ngf, n_stg=n_stg, norm_layer=norm_layer,
                          use_dropout=use_dropout, n_phy=n_phy, phyMode=phyMode, n_gateLayers=n_gateLayers, actiType=actiType, sz_std=sz_std) # a net, return all intermediate results, n_down is fixed here
    net = init_net(net, init_type, init_gain, gpu_ids) # init can be separated so wrap the Unet here  G = net + init

    # deprecated, only for positive initialization purpose.
    if phyMode == 'sfg' and if_posInit == 'w':    # must have the scalar initialize it to 0
        if len(gpu_ids)>0:
            fc_gate = net.module.fc_gate
        else:
            fc_gate = net.fc_gate
        with torch.no_grad():
            for module in fc_gate.children():   # loop fc to set the weight, special initialization
                if type(module) == nn.Linear:
                    module.weight.data.fill_(0.1)   # set data make backend error?!
            # print('after positive initialization, fc gated layer is') # check w initialization
            # for i, module in enumerate(fc_gate.children()):
            #     if type(module) == nn.Linear:
            #         print('weight {} is'.format(i), module.weight)
    return net
    # return init_net(net, init_type, init_gain, gpu_ids) # init can be separated so wrap the Unet here  G = net + init

# SA to a class
class D_SA(nn.Module):
    def __init__(self, li_ch_inp, n_sa=17, stgs_D=[1,0,0,0], n_layer=3, stride=1):
        '''
        semantic aware discriminator for multi-stage gaussian version.
        :param li_ch_inp:  the input channels of each stage.
        :param n_sa:    the number of semantic entities
        :param n_layer:    layer of the basic network
        :return: return ModuleList(ML) ->  ML[n_li_ch *ML[n_sa]
        '''
        super(D_SA, self).__init__()
        self.n_sa = n_sa        # how many semantic

        # full D version
        # li_D = nn.ModuleList([
        #     nn.ModuleList([D_SA_blk(ch, n_layer=n_layer, stride=stride) for i in range(n_sa)])
        #     for ch in li_ch_inp]) # 4x 17  64 , 128
        # 4 stg ->  17 jts ->

        # slim version
        self.stgs_D = stgs_D
        li_D = nn.ModuleList()  # if not used then empty
        for i, ch in enumerate(li_ch_inp):
            if stgs_D[i]:
                li_jt = nn.ModuleList([D_SA_blk(ch, n_layer=n_layer, stride=stride) for j in range(n_sa)])      # 16 jts
            else:
                li_jt = nn.ModuleList()   # empty
            li_D.append(li_jt)
        self.model = li_D

    def forward(self, li_stg):
        rst = []
        for i, ft in enumerate(li_stg):     # a list of features
            rst_stg = []
            if self.stgs_D[i]:      # there is disc
                for j in range(self.n_sa):
                    rst_stg.append(self.model[i][j](ft))    # input 30 x256 expect to have 64 channels?
            rst.append(rst_stg) # otherwise keep empty value
        return rst




def D_SA_blk(n_ch, n_f=64, n_layer=3, stride=1):
    '''
    bulding block for D_SA
    :param n_ch:  input channel
    :param n_f: the common fts in the net
    :param n_layer:
    :param if_upCh: if double the ch each step, no , too large actually
    :return:
    '''
    kw = 3  # original 4
    padw = 1
    li_module = [nn.Conv2d(n_ch, n_f, kernel_size=kw, stride=stride, padding=padw), nn.LeakyReLU(0.2, True)]
    mul = 1
    mul_pre = 1 # no use, but easy for understanding
    for i in range(1, n_layer):
        mul_pre = mul
        mul = min(2 ** i,  8)
        li_module += [
            nn.Conv2d(n_f*mul_pre, n_f*mul, kernel_size=kw, stride=stride, padding=padw),
            nn.BatchNorm2d(n_f * mul),
            nn.LeakyReLU(0.2, True) # true in place
        ]

    li_module += [
        nn.Conv2d(n_f * mul, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
    model = nn.Sequential(*li_module)
    return model

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], ch_out=1, kn_D=1):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, ch_out=ch_out)   # only pixel has ch_out right now
    elif netD == 'convD':     # classify if each pixel is real or fake
        net = ConvD(input_nc, ndf, norm_layer=norm_layer, ch_out=ch_out, n_layers=n_layers_D, kn_D=kn_D)   # only pixel has ch_out right now
    elif netD == 'convD_C1':     # classify if each pixel is real or fake
        net = ConvD_C1(input_nc, ndf, norm_layer=norm_layer, ch_out=ch_out, n_layers=n_layers_D, kn_D=kn_D)   # only pixel has ch_out right now
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def whtL1Loss(src, tar, baseWht = 1):       #  use a customer class for this loss
    '''
    auto weighting the absolute value with base weight
    :param src:
    :param tar:
    :param baseWht:
    :return:
    '''
    loss = torch.abs(src-tar)*(tar+baseWht)  # self weighted loss
    return loss

class WhtL1Loss(torch.nn.Module):
    def __init__(self, whtScal =100, clipMod='clip11', baseWht=1):
        super(WhtL1Loss, self).__init__()
        self.whtScal = whtScal
        self.clipMod = clipMod
        self.baseWht = baseWht
    def forward(self, src, tar):
        if self.clipMod == 'clip11':
            wht = (tar+1) / 2 * self.whtScal + self.baseWht
        elif self.clipMod == 'clip01':
            wht = tar * self.whtScal + self.baseWht
        else:       # no processing only adds wht in
            wht = tar + self.baseWht
        loss = (torch.abs(src-tar)*wht).mean()
        return loss

class autoWtL(torch.nn.Module):
    def __init__(self, whtScal =100, clipMod='clip01', baseWht=1, type_L = 'L2'):
        super(autoWtL, self).__init__()
        self.whtScal = whtScal
        self.clipMod = clipMod
        self.baseWht = baseWht
        self.type_L = type_L
    def forward(self, src, tar):
        if self.clipMod == 'clip11':
            wht = (tar+1) / 2 * self.whtScal + self.baseWht
        elif self.clipMod == 'clip01':
            wht = tar * self.whtScal + self.baseWht
        else:       # no processing only adds wht in
            wht = tar + self.baseWht
        if 'L1' == self.type_L:
            loss = (torch.abs(src-tar)*wht).mean()  # L1 loss
        elif 'L2' == self.type_L:
            loss = (wht*(src-tar)**2).mean()        # MSE loss
        else:
            print("no such loss implementation", self.type_L)
        return loss

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process. fixed 8 down for unet256
        """

        # no input handle, can't wire the input directly into center layer
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

def downBlk(in_f, out_f, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
    # generate the sequential module for each down sample step including  relu, conv, norm
    # for outter and inner, some part will be ignored accordingly
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d
    relu_blk = nn.LeakyReLU(0.2, True)
    conv2d_blk = nn.Conv2d(in_f, out_f,kernel_size=4, stride=2, padding=1, bias=use_bias)

    norm_blk = norm_layer(out_f)
    if outermost:
        module_ls = [conv2d_blk]
        # return nn.Sequential(conv2d_blk)    # conv only
    elif innermost:
        # return nn.Sequential(relu_blk, conv2d_blk) # r, c only
        module_ls = [relu_blk, conv2d_blk]
    else:
        module_ls = [relu_blk, conv2d_blk, norm_blk]
        if use_dropout:     # add dropout at down part, not in original
            module_ls += [nn.Dropout(0.5)]
    return nn.Sequential(*module_ls) # r,c,n

def upBlk(in_f, out_f, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
    '''
    simply generate the upsample module with c,r,n, skip is implemented in forward in unet structure, when use, please choose in_f accordingly to accept skip channels.
    :param in_f: for inner most, choose the correct in_f for physical injection vector
    :param out_f:
    :param outermost:
    :param innermost: not used for initial edition as inner and inter use same up block, can be extanded later for different structure
    :param norm_layer:
    :param use_dropout:
    :return:
    '''
    if type(norm_layer) == functools.partial:   # if batchnorm there is bia term, save the trouble to add to the conv layer just before it.
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d
    relu_blk = nn.ReLU(True)        # inplace Relu
    deconv2d_blk = nn.ConvTranspose2d(in_f, out_f, kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
    norm_blk = norm_layer(out_f)
    if outermost:
        module_ls = [relu_blk, deconv2d_blk]    # I don't add tanh here as PM will be the pressure data which is hard to normalized to 0 and 1 due to the pressure differences
        # return nn.Sequential(conv2d_blk)    # conv only
    else:  # inner and inter, same full structure
        module_ls = [relu_blk, deconv2d_blk, norm_blk]
        if use_dropout:     # add dropout at down part, not in original
            module_ls += [nn.Dropout(0.5)]
    return nn.Sequential(*module_ls) # r,c,n

class StkPmGenerator(nn.Module):
    '''
    generate stackeed pm generator. Each moduel generate output also the ngf feature maps.  input, current output and fts will be combined as input for next stage.
    '''
    def __init__(self,input_nc, output_nc, num_downs, ngf=64, n_stg=3, norm_layer=nn.BatchNorm2d, use_dropout=False, n_phy=1, phyMode= 'concat', n_gateLayers=1,actiType=None, sz_std=256):
        '''
        :param input_nc:
        :param output_nc:
        :param num_downs:
        :param ngf:
        :param n_stg:   how many stages are in the net
        :param norm_layer:
        :param use_dropout:
        :param n_phy:
        :param phyMode:
        :param n_gateLayers:
        :param actiType:
        :param sz_std:
        '''
        super(StkPmGenerator, self).__init__()
        self.mdLs = nn.ModuleList()
        if n_stg<1:
            print("wrong stage number", n_stg)
            quit(-1)
        # 1st input input_nc,  all later comes with input   input_nc + output_nc, + ngf
        netT = UnetPmGenerator(input_nc, output_nc, num_downs, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_phy=n_phy, phyMode=phyMode, n_gateLayers=n_gateLayers, actiType=actiType, sz_std=sz_std)
        self.mdLs.append(netT)
        for i in range(1, n_stg):   # for later has more input channels.
            netT = UnetPmGenerator(input_nc+output_nc+ngf, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_phy=n_phy, phyMode=phyMode, n_gateLayers=n_gateLayers, actiType=actiType, sz_std=sz_std)
            self.mdLs.append(netT)

    def forward(self, img, phyVec):
        outs = []
        input = img
        for i, netT in enumerate(self.mdLs):
            out, fts = netT(input, phyVec)
            outs.append(out)
            # update input
            input = torch.cat([img, out, fts], 1)

        return outs     #

class UnetPmGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_phy=1, phyMode= 'concat', n_gateLayers=1,actiType=None, sz_std=256):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
            use_dropout     -- only affect the up branch in original design
            n_phy           -- how many physical parameters in
            phyMode         -- physique injection mode,
                            concat: simple concate at center,
                            fc_gated: use a simple fc layer to generate gate
                            sim_gated: times phy[0] weight directly
            actiType        -- activaton for the final layer to regulate it to tanh or sigmoid

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        history:  8.13.19,  add channel wise fc, return also intermediate features.
        """
        # no input handle, can't wire the input directly into center layer
        super(UnetPmGenerator, self).__init__()
        # keep gated structure for forward usage
        self.n_phy = n_phy
        self.phyMode = phyMode
        self.n_gateLayers = n_gateLayers
        self.ngf = ngf
        # construct unetPm
        # down branch
        down_li = nn.ModuleList()
        down_li.append(downBlk(input_nc,ngf,outermost=True, norm_layer=norm_layer)) # n_d = 1
        ngf_tmp = ngf
        if num_downs<2:
            print('num_downs at least 2 layers but get', num_downs)
            quit(-1)
        for i in range(min(3, num_downs-2)):  # upscale ngf to 8      # n_d4,  minus exclude 1st and last.
            down_li.append(downBlk(ngf_tmp, ngf_tmp*2, norm_layer=norm_layer))
            ngf_tmp=ngf_tmp*2   # update the ngf
        if ngf_tmp != ngf*8:
            print('up ngf numbers wrong')
            exit(0)
        for i in range(num_downs-5):    # same channel  # n_d 7
            down_li.append(downBlk(ngf_tmp, ngf_tmp, norm_layer=norm_layer))
        down_li.append(downBlk(ngf_tmp, ngf_tmp, innermost=True, norm_layer=norm_layer))    # n_d 8 , inner nor norm
        self.down_li = down_li # save to instance
        # up branch
        # inner most
        up_li = nn.ModuleList()
        if 'concat' == phyMode: # in concat physique in the enc layer
            n_encUp = ngf_tmp+n_phy
        else:
            n_encUp = ngf_tmp
        up_li.append(upBlk(n_encUp, ngf_tmp, innermost=True, norm_layer=norm_layer))    # n_up = 1
        # same 8ngf section
        for i in range(num_downs-5):        # n_up = 4
            up_li.append(upBlk(ngf_tmp*2, ngf_tmp, norm_layer=norm_layer, use_dropout=use_dropout))
        for i in range(min(3, num_downs-2)):  # down ngf section  # n_up = 7
            up_li.append(upBlk(ngf_tmp*2, int(ngf_tmp/2), norm_layer=norm_layer))
            ngf_tmp=int(ngf_tmp/2) # reduce ngf_tmp
        if ngf_tmp != ngf:
            print('down ngf number wrong')
            exit(0)
        # output  session
        # up_li.append(upBlk(ngf_tmp*2, output_nc, outermost=True, norm_layer=norm_layer))     # direct output single version
        up_li.append(upBlk(ngf_tmp * 2, ngf, outermost=True, norm_layer=norm_layer))
        self.up_li = up_li
        self.extractor = nn.Conv2d(ngf, output_nc, kernel_size=1, stride=1, padding=0)  # output extraction
        if 'tanh' == actiType:
            self.actiFn = nn.Tanh()
        elif 'sigmoid' == actiType:
            self.actiFn = nn.Sigmoid()
            print('employ sigmoid activateion')
        else:
            self.actiFn = None

        if 'fc_gated' == phyMode: # add gated fc layers, single layer now
            li_gate = nn.ModuleList()
            if n_gateLayers<1:
                print('warning:n_gatedLayers smaller than 1, force it to 1')
                self.n_gateLayers=1
            li_gate.append(nn.Linear(n_phy, ngf*8)) # 1st, fc,  n_phy to ngf*8
            for i in range(n_gateLayers-1): # r, fc
                li_gate.append(nn.ReLU(True))
                li_gate.append(nn.Linear(ngf*8, ngf*8))
            self.fc_gate = nn.Sequential(*li_gate)
        elif phyMode in {'sfg', 'dcfg'}:     # final gated, if in simple final gate or deconv final gated, then
            if 'sfg' == phyMode:
                print('employ the simple final gated layer')
            li_gate = nn.ModuleList()
            n_hidden = 10
            if n_gateLayers < 1:
                print('warning:n_gatedLayers smaller than 1, force it to 1')
                self.n_gateLayers = 1
            if self.n_gateLayers == 1:
                li_gate.append(nn.Linear(n_phy, 1))
                li_gate.append(nn.ReLU(True))
            else:   # first to hidden layer numbers then
                li_gate.append(nn.Linear(n_phy, n_hidden))    # here we simply add the hidden neuron to be 10, but can be controlled later
                li_gate.append(nn.ReLU(True))
                for i in range(n_gateLayers-2): # all middle except begin and end are same 10 hidden, could be nothing here for n_gate=2
                    li_gate.append(nn.Linear(n_hidden, n_hidden))
                    li_gate.append(nn.ReLU(True))
                li_gate.append(nn.Linear(10, 1))
                li_gate.append(nn.ReLU(True))
            self.fc_gate = nn.Sequential(*li_gate)
            # self.fc_gate = nn.Linear(n_phy, 1)  # everything simply to one, then scale to size
        if 'dcfg' == phyMode:       # add a deconv layer
            print('employ dcfg final gated layer')
            self.dcfg = nn.ConvTranspose2d(output_nc, output_nc,
                                        kernel_size=sz_std, stride=1
                                        )   # same channel as output, but simply
            # self.dc_gate =
        # construct unet structure original unet
        # unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        # for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
        #     unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # # gradually reduce the number of filters from ngf * 8 to ngf
        # unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, img, phyVec):
        '''
        forward interface with the input data.  img and phyVec
        :param img:
        :param phyVec: should be already tensor like
        :return:
        '''
        # phyVec_in = phyVec[:,self.n_phy]  # could be empty
        rst_down = []
        x = img # input temp layers
        for i,module in enumerate(self.down_li):
            x = module(x)
            rst_down.append(x)
        y = rst_down.pop()  # last element
        if 'concat' == self.phyMode:        # code operation
            if self.n_phy>0:
                y = torch.cat((y, phyVec[:, :self.n_phy].view(-1, self.n_phy, 1,1)), 1)
        elif 'fc_gated' == self.phyMode:
            if self.n_phy<1:
                print('waring: n_phy <0 when gated, set it to 1')
                self.n_phy=1
            z1 = self.fc_gate(phyVec[:, :self.n_phy])
            z = z1.view(-1, 8*self.ngf, 1, 1)
            y = z*y     # pointwise multiplication
        elif 'sim_gated' == self.phyMode:
            y = y * phyVec[:, 0].view(-1,1,1,1)

        for i, module in enumerate(self.up_li):
            if i>0: # all other module , concate result first
                y = torch.cat([y, rst_down.pop()], 1)
            y = module(y)
        out = self.extractor(y) # get the final image out
        if self.actiFn:     # if run the sigmoid function on final output
            out= self.actiFn(out)       # y to out, y is the feature vectors

        if 'sfg' == self.phyMode:   # final gating
            scal = self.fc_gate(phyVec[:, :self.n_phy])  # to scalar
            for i in range(2):  # expand the later part
                scal = scal.unsqueeze(-1)
            sfg = scal.expand_as(y)
            out = sfg * out # gated operation
        elif 'dcfg' == self.phyMode:
            scal = self.fc_gate(phyVec[:, :self.n_phy])  # to scalar
            for i in range(2):  # expand the later part
                scal = scal.unsqueeze(-1)
            dcfg = self.dcfg(scal)
            out = out * dcfg    # gate it
        return out, y    # return out put and also feature

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)      # inplace relu
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:   # outer most not down relu
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:     #  innermost no submodule , no down norm
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, stride=1):
        """Construct a PatchGAN discriminator
        hist: add stride control to be 1

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 3      # original 4, will shrink.
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=stride, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=stride, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)
    history: 8/16/20, ch_out control the output channel
    """

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, ch_out=1):
        """Construct a 1x1 PatchGAN discriminator, 3 layers

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """

        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ch_out, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class ConvD(nn.Module):
    """pure conv discriminator , similar to pixel one, but with flexible kernel size and layer control, and stride
    default in pixel gan format
    """

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, ch_out=1, n_layers=3, kn_D=1, stride=1, opts=None):
        """
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(ConvD, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        kw = kn_D  # original 4, will shrink.
        padw = int((kn_D-1)/2)
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=stride, padding=padw), nn.LeakyReLU(0.2, True)]     # intermediate 2048 -> 64
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=stride, padding=padw,
                          bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult = min(2 ** (n_layers-1), 8)

        # branch out


        # original design
        sequence += [
            nn.Conv2d(ndf * nf_mult, ch_out, kernel_size=kw, stride=stride, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)       # last layer simplify it

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class ConvD_C1(nn.Module):
    """convD structure with C1 added for whole image classification. share bb_D. two output
    """

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, ch_out=1, n_layers=3, kn_D=1, stride=1):
        """
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(ConvD_C1, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        kw = kn_D  # original 4, will shrink.
        padw = int((kn_D - 1) / 2)
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=stride, padding=padw),
                    nn.LeakyReLU(0.2, True)]  # intermediate 2048 -> 64
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=stride, padding=padw,
                          bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        # add one to reduce the ch  to ndf
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** (n_layers - 1), 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf, kernel_size=kw, stride=stride, padding=padw,
                      bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True)
        ]

        # bb_D
        self.bb_D = nn.Sequential(*sequence)

        # head c1
        nr_fc = 20
        seq_c1 = [
            nn.Linear(ndf*8*8, nr_fc),
            # norm_layer(nr_fc),        # not working possibly the 2D issue
            nn.BatchNorm1d(nr_fc),
            nn.LeakyReLU(0.2, True),
            nn.Linear(nr_fc, 1)
        ]
        self.ft_fc_len = ndf * 8 * 8
        self.head_C1 = nn.Sequential(*seq_c1)       # conventional 1 output

        # head Pch
        self.head_pch = nn.Conv2d(ndf, ch_out, kernel_size=kw, stride=stride, padding=padw)

        # original design
        # sequence += [
        #     nn.Conv2d(ndf * nf_mult, ch_out, kernel_size=kw, stride=stride,
        #               padding=padw)]  # output 1 channel prediction map
        # self.model = nn.Sequential(*sequence)  # last layer simplify it

    def forward(self, input):
        """Standard forward."""
        # original
        # return self.model(input)

        # branch out
        ft = self.bb_D(input)
        C1 = self.head_C1(ft.view(-1, self.ft_fc_len))          # expec 4d get 2D instead
        pch = self.head_pch(ft)

        return pch, C1      # return two