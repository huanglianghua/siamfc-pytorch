import numpy as np
import scipy.io
import sys
import six
import os.path
from PIL import Image, ImageStat
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from src.crops import extract_crops_z, extract_crops_x, pad_frame
sys.path.append('../')


class SiameseNet(nn.Module):

    def __init__(self, root_pretrained=None, net=None):
        super(SiameseNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 32, 3, 1, groups=2)
        )
        self.branch = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        )
        self.bn_adjust = nn.BatchNorm2d(1)
        self._initialize_weights()

        self.cuda = torch.cuda.is_available()
        if net is not None:
            net_path = os.path.join(root_pretrained, net)
            if os.path.splitext(net_path)[1] == '.mat':
                load_siamfc_from_matconvnet(net_path, self)
            elif os.path.splitext(net_path)[1] == '.pth':
                if self.cuda:
                    self.load_state_dict(torch.load(net_path))
                else:
                    self.load_state_dict(torch.load(
                        net_path, 
                        map_location=lambda storage, loc: storage
                    ))
            else:
                raise Exception('unknown file extention!')
            for m in self.modules():
                m.training = False

    def forward(self, z, x):
        assert z.size()[:2] == x.size()[:2]

        z = self.branch(z)
        x = self.branch(x)

        out = self.xcorr(z, x)
        out = self.bn_adjust(out)

        return out

    def xcorr(self, z, x):
        out = []
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i,:,:,:].unsqueeze(0), z[i,:,:,:].unsqueeze(0)))
        
        return torch.cat(out, dim=0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def get_template_z(self, pos_x, pos_y, z_sz, image, 
                       design):
        if isinstance(image, six.string_types):
            image = Image.open(image)
        avg_chan = ImageStat.Stat(image).mean
        frame_padded_z, npad_z = pad_frame(image, image.size, pos_x, pos_y, z_sz, avg_chan)
        z_crops = extract_crops_z(frame_padded_z, npad_z, pos_x, pos_y, z_sz, design.exemplar_sz)
        template_z = self.branch(Variable(z_crops))
        return image, template_z

    def get_scores(self, pos_x, pos_y, scaled_search_area, template_z, filename,
                   design, final_score_sz):
        image = Image.open(filename)
        avg_chan = ImageStat.Stat(image).mean
        frame_padded_x, npad_x = pad_frame(image, image.size, pos_x, pos_y, scaled_search_area[2], avg_chan)
        x_crops = extract_crops_x(frame_padded_x, npad_x, pos_x, pos_y, scaled_search_area[0], scaled_search_area[1], scaled_search_area[2], design.search_sz)
        template_x = self.branch(Variable(x_crops))
        template_z = template_z.repeat(template_x.size(0), 1, 1, 1)
        scores = self.xcorr(template_z, template_x)
        scores = self.bn_adjust(scores)
        # TODO: any elegant alternator?
        scores = scores.squeeze().permute(1, 2, 0).data.numpy()
        scores_up = cv2.resize(scores, (final_score_sz, final_score_sz), interpolation=cv2.INTER_CUBIC)
        scores_up = scores_up.transpose((2, 0, 1))
        return image, scores_up


def load_siamfc_from_matconvnet(net_path, model):
    params_names_list, params_values_list = load_matconvnet(net_path)

    params_values_list = [torch.from_numpy(p) for p in params_values_list]
    for l, p in enumerate(params_values_list):
        param_name = params_names_list[l]
        if 'conv' in param_name and param_name[-1] == 'f':
            p = p.permute(3, 2, 0, 1)
        p = torch.squeeze(p)
        params_values_list[l] = p

    net = nn.Sequential(
        model.conv1,
        model.conv2,
        model.conv3,
        model.conv4,
        model.conv5
    )

    for l, layer in enumerate(net):
        layer[0].weight.data[:] = params_values_list[params_names_list.index('br_conv%df' % (l + 1))]
        layer[0].bias.data[:] = params_values_list[params_names_list.index('br_conv%db' % (l + 1))]

        if l < len(net) - 1:
            layer[1].weight.data[:] = params_values_list[params_names_list.index('br_bn%dm' % (l + 1))]
            layer[1].bias.data[:] = params_values_list[params_names_list.index('br_bn%db' % (l + 1))]

            bn_moments = params_values_list[params_names_list.index('br_bn%dx' % (l + 1))]
            layer[1].running_mean[:] = bn_moments[:,0]
            layer[1].running_var[:] = bn_moments[:,1] ** 2
        else:
            model.bn_adjust.weight.data[:] = params_values_list[params_names_list.index('fin_adjust_bnm')]
            model.bn_adjust.bias.data[:] = params_values_list[params_names_list.index('fin_adjust_bnb')]

            bn_moments = params_values_list[params_names_list.index('fin_adjust_bnx')]
            model.bn_adjust.running_mean[:] = bn_moments[0]
            model.bn_adjust.running_var[:] = bn_moments[1] ** 2

    return model

def load_matconvnet(net_path):
    mat = scipy.io.loadmat(net_path)
    net_dot_mat = mat.get('net')
    params = net_dot_mat['params']
    params = params[0][0]
    params_names = params['name'][0]
    params_names_list = [params_names[p][0] for p in range(params_names.size)]
    params_values = params['value'][0]
    params_values_list = [params_values[p] for p in range(params_values.size)]

    return params_names_list, params_values_list