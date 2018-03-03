import tensorflow as tf
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
from src.convolutional import set_convolutional
from src.crops import extract_crops_z, extract_crops_x, pad_frame
sys.path.append('../')

pos_x_ph = tf.placeholder(tf.float64)
pos_y_ph = tf.placeholder(tf.float64)
z_sz_ph = tf.placeholder(tf.float64)
x_sz0_ph = tf.placeholder(tf.float64)
x_sz1_ph = tf.placeholder(tf.float64)
x_sz2_ph = tf.placeholder(tf.float64)

# the follow parameters *have to* reflect the design of the network to be imported
_conv_stride = np.array([2,1,1,1,1])
_filtergroup_yn = np.array([0,1,0,1,1], dtype=bool)
_bnorm_yn = np.array([1,1,1,1,0], dtype=bool)
_relu_yn = np.array([1,1,1,1,0], dtype=bool)
_pool_stride = np.array([2,1,0,0,0]) # 0 means no pool
_pool_sz = 3
_bnorm_adjust = True
assert len(_conv_stride) == len(_filtergroup_yn) == len(_bnorm_yn) == len(_relu_yn) == len(_pool_stride), ('These arrays of flags should have same length')
assert all(_conv_stride) >= True, ('The number of conv layers is assumed to define the depth of the network')
_num_layers = len(_conv_stride)


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


# def build_tracking_graph(final_score_sz, design, env):
#     # Make a queue of file names
#     # filename_queue = tf.train.string_input_producer(frame_name_list, shuffle=False, capacity=num_frames)
#     # image_reader = tf.WholeFileReader()
#     # # Read a whole file from the queue
#     # image_name, image_file = image_reader.read(filename_queue)

#     filename = tf.placeholder(tf.string, [], name='filename')
#     image_file = tf.read_file(filename)
#     # Decode the image as a JPEG file, this will turn it into a Tensor
#     image = tf.image.decode_jpeg(image_file)
#     image = 255.0 * tf.image.convert_image_dtype(image, tf.float32)
#     frame_sz = tf.shape(image)
#     # used to pad the crops
#     if design.pad_with_image_mean:
#         avg_chan = tf.reduce_mean(image, axis=(0,1), name='avg_chan')
#     else:
#         avg_chan = None
#     # pad with if necessary
#     frame_padded_z, npad_z = pad_frame(image, frame_sz, pos_x_ph, pos_y_ph, z_sz_ph, avg_chan)
#     frame_padded_z = tf.cast(frame_padded_z, tf.float32)
#     # extract tensor of z_crops
#     z_crops = extract_crops_z(frame_padded_z, npad_z, pos_x_ph, pos_y_ph, z_sz_ph, design.exemplar_sz)
#     frame_padded_x, npad_x = pad_frame(image, frame_sz, pos_x_ph, pos_y_ph, x_sz2_ph, avg_chan)
#     frame_padded_x = tf.cast(frame_padded_x, tf.float32)
#     # extract tensor of x_crops (3 scales)
#     x_crops = extract_crops_x(frame_padded_x, npad_x, pos_x_ph, pos_y_ph, x_sz0_ph, x_sz1_ph, x_sz2_ph, design.search_sz)
#     # use crops as input of (MatConvnet imported) pre-trained fully-convolutional Siamese net
#     template_z, templates_x, p_names_list, p_val_list = _create_siamese(os.path.join(env.root_pretrained,design.net), x_crops, z_crops)
#     template_z = tf.squeeze(template_z)
#     templates_z = tf.stack([template_z, template_z, template_z])
#     # compare templates via cross-correlation
#     scores = _match_templates(templates_z, templates_x, p_names_list, p_val_list)
#     # upsample the score maps
#     scores_up = tf.image.resize_images(scores, [final_score_sz, final_score_sz],
#         method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
#     return filename, image, templates_z, scores_up


# # import pretrained Siamese network from matconvnet
# def _create_siamese(net_path, net_x, net_z):
#     # read mat file from net_path and start TF Siamese graph from placeholders X and Z
#     params_names_list, params_values_list = _import_from_matconvnet(net_path)

#     # loop through the flag arrays and re-construct network, reading parameters of conv and bnorm layers
#     for i in xrange(_num_layers):
#         print('> Layer '+str(i+1))
#         # conv
#         conv_W_name = _find_params('conv'+str(i+1)+'f', params_names_list)[0]
#         conv_b_name = _find_params('conv'+str(i+1)+'b', params_names_list)[0]
#         print('\t\tCONV: setting '+conv_W_name+' '+conv_b_name)
#         print('\t\tCONV: stride '+str(_conv_stride[i])+', filter-group '+str(_filtergroup_yn[i]))
#         conv_W = params_values_list[params_names_list.index(conv_W_name)]
#         conv_b = params_values_list[params_names_list.index(conv_b_name)]
#         # batchnorm
#         if _bnorm_yn[i]:
#             bn_beta_name = _find_params('bn'+str(i+1)+'b', params_names_list)[0]
#             bn_gamma_name = _find_params('bn'+str(i+1)+'m', params_names_list)[0]
#             bn_moments_name = _find_params('bn'+str(i+1)+'x', params_names_list)[0]
#             print('\t\tBNORM: setting '+bn_beta_name+' '+bn_gamma_name+' '+bn_moments_name)
#             bn_beta = params_values_list[params_names_list.index(bn_beta_name)]
#             bn_gamma = params_values_list[params_names_list.index(bn_gamma_name)]
#             bn_moments = params_values_list[params_names_list.index(bn_moments_name)]
#             bn_moving_mean = bn_moments[:,0]
#             bn_moving_variance = bn_moments[:,1]**2 # saved as std in matconvnet
#         else:
#             bn_beta = bn_gamma = bn_moving_mean = bn_moving_variance = []
        
#         # set up conv "block" with bnorm and activation 
#         net_x = set_convolutional(net_x, conv_W, np.swapaxes(conv_b,0,1), _conv_stride[i], \
#                             bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance, \
#                             filtergroup=_filtergroup_yn[i], batchnorm=_bnorm_yn[i], activation=_relu_yn[i], \
#                             scope='conv'+str(i+1), reuse=False)
        
#         # notice reuse=True for Siamese parameters sharing
#         net_z = set_convolutional(net_z, conv_W, np.swapaxes(conv_b,0,1), _conv_stride[i], \
#                             bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance, \
#                             filtergroup=_filtergroup_yn[i], batchnorm=_bnorm_yn[i], activation=_relu_yn[i], \
#                             scope='conv'+str(i+1), reuse=True)    
        
#         # add max pool if required
#         if _pool_stride[i]>0:
#             print('\t\tMAX-POOL: size '+str(_pool_sz)+ ' and stride '+str(_pool_stride[i]))
#             net_x = tf.nn.max_pool(net_x, [1,_pool_sz,_pool_sz,1], strides=[1,_pool_stride[i],_pool_stride[i],1], padding='VALID', name='pool'+str(i+1))
#             net_z = tf.nn.max_pool(net_z, [1,_pool_sz,_pool_sz,1], strides=[1,_pool_stride[i],_pool_stride[i],1], padding='VALID', name='pool'+str(i+1))

#     print

#     return net_z, net_x, params_names_list, params_values_list


# def _import_from_matconvnet(net_path):
#     mat = scipy.io.loadmat(net_path)
#     net_dot_mat = mat.get('net')
#     # organize parameters to import
#     params = net_dot_mat['params']
#     params = params[0][0]
#     params_names = params['name'][0]
#     params_names_list = [params_names[p][0] for p in xrange(params_names.size)]
#     params_values = params['value'][0]
#     params_values_list = [params_values[p] for p in xrange(params_values.size)]
#     return params_names_list, params_values_list


# # find all parameters matching the codename (there should be only one)
# def _find_params(x, params):
#     matching = [s for s in params if x in s]
#     assert len(matching)==1, ('Ambiguous param name found')    
#     return matching


# def _match_templates(net_z, net_x, params_names_list, params_values_list):
#     # finalize network
#     # z, x are [B, H, W, C]
#     net_z = tf.transpose(net_z, perm=[1,2,0,3])
#     net_x = tf.transpose(net_x, perm=[1,2,0,3])
#     # z, x are [H, W, B, C]
#     Hz, Wz, B, C = tf.unstack(tf.shape(net_z))
#     Hx, Wx, Bx, Cx = tf.unstack(tf.shape(net_x))
#     # assert B==Bx, ('Z and X should have same Batch size')
#     # assert C==Cx, ('Z and X should have same Channels number')
#     net_z = tf.reshape(net_z, (Hz, Wz, B*C, 1))
#     net_x = tf.reshape(net_x, (1, Hx, Wx, B*C))
#     net_final = tf.nn.depthwise_conv2d(net_x, net_z, strides=[1,1,1,1], padding='VALID')
#     # final is [1, Hf, Wf, BC]
#     net_final = tf.concat(tf.split(net_final, 3, axis=3), axis=0)
#     # final is [B, Hf, Wf, C]
#     net_final = tf.expand_dims(tf.reduce_sum(net_final, axis=3), axis=3)
#     # final is [B, Hf, Wf, 1]
#     if _bnorm_adjust:
#         bn_beta = params_values_list[params_names_list.index('fin_adjust_bnb')]
#         bn_gamma = params_values_list[params_names_list.index('fin_adjust_bnm')]
#         bn_moments = params_values_list[params_names_list.index('fin_adjust_bnx')]
#         bn_moving_mean = bn_moments[:,0]
#         bn_moving_variance = bn_moments[:,1]**2
#         net_final = tf.layers.batch_normalization(net_final, beta_initializer=tf.constant_initializer(bn_beta),
#                                                 gamma_initializer=tf.constant_initializer(bn_gamma),
#                                                 moving_mean_initializer=tf.constant_initializer(bn_moving_mean),
#                                                 moving_variance_initializer=tf.constant_initializer(bn_moving_variance),
#                                                 training=False, trainable=False)

#     return net_final
