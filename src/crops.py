from __future__ import division
import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms.functional as F


def pad_frame(im, frame_sz, pos_x, pos_y, patch_sz, avg_chan):
    c = patch_sz / 2
    xleft_pad = max(0, -int(round(pos_x - c)))
    ytop_pad = max(0, -int(round(pos_y - c)))
    xright_pad = max(0, int(round(pos_x + c)) - frame_sz[1])
    ybottom_pad = max(0, int(round(pos_y + c)) - frame_sz[0])
    npad = max((xleft_pad, ytop_pad, xright_pad, ybottom_pad))
    if avg_chan is not None:
        # TODO: PIL Image doesn't allow float RGB image
        avg_chan = tuple([int(round(c)) for c in avg_chan])
        im_padded = ImageOps.expand(im, border=npad, fill=avg_chan)
    else:
        im_padded = ImageOps.expand(im, border=npad, fill=0)
    return im_padded, npad


def extract_crops_z(im, npad, pos_x, pos_y, sz_src, sz_dst):
    c = sz_src / 2
    # get top-right corner of bbox and consider padding
    tr_x = npad + int(round(pos_x - c))
    # Compute size from rounded co-ords to ensure rectangle lies inside padding.
    tr_y = npad + int(round(pos_y - c))
    width = round(pos_x + c) - round(pos_x - c)
    height = round(pos_y + c) - round(pos_y - c)
    crop = im.crop((int(tr_x),
                    int(tr_y),
                    int(tr_x + width),
                    int(tr_y + height)))
    crop = crop.resize((sz_dst, sz_dst), Image.BILINEAR)
    crops = 255.0 * F.to_tensor(crop).unsqueeze(0)
    return crops


def extract_crops_x(im, npad, pos_x, pos_y, sz_src0, sz_src1, sz_src2, sz_dst):
    # take center of the biggest scaled source patch
    c = sz_src2 / 2
    # get top-right corner of bbox and consider padding
    tr_x = npad + int(round(pos_x - c))
    tr_y = npad + int(round(pos_y - c))
    # Compute size from rounded co-ords to ensure rectangle lies inside padding.
    width = round(pos_x + c) - round(pos_x - c)
    height = round(pos_y + c) - round(pos_y - c)
    search_area = im.crop((int(tr_x),
                           int(tr_y),
                           int(tr_x + width),
                           int(tr_y + height)))
    # TODO: Use computed width and height here?
    offset_s0 = (sz_src2 - sz_src0) / 2
    offset_s1 = (sz_src2 - sz_src1) / 2
    
    crop_s0 = search_area.crop((int(offset_s0),
                                int(offset_s0),
                                int(offset_s0 + round(sz_src0)),
                                int(offset_s0 + round(sz_src0))))
    crop_s0 = crop_s0.resize((sz_dst, sz_dst), Image.BILINEAR)
    crop_s1 = search_area.crop((int(offset_s1),
                                int(offset_s1),
                                int(offset_s1 + round(sz_src1)),
                                int(offset_s1 + round(sz_src1))))
    crop_s1 = crop_s1.resize((sz_dst, sz_dst), Image.BILINEAR)
    crop_s2 = search_area.resize((sz_dst, sz_dst), Image.BILINEAR)

    crop_s0 = 255.0 * F.to_tensor(crop_s0)
    crop_s1 = 255.0 * F.to_tensor(crop_s1)
    crop_s2 = 255.0 * F.to_tensor(crop_s2)
    crops = torch.stack((crop_s0, crop_s1, crop_s2))
    return crops