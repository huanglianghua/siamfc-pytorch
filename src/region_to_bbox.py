import numpy as np

def region_to_bbox(region, center=True):

    n = len(region)
    assert n==4 or n==8, ('GT region format is invalid, should have 4 or 8 entries.')

    if n==4:
        return _rect(region, center)
    else:
        return _poly(region, center)

# we assume the grountruth bounding boxes are saved with 0-indexing
def _rect(region, center):
    
    if center:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
        return cx, cy, w, h
    else:
        #region[0] -= 1
        #region[1] -= 1
        return region


def _poly(region, center):
    cx = np.mean(region[::2])
    cy = np.mean(region[1::2])
    x1 = np.min(region[::2])
    x2 = np.max(region[::2])
    y1 = np.min(region[1::2])
    y2 = np.max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1/A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1

    if center:
        return cx, cy, w, h
    else:
        return cx-w/2, cy-h/2, w, h

