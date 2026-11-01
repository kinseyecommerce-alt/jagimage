import numpy as np



def test_a_sleeve_lit_by_the_pants_colour_stays_sleeve():
    from scipy import ndimage

    from jagimage.recolor import garment_mask, rgb_to_lab, split_outfit

    mint, lav, grey = np.array([170, 225, 205.]), np.array([205, 180, 235.]), np.array([236, 236, 238.])
    img = np.tile(grey, (240, 200, 1))
    img[20:200, 40:150] = lav          # pants
    img[0:120, 50:150] = mint          # body of the top, over the waistband
    img[0:190, 20:70] = mint * 0.9     # a sleeve hanging beside the pants, in front of them
    xs = np.linspace(0, 1, 30)[None, :, None]
    img[110:190, 40:70] = (mint * 0.85) * (1 - 0.8 * xs) + (lav * 0.8) * 0.8 * xs  # lit purple towards the pants
    img = np.pad(img, ((40, 160), (200, 200), (0, 0)), mode="edge")  # the studio backdrop around the child
    img[:, :200] = img[:, -200:] = grey
    img[-160:] = grey
    img = ndimage.gaussian_filter(img, (0.6, 0.6, 0))
    lab = rgb_to_lab(np.clip(img, 0, 255).astype(np.uint8))
    top, _ = garment_mask(lab, "#AAE1CD")
    bottom, _ = garment_mask(lab, "#CDB4EB")
    t2, b2 = split_outfit(lab, top, bottom)
    sleeve = np.zeros(top.shape, bool)
    sleeve[152:228, 222:268] = True
    assert (t2[sleeve] > 0.5).mean() > 0.97 and (b2[sleeve] > 0.5).mean() < 0.02
    pants = np.zeros(top.shape, bool)
    pants[170:235, 275:345] = True
    assert (b2[pants] > 0.5).mean() > 0.97
