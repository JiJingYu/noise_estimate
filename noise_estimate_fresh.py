import csv
from itertools import product

import cv2
import numpy as np
import tqdm


def image2cols(image, patch, stride):
    """
    Inputs:
    image: the noisy image to be patchized
    patch: the predefined size of patches
    stride: the predefined gap between neighbor patches

    Outputs:
    res: the set of decomposed patches
    Author: Jingyu Ji
    """
    res = []
    H, W, C = image.shape
    for i in range(0, H - patch, stride):
        for j in range(0, W - patch, stride):
            res.append(image[i:i + patch, j:j + patch].flatten())
    return np.vstack(res).T


def noise_estimate(image, patch):
    p_out = image2cols(image = image, patch = patch, stride = 3)
    n_features, n_samples = p_out.shape
    mu = np.mean(p_out, axis = 1)
    sigma = (p_out - mu[:, None]) @ (p_out - mu[:, None]).T / n_samples
    eigvalue = np.sort(np.linalg.eigvals(sigma))
    for compont_count in range(n_features - 1, 0, -1):
        Mean = np.mean(eigvalue[:compont_count])
        if np.sum(eigvalue[:compont_count] > Mean) == np.sum(eigvalue[:compont_count] < Mean):
            break
    return np.sqrt(Mean)


def test_noise_estimate_param(sigma, patch, image_idx, crop):
    filename = 'data/{:05d}.bmp'.format(image_idx)
    img = cv2.imread(filename)
    noisy_image = img + np.random.randn(*img.shape) * sigma
    res = []
    sigma_pred = noise_estimate(noisy_image, patch)
    res.append(sigma_pred)
    return res


def test_noise_estimate():
    sigmas = [25]
    patchs = [5, 6]
    image_idxs = np.arange(1, 11, 1)
    crops = [280]  # 这个参数没用了
    args = [sigmas, patchs, image_idxs, crops]

    csvfile = open('names_6.csv', 'w')
    writer = csv.DictWriter(csvfile, fieldnames = ['sigma', 'patch', 'image_idx', 'crop', 'predict'])
    writer.writeheader()
    pbar = tqdm.tqdm(total = np.product(list(map(len, args))))
    for sigma, patch, image_idx, crop in product(*args):
        pbar.update(1)
        res = test_noise_estimate_param(sigma, patch, image_idx, crop)
        for it in res:
            writer.writerow({ 'sigma'    : sigma,
                              'patch'    : patch,
                              'image_idx': image_idx,
                              'crop'     : crop,
                              'predict'  : it })
    pbar.close()
    csvfile.close()


if __name__ == '__main__':
    test_noise_estimate()
