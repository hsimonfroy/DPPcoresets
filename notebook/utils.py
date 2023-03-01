import numpy as np

def get_hypercube_data(n,d, border=1, remove_subcube=False):
    data = border*2*(np.random.rand(n, d) - .5)
    if remove_subcube:
        filtered_data = []
        for point in data:
            if (point < 0).any():
                filtered_data.append(point)
        return np.array(filtered_data)
    else:
        return data

def get_corners(d):
    corners = []
    for i in range(2**d):
        corner = np.zeros(d)
        for j in range(d):
            if (i & (1 << j)) != 0: corner[j] = 1
            else: corner[j] = -1
        corners.append(corner)
    return np.array(corners)

def get_evenly_spaced_circle(radius, n_circles):
    angles = 2*np.pi*np.arange(n_circles)/n_circles
    means = np.stack((radius*np.cos(angles), radius*np.sin(angles)))
    return means.T

def get_cluster_data(n, d, k, variance=1/20, border=1):
    if k<=2**d:
        means = 0.6*get_corners(d)[:k]
    covariances = [variance*np.identity(d) for i in range(k)]
    data = []
    for i in range(k):
        data.extend(np.random.multivariate_normal(means[i], covariances[i], int(n/k)))
    filtered_data = []
    for point in data:
        if (np.abs(point) < border).all():
            filtered_data.append(point)
    return np.array(filtered_data)

def get_circle_data(n, d, n_circles, radius, variance=1/20, border=1):
    means = get_evenly_spaced_circle(radius, n_circles)
    covariances = [variance*np.identity(d) for i in range(n_circles)]
    data = []
    for i in range(n_circles):
        data.extend(np.random.multivariate_normal(means[i], covariances[i], int(n/n_circles)))
    filtered_data = []
    for point in data:
        if (np.abs(point) < border).all():
            filtered_data.append(point)
    return np.array(filtered_data)