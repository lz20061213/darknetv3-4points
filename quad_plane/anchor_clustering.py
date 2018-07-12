# -*- coding: utf-8 -*-
import numpy as np
from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import cv2
import re, os
import xml.etree.ElementTree as ET
import time

# Original code @ferada http://codereview.stackexchange.com/questions/128315/k-means-clustering-algorithm-implementation

best_clusters = []
best_avg_iou = 0
best_avg_iou_iteration = 0


def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2.
    l2 = x2 - w2 / 2.
    left = l1 if l1 > l2 else l2
    r1 = x1 + w1 / 2.
    r2 = x2 + w2 / 2.
    right = r1 if r1 < r2 else r2
    return right - left


def intersection(a, b):
    w = overlap(a[0], a[2], b[0], b[2])
    h = overlap(a[1], a[3], b[1], b[3])
    if w < 0 or h < 0:
        return 0
    return w * h


def area(x):
    return x[2] * x[3]


def union(a, b):
    return area(a) + area(b) - intersection(a, b)


def iou(a, b):
    return intersection(a, b) / union(a, b)


def niou(a, b):
    return 1. - iou(a, b)


def equals(points1, points2):
    if len(points1) != len(points2):
        return False

    for point1, point2 in izip(points1, points2):
        if point1[0] != point2[0] or point1[1] != point2[1] or point1[2] != point2[2] or point1[3] != point2[3]:
            return False

    return True


def compute_centroids(clusters):
    return [np.mean(cluster, axis=0) for cluster in clusters]


def closest_centroid(point, centroids):
    min_distance = float('inf')
    belongs_to_cluster = None
    for j, centroid in enumerate(centroids):
        dist = niou(point, centroid)

        if dist < min_distance:
            min_distance = dist
            belongs_to_cluster = j

    return belongs_to_cluster, min_distance


def kmeans(k, centroids, points, iter_count=0, iteration_cutoff=25):
    global best_clusters
    global best_avg_iou
    global best_avg_iou_iteration
    clusters = [[] for _ in range(k)]
    clusters_iou = []
    clusters_niou = []

    for point in points:
        idx, dist = closest_centroid(point, centroids)
        clusters[idx].append(point)
        clusters_niou.append(dist)
        clusters_iou.append(1. - dist)

    avg_iou = np.mean(clusters_iou)
    if avg_iou > best_avg_iou:
        best_avg_iou = avg_iou
        best_clusters = clusters
        best_avg_iou_iteration = iter_count

    print("Iteration {}".format(iter_count))
    print("Average iou to closest centroid = {}".format(avg_iou))
    print("Sum of all distances (cost) = {}\n".format(np.sum(clusters_niou)))

    new_centroids = compute_centroids(clusters)

    for i in range(len(new_centroids)):
        shift = niou(centroids[i], new_centroids[i])
        print("Cluster {} size: {}".format(i, len(clusters[i])))
        print("Centroid {} distance shift: {}\n\n".format(i, shift))

    if iter_count < best_avg_iou_iteration + iteration_cutoff:
        kmeans(k, new_centroids, points, iter_count + 1, iteration_cutoff)

    return


def plot_anchors(pascal_anchors, coco_anchors):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.set_ylim([0, 500])
    ax1.set_xlim([0, 900])

    for i in range(len(pascal_anchors)):
        if area(pascal_anchors[i]) > area(coco_anchors[i]):
            bbox1 = pascal_anchors[i]
            color1 = "white"
            bbox2 = coco_anchors[i]
            color2 = "blue"
        else:
            bbox1 = coco_anchors[i]
            color1 = "blue"
            bbox2 = pascal_anchors[i]
            color2 = "white"

        lower_right_x = bbox1[0] - (bbox1[2] / 2.0)
        lower_right_y = bbox1[1] - (bbox1[3] / 2.0)

        ax1.add_patch(
            patches.Rectangle(
                (lower_right_x, lower_right_y),  # (x,y)
                bbox1[2],  # width
                bbox1[3],  # height
                facecolor=color1
            )
        )

        lower_right_x = bbox2[0] - (bbox2[2] / 2.0)
        lower_right_y = bbox2[1] - (bbox2[3] / 2.0)

        ax1.add_patch(
            patches.Rectangle(
                (lower_right_x, lower_right_y),  # (x,y)
                bbox2[2],  # width
                bbox2[3],  # height
                facecolor=color2
            )
        )
    plt.show()


def getBBoxes(xml_path, dim=416):
    bboxes = []

    root = ET.parse(xml_path)
    sizes = root.find('size')
    width = int(sizes.find('width').text)
    height = int(sizes.find('height').text)

    objects = root.findall('object')
    for obj in objects:
        bndbox = obj.find('polygon')
        x1 = float(bndbox.find('x1').text)
        y1 = float(bndbox.find('y1').text)
        x2 = float(bndbox.find('x2').text)
        y2 = float(bndbox.find('y2').text)
        x3 = float(bndbox.find('x3').text)
        y3 = float(bndbox.find('y3').text)
        x4 = float(bndbox.find('x4').text)
        y4 = float(bndbox.find('y4').text)
        xs = [x1, x2, x3, x4]
        ys = [y1, y2, y3, y4]
        xmin = np.min(xs, axis=0)
        xmax = np.max(xs, axis=0)
        ymin = np.min(ys, axis=0)
        ymax = np.max(ys, axis=0)
        w = (xmax - xmin) * dim / width
        h = (ymax - ymin) * dim / height
        bboxes.append([0, 0, w, h])

    return bboxes


if __name__ == "__main__":
    # Load pascal and coco label data (original coordinates)
    # shape: [[x1,y1,w1,h1],...,[xn,yn,wn,hn]]
    bbox_data = []
    root_path = '/home/yangruyin/data/PLANE/VOCdevkit07/'  # image root path
    images_path = os.path.join(root_path, 'JPEGImages')
    xmls_path = os.path.join(root_path, 'Annotations')
    lists_path = os.path.join(root_path, 'ImageSets/Main')
    listfid = open(os.path.join(lists_path, 'trainval.txt'), 'r')  # train_file list
    lines = listfid.readlines()
    print 'Load bboxes ...'
    start_time = time.time()
    for i in range(len(lines)):
        end_time = time.time()
        if end_time - start_time > 1:
            print '{}/{}'.format(i + 1, len(lines))
            start_time = time.time()
        filename = lines[i].strip()
        # print filename
        # img_path = os.path.join(images_path, filename+'.jpg')
        xml_path = os.path.join(xmls_path, filename + '.xml')
        bboxes = getBBoxes(xml_path)
        bbox_data.extend(bboxes)
    print 'Finish bboxes loading ...'

    bbox_data = np.array(bbox_data)

    # Set x,y coordinates to origin, defined in getBBoxes
    # for i in range(len(hand_data)):
    # hand_data[i][0] = 0
    # hand_data[i][1] = 0

    # k-means picking the first k points as centroids
    k = 9
    np.random.shuffle(bbox_data)
    centroids = bbox_data[:k]
    kmeans(k, centroids, bbox_data)

    # Get anchor boxes from best clusters
    bbox_anchors = np.asarray([np.mean(cluster, axis=0) for cluster in best_clusters])

    # Sort by width
    bbox_anchors = bbox_anchors[bbox_anchors[:, 2].argsort()]

    # scaled pascal anchors from cfg (for comparison): 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
    print("\nk-means clustering pascal anchor points (original coordinates) \
    \nFound at iteration {} with best average IoU: {} \
    \n{}".format(best_avg_iou_iteration, best_avg_iou, bbox_anchors))
