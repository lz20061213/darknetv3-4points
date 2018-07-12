#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes) {
    int i;
    layer l = {0};
    l.type = YOLO;

    l.n = n;  // anchor nums in this layer
    l.total = total;  // total anchor nums in this network
    l.batch = batch;
    l.h = h;
    l.w = w;
    //l.c = n*(classes + 4 + 1);
    l.c = n * (classes + 12 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(total * 2, sizeof(float));
    if (mask) l.mask = mask;
    else {
        l.mask = calloc(n, sizeof(int));
        for (i = 0; i < n; ++i) {
            l.mask[i] = i;
        }
    }
    l.bias_updates = calloc(n * 2, sizeof(float));
    //l.outputs = h*w*n*(classes + 4 + 1);
    l.outputs = h * w * l.c;
    l.inputs = l.outputs;
    //l.truths = 90*(4 + 1);
    l.truths = 90 * (12 + 1);
    l.delta = calloc(batch * l.outputs, sizeof(float));
    l.output = calloc(batch * l.outputs, sizeof(float));
    for (i = 0; i < total * 2; ++i) {
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "detection\n");
    srand(0);

    return l;
}

void resize_yolo_layer(layer *l, int w, int h) {
    l->w = w;
    l->h = h;

    //l->outputs = h*w*l->n*(l->classes + 4 + 1);
    l->outputs = h * w * l->n * (l->classes + 12 + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch * l->outputs * sizeof(float));
    l->delta = realloc(l->delta, l->batch * l->outputs * sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride) {
    box b;
    b.x = (i + x[index + 0 * stride]) / lw;
    b.y = (j + x[index + 1 * stride]) / lh;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    return b;
}

poly get_yolo_poly(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride) {
    poly p;
    p.x = (i + x[index + 0 * stride]) / lw;
    p.y = (j + x[index + 1 * stride]) / lh;
    p.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
    p.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    float cpx1, cpy1, cpx2, cpy2, cpx3, cpy3, cpx4, cpy4;
    cpx1 = p.x - p.w / 2;
    cpy1 = p.y;
    cpx2 = p.x;
    cpy2 = p.y - p.h / 2;
    cpx3 = p.x + p.w / 2;
    cpy3 = p.y;
    cpx4 = p.x;
    cpy4 = p.y + p.h / 2;
    p.px1 = cpx1;
    p.py1 = cpy1 + x[index + 5 * stride] * p.h / 2;
    p.px2 = cpx2 + x[index + 6 * stride] * p.w / 2;
    p.py2 = cpy2;
    p.px3 = cpx3;
    p.py3 = cpy3 + x[index + 9 * stride] * p.h / 2;
    p.px4 = cpx4 + x[index + 10 * stride] * p.w / 2;
    p.py4 = cpy4;

    //printf("predict: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", p.x, p.y, p.w, p.h, p.px1, p.py1, p.px2, p.py2, p.px3, p.py3, p.px4, p.py4);
    return p;
}

float delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h,
                     float *delta, float scale, int stride) {
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x * lw - i);
    float ty = (truth.y * lh - j);
    float tw = log(truth.w * w / biases[2 * n]);
    float th = log(truth.h * h / biases[2 * n + 1]);

    delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
    delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);
    delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
    delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);
    return iou;
}

float delta_yolo_poly(poly truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h,
                      float *delta, float scale, int stride) {
    poly pred = get_yolo_poly(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = poly_iou(pred, truth);

//    printf("%f\n", iou);
//    printf("trutu in yolo: %f %f %f %f %f %f %f %f\n", truth.px1, truth.py1, truth.px2, truth.py2,
//           truth.px3, truth.py3, truth.px4, truth.py4);
//    printf("some thing: %f, %f, %d, %d, %f, %f, %d, %d, %f, %f\n", truth.x, truth.y, lw, lh, truth.w, truth.h, w, h, biases[2 * n], biases[2 * n + 1]);

    float tx = (truth.x * lw - i);
    float ty = (truth.y * lh - j);
    float tw = log(truth.w * w / biases[2 * n]);
    float th = log(truth.h * h / biases[2 * n + 1]);
    float cpx1, cpy1, cpx2, cpy2, cpx3, cpy3, cpx4, cpy4;
    cpx1 = truth.x - truth.w / 2; // fixed
    cpy1 = truth.y;
    cpx2 = truth.x;
    cpy2 = truth.y - truth.h / 2; // fixed
    cpx3 = truth.x + truth.w / 2;  // fixed
    cpy3 = truth.y;
    cpx4 = truth.x;
    cpy4 = truth.y + truth.h / 2;  // fixed

    //printf("center anchor: %f, %f, %f, %f, %f, %f, %f, %f\n", cpx1, cpy1, cpx2, cpy2, cpx3, cpy3, cpx4, cpy4);

    float tpx1 = (truth.px1 - cpx1) / truth.w * 2;
    float tpy1 = (truth.py1 - cpy1) / truth.h * 2;
    float tpx2 = (truth.px2 - cpx2) / truth.w * 2;
    float tpy2 = (truth.py2 - cpy2) / truth.h * 2;
    float tpx3 = (truth.px3 - cpx3) / truth.w * 2;
    float tpy3 = (truth.py3 - cpy3) / truth.h * 2;
    float tpx4 = (truth.px4 - cpx4) / truth.w * 2;
    float tpy4 = (truth.py4 - cpy4) / truth.h * 2;

    //printf("predict: %f, %f, %f, %f, %f, %f, %f, %f\n", tpx1, tpy1, tpx2, tpy2, tpx3, tpy3, tpx4, tpy4);

    delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
    delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);
    delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
    delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);
    delta[index + 4 * stride] = 0;
    delta[index + 5 * stride] = scale * (tpy1 - x[index + 5 * stride]);
    delta[index + 6 * stride] = scale * (tpx2 - x[index + 6 * stride]);
    delta[index + 7 * stride] = 0;
    delta[index + 8 * stride] = 0;
    delta[index + 9 * stride] = scale * (tpy3 - x[index + 9 * stride]);
    delta[index + 10 * stride] = scale * (tpx4 - x[index + 10 * stride]);
    delta[index + 11 * stride] = 0;

    return iou;
}

void delta_yolo_class(float *output, float *delta, int index, int class, int classes, int stride, float *avg_cat) {
    int n;
    if (delta[index]) {
        delta[index + stride *
        class] = 1 - output[index + stride *
        class];
        if (avg_cat) *avg_cat += output[index + stride *
        class];
        return;
    }
    for (n = 0; n < classes; ++n) {
        delta[index + stride * n] = ((n ==
        class)?1 : 0) -output[index + stride * n];
        if (n == class &&avg_cat) *avg_cat += output[index + stride * n];
    }
}

static int entry_index(layer l, int batch, int location, int entry) {
    int n = location / (l.w * l.h);
    int loc = location % (l.w * l.h);
    //return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
    return batch * l.outputs + n * l.w * l.h * (12 + l.classes + 1) + entry * l.w * l.h + loc;
}

void forward_yolo_layer(const layer l, network net) {
    int i, j, b, t, n;
    memcpy(l.output, net.input, l.outputs * l.batch * sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b) {
        for (n = 0; n < l.n; ++n) {
            int index = entry_index(l, b, n * l.w * l.h, 0);   // bbox/poly active, linear
            activate_array(l.output + index, 2 * l.w * l.h, LOGISTIC);  // x, y activate
            index = entry_index(l, b, n * l.w * l.h, 4);
            activate_array(l.output + index, 8 * l.w * l.h, TANH);  // px1, py1, px2, py2, px3, py3, px4, py4 activate
            //index = entry_index(l, b, n*l.w*l.h, 4);  // class active
            index = entry_index(l, b, n * l.w * l.h, 12);
            activate_array(l.output + index, (1 + l.classes) * l.w * l.h, LOGISTIC);
        }
    }
#endif

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if (!net.train) return;
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int poly_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 0);
                    //printf("%d\n", poly_index);
                    poly pred = get_yolo_poly(l.output, l.biases, l.mask[n], poly_index, i, j, l.w, l.h, net.w, net.h,
                                              l.w * l.h);
                    float best_iou = 0;
                    int best_t = 0;
                    for (t = 0; t < l.max_boxes; ++t) {
                        poly truth = float_to_poly(net.truth + t * (12 + 1) + b * l.truths, 1);
//                        printf("truth in yolo: %f %f %f %f %f %f %f %f\n", truth.px1, truth.py1, truth.px2, truth.py2,
//           truth.px3, truth.py3, truth.px4, truth.py4);
                        if (!truth.x) break;
                        float iou = poly_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    int obj_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 12);
                    avg_anyobj += l.output[obj_index];
                    l.delta[obj_index] = 0 - l.output[obj_index];
                    if (best_iou > l.ignore_thresh) {
                        l.delta[obj_index] = 0;
                    }
                    if (best_iou > l.truth_thresh) {
                        l.delta[obj_index] = 1 - l.output[obj_index];

                        //int class = net.truth[best_t*(4 + 1) + b*l.truths + 4];
                        int
                        class = net.truth[best_t * (12 + 1) + b * l.truths + 12];
                        if (l.map) class = l.map[class];
                        //int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        int class_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 12 + 1);
                        delta_yolo_class(l.output, l.delta, class_index,
                        class, l.classes, l.w * l.h, 0);
                        poly truth = float_to_poly(net.truth + best_t * (12 + 1) + b * l.truths, 1);
                        delta_yolo_poly(truth, l.output, l.biases, l.mask[n], poly_index, i, j, l.w, l.h, net.w, net.h,
                                        l.delta, (2 - truth.w * truth.h), l.w * l.h);
                    }
                }
            }
        }
        for (t = 0; t < l.max_boxes; ++t) {
            poly truth = float_to_poly(net.truth + t * (12 + 1) + b * l.truths, 1);

            if (!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            poly truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for (n = 0; n < l.total; ++n) {
                poly pred = {0};
                pred.w = l.biases[2 * n] / net.w;
                pred.h = l.biases[2 * n + 1] / net.h;
                float iou = poly_iou(pred, truth_shift);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_n = n;
                }
            }

            int mask_n = int_index(l.mask, best_n, l.n);
            if (mask_n >= 0) {
                int poly_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 0);
                float iou = delta_yolo_poly(truth, l.output, l.biases, best_n, poly_index, i, j, l.w, l.h, net.w, net.h,
                                            l.delta, (2 - truth.w * truth.h), l.w * l.h);

                //int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                int obj_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 12);
                avg_obj += l.output[obj_index];
//                printf("avg_obj in yolo_layer: %f\n", avg_obj);
                l.delta[obj_index] = 1 - l.output[obj_index];

                //int class = net.truth[t*(4 + 1) + b*l.truths + 4];
                int
                class = net.truth[t * (12 + 1) + b * l.truths + 12];
                if (l.map) class = l.map[class];
                //int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                int class_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 12 + 1);
                delta_yolo_class(l.output, l.delta, class_index,
                class, l.classes, l.w * l.h, &avg_cat);

                ++count;
                ++class_count;
                if (iou > .5) recall += 1;
                if (iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index,
           avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w * l.h * l.n * l.batch),
           recall / count, recall75 / count, count);
}

void backward_yolo_layer(const layer l, network net) {
    axpy_cpu(l.batch * l.inputs, 1, l.delta, 1, net.delta, 1);
}


void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative) {
    int i;
    int new_w = 0;
    int new_h = 0;
    if (((float) netw / w) < ((float) neth / h)) {
        new_w = netw;
        new_h = (h * netw) / w;
    } else {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    for (i = 0; i < n; ++i) {
        box b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / ((float) new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / ((float) new_h / neth);
        b.w *= (float) netw / new_w;
        b.h *= (float) neth / new_h;
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}


void correct_yolo_polys(detectionp *dets, int n, int w, int h, int netw, int neth, int relative) {
    int i;
    int new_w = 0;
    int new_h = 0;
    if (((float) netw / w) < ((float) neth / h)) {
        new_w = netw;
        new_h = (h * netw) / w;
    } else {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    //printf("%d, %d\n", netw, new_w);
    for (i = 0; i < n; ++i) {
        poly p = dets[i].ppoly;
        p.x = (p.x - (netw - new_w) / 2. / netw) / ((float) new_w / netw);
        p.y = (p.y - (neth - new_h) / 2. / neth) / ((float) new_h / neth);
        p.w *= (float) netw / new_w;
        p.h *= (float) neth / new_h;
        p.px1 = (p.px1 - (netw - new_w) / 2. / netw) / ((float) new_w / netw);
        p.py1 = (p.py1 - (neth - new_h) / 2. / neth) / ((float) new_h / neth);
        p.px2 = (p.px2 - (netw - new_w) / 2. / netw) / ((float) new_w / netw);
        p.py2 = (p.py2 - (neth - new_h) / 2. / neth) / ((float) new_h / neth);
        p.px3 = (p.px3 - (netw - new_w) / 2. / netw) / ((float) new_w / netw);
        p.py3 = (p.py3 - (neth - new_h) / 2. / neth) / ((float) new_h / neth);
        p.px4 = (p.px4 - (netw - new_w) / 2. / netw) / ((float) new_w / netw);
        p.py4 = (p.py4 - (neth - new_h) / 2. / neth) / ((float) new_h / neth);
        if (!relative) {
            p.x *= w;
            p.w *= w;
            p.y *= h;
            p.h *= h;
            p.px1 *= w;
            p.px2 *= w;
            p.px3 *= w;
            p.px4 *= w;
            p.py1 *= h;
            p.py2 *= h;
            p.py3 *= h;
            p.py4 *= h;
        }
        dets[i].ppoly = p;
    }
}


int yolo_num_detections(layer l, float thresh) {
    int i, n;
    int count = 0;
    for (i = 0; i < l.w * l.h; ++i) {
        for (n = 0; n < l.n; ++n) {
            int obj_index = entry_index(l, 0, n * l.w * l.h + i, 12);
            if (l.output[obj_index] > thresh) {
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l) {
    int i, j, n, z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w / 2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for (z = 0; z < l.classes + 12 + 1; ++z) {
                    int i1 = z * l.w * l.h * l.n + n * l.w * l.h + j * l.w + i;
                    int i2 = z * l.w * l.h * l.n + n * l.w * l.h + j * l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if (z == 0) {
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for (i = 0; i < l.outputs; ++i) {
        l.output[i] = (l.output[i] + flip[i]) / 2.;
    }
}

int
get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detectionp *dets) {
    int i, j, n;
    float *predictions = l.output;
    if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w * l.h; ++i) {
        int row = i / l.w;
        int col = i % l.w;
        for (n = 0; n < l.n; ++n) {
            int obj_index = entry_index(l, 0, n * l.w * l.h + i, 12);
            float objectness = predictions[obj_index];
            if (objectness <= thresh) continue;
            int poly_index = entry_index(l, 0, n * l.w * l.h + i, 0);
            dets[count].ppoly = get_yolo_poly(predictions, l.biases, l.mask[n], poly_index, col, row, l.w, l.h, netw,
                                              neth, l.w * l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for (j = 0; j < l.classes; ++j) {
                int class_index = entry_index(l, 0, n * l.w * l.h + i, 12 + 1 + j);
                float prob = objectness * predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    correct_yolo_polys(dets, count, w, h, netw, neth, relative);
    return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);  // x, y activate
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_gpu(l.output_gpu + index, 8*l.w*l.h, TANH);  // px1, py1, px2, py2, px3, py3, px4, py4 activate
            //index = entry_index(l, b, n*l.w*l.h, 4);
            index = entry_index(l, b, n*l.w*l.h, 12);
            activate_array_gpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_yolo_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_yolo_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

