__device__ float pick_min(float *array, int vec_size) {
    int i;
    float mmin = 3.4e+38;
    for(i = 0; i < vec_size; i++) {
        mmin = fminf(array[i], mmin);
    }
    return mmin;
}

__device__ float pick_max(float *array, int vec_size) {
    int i;
    float mmax = -3.4e+38;
    for(i = 0; i < vec_size; i++) {
        mmax = fmaxf(array[i], mmax);
    }
    return mmax;
}


__device__ float hausdorff(float *a, float *b, int vec_size) {
    int i;
    float c1 = -3.4e+38;
    float c2 = -3.4e+38;

    float b_min = pick_min(b, vec_size);
    float a_max = pick_max(a, vec_size);

    for(i = 0; i < vec_size; i++) {
        float buf1 = a[i] - b_min;
        float buf2 = b[i] - a_max;

        if(buf1 > c1) {
            c1 = buf1;
        }
        if(buf2 > c2) {
            c2 = buf2;
        }
    }
    return fmaxf(c1, c2);
}

__global__ void nn_hausdorff(float *dest, int vec_size, int n_vecs, float *vecs) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    int row = id / n_vecs, column = id % n_vecs;
    dest[id] = hausdorff(&vecs[row * vec_size], &vecs[column * vec_size], vec_size);
}