#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include "common.h"


void usage(int argc, char** argv);
void verify(int* sol, int* ans, int n);
void prefix_sum(int* src, int* prefix, int n);
void prefix_sum_p1(int* src, int* prefix, int n);
void prefix_sum_p2(int* src, int* prefix, int n);


int main(int argc, char** argv)
{
    // get inputs
    uint32_t n = 1048576;
    unsigned int seed = time(NULL);
    if(argc > 2) {
        n = atoi(argv[1]); 
        seed = atoi(argv[2]);
    } else {
        usage(argc, argv);
        printf("using %"PRIu32" elements and time as seed\n", n);
    }


    // set up data 
    int* prefix_array = (int*) AlignedMalloc(sizeof(int) * n);  
    int* input_array = (int*) AlignedMalloc(sizeof(int) * n);
    srand(seed);
    for(int i = 0; i < n; i++) {
        input_array[i] = rand() % 100;
    }


    // set up timers
    uint64_t start_t;
    uint64_t end_t;
    InitTSC();


    // execute serial prefix sum and use it as ground truth
    start_t = ReadTSC();
    prefix_sum(input_array, prefix_array, n);
    end_t = ReadTSC();
    printf("Time to do O(N-1) prefix sum on a %"PRIu32" elements: %g (s)\n", 
           n, ElapsedTime(end_t - start_t));


    // execute parallel prefix sum which uses a NlogN algorithm
    int* input_array1 = (int*) AlignedMalloc(sizeof(int) * n);  
    int* prefix_array1 = (int*) AlignedMalloc(sizeof(int) * n);  
    memcpy(input_array1, input_array, sizeof(int) * n);
    start_t = ReadTSC();
    prefix_sum_p1(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do O(NlogN) //prefix sum on a %"PRIu32" elements: %g (s)\n",
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);

    
    // execute parallel prefix sum which uses a 2(N-1) algorithm
    memcpy(input_array1, input_array, sizeof(int) * n);
    memset(prefix_array1, 0, sizeof(int) * n);
    start_t = ReadTSC();
    prefix_sum_p2(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do 2(N-1) //prefix sum on a %"PRIu32" elements: %g (s)\n", 
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);


    // free memory
    AlignedFree(prefix_array);
    AlignedFree(input_array);
    AlignedFree(input_array1);
    AlignedFree(prefix_array1);


    return 0;
}

void usage(int argc, char** argv)
{
    fprintf(stderr, "usage: %s <# elements> <rand seed>\n", argv[0]);
}


void verify(int* sol, int* ans, int n)
{
    int err = 0;
    for(int i = 0; i < n; i++) {
        if(sol[i] != ans[i]) {
            err++;
        }
    }
    if(err != 0) {
        fprintf(stderr, "There was an error: %d\n", err);
    } else {
        fprintf(stdout, "Pass\n");
    }
}

void prefix_sum(int* src, int* prefix, int n)
{
    prefix[0] = src[0];
    for(int i = 1; i < n; i++) {
        prefix[i] = src[i] + prefix[i - 1];
    }
}

void prefix_sum_p1(int* src, int* prefix, int n) 
{
    int *t0;
    int *t1;
    int *tmp;
    // initialize the values for the back and forth read/write pattern for parallelization
    t0 = src;
    t1 = prefix;
    int depth_outer_loop = (int)log2(n); //ceil returns double

    for (int i=0; i < depth_outer_loop; i++) {
        // set the stride for the addition between the two copies of the list
        int stride = 1 << i; // 2^i 
        #pragma omp parallel for
        for (int j=0; j < n; j++) {
            // How do I just copy the first stride indices for each inner loop????
            if (j < stride) {
                t1[j] = t0[j];
            } else {
                t1[j] = t0[j] + t0[j-stride]; 
            }
        }
        tmp = t1;
        t1 = t0;
        t0 = tmp;
    }
    // Citation: Gemini helped me fix this error where prefix=t1 -> t0 after last loop summing in src instead of prefix
    if (t0 != prefix) {
        // If t0 isn't pointing to the original prefix array,
        // it means the final data is in src, so we copy it over.
        memcpy(prefix, t0, n * sizeof(int));
    }
}

void prefix_sum_p2(int* src, int* prefix, int n)
{
    // We need our own temp_cpy of the array to build the binary tree
    int* binary_tree = (int*) AlignedMalloc(sizeof(int) * n);
    memcpy(binary_tree, src, sizeof(int) * n);
    int depth_outer_loop = (int)log2(n);

    //Calculate prefix sum TRAVERSE 1
    for (int i=0; i < depth_outer_loop; i++) {
        int stride = 1 << i; //2^i is our stride for the binary tree indices

        #pragma omp parallel for
        for (int j=0; j < n; j += (2 * stride)) {
            // WHY are we allowed to store the result in the right child????
            binary_tree[j + ((2*stride) -1)] += binary_tree[j + stride -1]; // this is left and right child summed into right, continue with each stride
        }
    }

    // Second traversal start from top instead of bottom (slides)

    // Citation: Google Gemini says we must set the root to zero?
    binary_tree[n-1] = 0;

    for (int i = (depth_outer_loop - 1), i >= 0; i--) {
        int stride = 1 << i;

        #pragma omp parallel for
        for (int j = 0; j < n; j += (2 * stride)) {
            // work in reverse
            int left_child = j + stride - 1;
            int right_child = j + (2*stride) - 1;

            // if we swap the values and add we get an incremental prefix sum for the tree/list ({1,2}, {3,4}, ...) on final iteratio
            int tmp = binary_tree[left_child];
            binary_tree[left_child] = binary_tree[right_child];
            binary_tree[right_child] += tmp;
        }
    }

    // I thought this would work, but apparently the following pass over the list is needed to ensure correctness

    // OHHHHHHHH!!!!! we got rid of the actual value of the element from src
    #pragma omp parallel for
    for (int i=0; i<n; i++) {
        prefix[i] = binary_tree[i] + src[i];
    }

    // I guess I don't need the temp copy since we could treat prefix as the tree and then add src[i] in the final loop??
    AllignedFree(binary_tree);

    
}



