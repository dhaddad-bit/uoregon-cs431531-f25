#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "common.h"
#include <inttypes.h>
#include <time.h>

#define PI 3.1415926535

void usage(int argc, char** argv);
double calcPi_Serial(int num_steps);
double calcPi_P1(int num_steps);
double calcPi_P2(int num_steps);

int main(int argc, char** argv)
{
    // get input values
    uint64_t num_steps = 100000;
    if (argc > 1) {
        num_steps = atoll(argv[1]); //Changed to atoll FROM atoi for 64-bit (long)
    } else {
        usage(argc, argv);
    }
    fprintf(stdout, "The first 10 digits of Pi are %0.10f\n", PI);
    
    // timer setup
    uint64_t start_t; 
    uint64_t end_t;  
    InitTSC();

    // calculate in serial
    start_t = ReadTSC();
    double Pi0 = calcPi_Serial(num_steps);
    end_t = ReadTSC();
    printf("Time to calculate Pi serially with %"PRIu64" guesses is: %g\n",
           num_steps, ElapsedTime(end_t - start_t));
    printf("Pi is %0.10f\n", Pi0);

    // calculate in parallel with integration
    start_t = ReadTSC();
    double Pi1 = calcPi_P1(num_steps);
    end_t = ReadTSC();
    printf("Time to calculate Pi in // with %"PRIu64" guesses is: %g\n",
           num_steps, ElapsedTime(end_t - start_t));
    printf("Pi is %0.10f\n", Pi1);  

    // calculate in parallel with Monte Carlo
    start_t = ReadTSC();
    double Pi2 = calcPi_P2(num_steps);
    end_t = ReadTSC();
    printf("Time to calculate Pi in // with %"PRIu64" guesses is: %g\n",
           num_steps, ElapsedTime(end_t - start_t));
    printf("Pi is %0.10f\n", Pi2);

    return 0;
}

void usage(int argc, char** argv)
{
    fprintf(stdout, "usage: %s <# steps>\n", argv[0]);
}  

double calcPi_Serial(int num_steps) 
{
    double step;
    double x, sum = 0.0;

    // step is the width of each rectangle
    step = 1.0 / (double) num_steps;

    // Loop through all rectangles
    for (uint64_t i = 0; i < num_steps; i++) {
        // x is the midpoint of the rectangle's base
        x = (i + 0.5) * step; // One mult per parallel region vs synchronized addition if declared outside?
        // The height of the rectangle is (1.0*4) / (1.0 + x*x)
        // Add its area (height * width) to the sum
        sum = sum + 4.0 / (1.0 + x * x);
    }

    // The final value of pi is the sum of areas * the width of each rectangle
    double pi = step * sum;
    return pi;
}

double calcPi_P1(int num_steps)
{
    double step;
    double sum = 0.0;
    double pi = 0.0;

    step = 1.0 / (double) num_steps;

    // We need to split the for-loop iterations among all available threads.
    // Sum is forced to be shared by all threads (lecture)
    #pragma omp parallel for
    for (uint64_t i = 0; i < num_steps; i++) {
        double x = (i + 0.5) * step; // One mult per parallel region vs synchronized addition if declared outside?
        double h = 4.0 / (1.0 + x * x);

        /* 
        // Entering Critical section (no longer needed but kept for clarity))
        #pragma omp critical
        {
            sum = sum + h;
        }
        // Exiting Critical section 
        */
        #pragma omp atomic // Protects sum from race condition
            sum = sum + h;
        
    }

    pi = step * sum;
    return pi;
}

double calcPi_P2(int num_steps)
{
    double pi=0.0;
    int in_circle = 0;

    // Begin the parallel region so each thread gets private in_circle
    #pragma omp parallel reduction(+:in_circle)
    {
        // NEEED OWN COPY OF SEED BEFORE LOOP?
        unsigned int seed = (unsigned int)time(NULL) + (unsigned int)omp_get_thread_num(); //Google helped with this line

        // now we can start a for loop and DISTRIBUTE iterations 
        // (embarassingly Parallel) to each thread
        #pragma omp for // let OpenMP handle following for loop distribution (no other specifiers????)
        for (uint64_t i = 0; i < num_steps; i++) {
            // Now generate random (x,y point)
            double x = (double)rand_r(&seed) / (double)RAND_MAX; //rand_r is reentrant version of rand
            double y = (double)rand_r(&seed) / (double)RAND_MAX; // Helped by google for more randomized values

            // Check if point is in unit circle
            if ((x*x + y*y) <= 1.0) {
                in_circle++;
            }
        }
    } // THIS ENDS ALL PARALLEL REGIONS DEFINED
    // Calculate Pi based off of accumulated in_circle values which were reduced
    pi = 4.0 * ((double)in_circle/(double)num_steps); 
    return pi;
}