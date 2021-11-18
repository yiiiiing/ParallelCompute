/* COMP.CE.350 Parallelization Excercise 2021
   Copyright (c) 2016 Matias Koskela matias.koskela@tut.fi
                      Heikki Kultala heikki.kultala@tut.fi
                      Topi Leppanen  topi.leppanen@tuni.fi

VERSION 1.1 - updated to not have stuck satellites so easily
VERSION 1.2 - updated to not have stuck satellites hopefully at all.
VERSION 19.0 - make all satellites affect the color with weighted average.
               add physic correctness check.
VERSION 20.0 - relax physic correctness check
*/

// Example compilation on linux
// no optimization:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm
// most optimizations: gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2
// +vectorization +vectorize-infos: gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec
// +math relaxation:  gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec -ffast-math
// prev and OpenMP:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec -ffast-math -fopenmp
// prev and OpenCL:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec -ffast-math -fopenmp -lOpenCL

// Example compilation on macos X
// no optimization:   gcc -o parallel parallel.c -std=c99 -framework GLUT -framework OpenGL
// most optimization: gcc -o parallel parallel.c -std=c99 -framework GLUT -framework OpenGL -O3



#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h> // printf
#include <math.h> // INFINITY
#include <stdlib.h>
#include <string.h>

// Window handling includes
#ifndef __APPLE__
#include <GL/gl.h>
#include <GL/glut.h>
#else
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#endif
// These are used to decide the window size
#define WINDOW_HEIGHT 1024
#define WINDOW_WIDTH  1024

// The number of satellites can be changed to see how it affects performance.
// Benchmarks must be run with the original number of satellites
#define SATELLITE_COUNT 64

// These are used to control the satellite movement
#define SATELLITE_RADIUS 3.16f
#define MAX_VELOCITY 0.1f
#define GRAVITY 1.0f
#define DELTATIME 32
#define PHYSICSUPDATESPERFRAME 100000

// Some helpers to window size variables
#define SIZE WINDOW_WIDTH*WINDOW_HEIGHT
#define HORIZONTAL_CENTER (WINDOW_WIDTH / 2)
#define VERTICAL_CENTER (WINDOW_HEIGHT / 2)

// Is used to find out frame times
int previousFrameTimeSinceStart = 0;
int previousFinishTime = 0;
unsigned int frameNumber = 0;
unsigned int seed = 0;

// Stores 2D data like the coordinates
typedef struct{
   float x;
   float y;
} floatvector;

// Stores 2D data like the coordinates
typedef struct{
   double x;
   double y;
} doublevector;

// Stores rendered colors. Each float may vary from 0.0f ... 1.0f
typedef struct{
   float red;
   float green;
   float blue;
} color;

// Stores the satellite data, which fly around black hole in the space
typedef struct{
   color identifier;
   floatvector position;
   floatvector velocity;
} satellite;

// Pixel buffer which is rendered to the screen
color* pixels;

// Pixel buffer which is used for error checking
color* correctPixels;

// Buffer for all satellites in the space
satellite* satellites;
satellite* backupSatelites;




// ## You may add your own variables here ##
int totalTime_allFrames;
int physicsTime_allFrames;
int graphicsTime_allFrames;
#define RUNNING_FRAMES 20

// ## You may add your own initialization routines here ##
void init(){


}

// ## You are asked to make this code parallel ##
// Physics engine loop. (This is called once a frame before graphics engine) 
// Moves the satellites based on gravity
// This is done multiple times in a frame because the Euler integration 
// is not accurate enough to be done only once
void parallelPhysicsEngine(){

   // double precision required for accumulation inside this routine,
   // but float storage is ok outside these loops.  
   doublevector tmpPosition[SATELLITE_COUNT];
   doublevector tmpVelocity[SATELLITE_COUNT];
   for (int i = 0; i < SATELLITE_COUNT; ++i) {
   tmpPosition[i].x = satellites[i].position.x;
   tmpPosition[i].y = satellites[i].position.y;
   tmpVelocity[i].x = satellites[i].velocity.x;
   tmpVelocity[i].y = satellites[i].velocity.y;
   }

   // Physics iteration loop
   //#pragma omp parallel num_threads(3)
   {
      for(int physicsUpdateIndex = 0; 
      physicsUpdateIndex < PHYSICSUPDATESPERFRAME;
      ++physicsUpdateIndex)
      {
            
         // Physics satellite loop
         //#pragma omp parallel for
         for(int i = 0; i < SATELLITE_COUNT; ++i){

            // Distance to the blackhole (bit ugly code because C-struct cannot have member functions)
            doublevector positionToBlackHole = {.x = tmpPosition[i].x -
               HORIZONTAL_CENTER, .y = tmpPosition[i].y - VERTICAL_CENTER};
            double distToBlackHoleSquared =
               positionToBlackHole.x * positionToBlackHole.x +
               positionToBlackHole.y * positionToBlackHole.y;
            double distToBlackHole = sqrt(distToBlackHoleSquared);

            // Gravity force
            doublevector normalizedDirection = {
               .x = positionToBlackHole.x / distToBlackHole,
               .y = positionToBlackHole.y / distToBlackHole};
               double accumulation = GRAVITY / distToBlackHoleSquared;

            // Delta time is used to make velocity same despite different FPS
            // Update velocity based on force
         
            tmpVelocity[i].x -= accumulation * normalizedDirection.x *
               DELTATIME / PHYSICSUPDATESPERFRAME;
            
            tmpVelocity[i].y -= accumulation * normalizedDirection.y *
               DELTATIME / PHYSICSUPDATESPERFRAME;

            // Update position based on velocity
            
            tmpPosition[i].x += tmpVelocity[i].x * DELTATIME / PHYSICSUPDATESPERFRAME;
            
            tmpPosition[i].y += tmpVelocity[i].y * DELTATIME / PHYSICSUPDATESPERFRAME;
         }      
            
      }
   }
   

   // double precision required for accumulation inside this routine,
   // but float storage is ok outside these loops.
   // copy back the float storage. 
   for (int i = 0; i < SATELLITE_COUNT; ++i) {
      satellites[i].position.x = tmpPosition[i].x;
      satellites[i].position.y = tmpPosition[i].y;
      satellites[i].velocity.x = tmpVelocity[i].x;
      satellites[i].velocity.y = tmpVelocity[i].y;
   }
}

// ## You are asked to make this code parallel ##
// Rendering loop (This is called once a frame after physics engine) 
// Decides the color for each pixel.
void parallelGraphicsEngine(){

   #pragma opm paralle num_threads(2)
   {
      //printf("thread number is %d\n", omp_get_num_threads());
      // Graphics pixel loop
      #pragma omp parallel for schedule(dynamic) 
      for(int i = 0 ;i < SIZE; ++i) {
         // Row wise ordering
         floatvector pixel = {.x = i % WINDOW_WIDTH, .y = i / WINDOW_WIDTH};

         // This color is used for coloring the pixel
         color renderColor = {.red = 0.f, .green = 0.f, .blue = 0.f};

         // Find closest satellite
         float shortestDistance = INFINITY;

         //float weights = 0.f;
         // array of weights
         float weightsArray[SATELLITE_COUNT];
         float distancesArray[SATELLITE_COUNT];

         int hitsSatellite = 0;

         // First Graphics satellite loop: Find the closest satellite.    
         for (int j = 0; j < SATELLITE_COUNT; ++j)
         {
            floatvector difference = {.x = pixel.x - satellites[j].position.x,
                                       .y = pixel.y - satellites[j].position.y};
            distancesArray[j] = sqrt(difference.x * difference.x + 
                                    difference.y * difference.y);

            if(distancesArray[j] < SATELLITE_RADIUS) {
               renderColor.red = 1.0f;
               renderColor.green = 1.0f;
               renderColor.blue = 1.0f;
               hitsSatellite = 1;
            } else {
               weightsArray[j] = 1.0f / (distancesArray[j]*distancesArray[j]*distancesArray[j]*distancesArray[j]);
            }
         }
         
         // Second graphics loop: Calculate the color based on distance to every satellite.
         if (!hitsSatellite) {
            // sum weights array
            // and find the min distance in distancesArray
            float weights = 0;
            for (int j = 0; j < SATELLITE_COUNT; ++j)
            {
               weights += weightsArray[j];
            }

            for (int j = 0; j < SATELLITE_COUNT; ++j)
            {
               if (distancesArray[j] < shortestDistance){
                  shortestDistance = distancesArray[j];
                  renderColor = satellites[j].identifier;
               }
            }
            
            color colorsArray[SATELLITE_COUNT];
            for(int j = 0; j < SATELLITE_COUNT; ++j){
               colorsArray[j].red = (satellites[j].identifier.red *
                                    weightsArray[j] /weights) * 3.0f;

               colorsArray[j].green = (satellites[j].identifier.green *
                                       weightsArray[j] / weights) * 3.0f;

               colorsArray[j].blue = (satellites[j].identifier.blue *
                                    weightsArray[j] / weights) * 3.0f;
            }

            // sum all the color in colorsArray
            float red = 0.f, green = 0.f, blue = 0.f;
            #pragma omp parallel for reduction(+: red, green, blue)
            for (int j = 0; j < SATELLITE_COUNT; ++j)
            {
               red += colorsArray[j].red;
               green += colorsArray[j].green;
               blue += colorsArray[j].blue;
            }
            renderColor.red += red;
            renderColor.blue += blue;
            renderColor.green += green;
         }
         
         pixels[i] = renderColor;
      }
   }
    
}

// ## You may add your own destrcution routines here ##
void destroy(){


}




////////////////////////////////////////////////
// ¤¤ TO NOT EDIT ANYTHING AFTER THIS LINE ¤¤ //
////////////////////////////////////////////////

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Sequential rendering loop used for finding errors
void sequentialGraphicsEngine(){

    // Graphics pixel loop
    for(int i = 0 ;i < SIZE; ++i) {

      // Row wise ordering
      floatvector pixel = {.x = i % WINDOW_WIDTH, .y = i / WINDOW_WIDTH};

      // This color is used for coloring the pixel
      color renderColor = {.red = 0.f, .green = 0.f, .blue = 0.f};

      // Find closest satellite
      float shortestDistance = INFINITY;

      float weights = 0.f;
      int hitsSatellite = 0;

      // First Graphics satellite loop: Find the closest satellite.
      for(int j = 0; j < SATELLITE_COUNT; ++j){
         floatvector difference = {.x = pixel.x - satellites[j].position.x,
                                   .y = pixel.y - satellites[j].position.y};
         float distance = sqrt(difference.x * difference.x + 
                               difference.y * difference.y);

         if(distance < SATELLITE_RADIUS) {
            renderColor.red = 1.0f;
            renderColor.green = 1.0f;
            renderColor.blue = 1.0f;
            hitsSatellite = 1;
            break;
         } else {
            float weight = 1.0f / (distance*distance*distance*distance);
            weights += weight;
            if(distance < shortestDistance){
               shortestDistance = distance;
               renderColor = satellites[j].identifier;
            }
         }
      }

      // Second graphics loop: Calculate the color based on distance to every satellite.
      if (!hitsSatellite) {
         for(int j = 0; j < SATELLITE_COUNT; ++j){
            floatvector difference = {.x = pixel.x - satellites[j].position.x,
                                      .y = pixel.y - satellites[j].position.y};
            float dist2 = (difference.x * difference.x +
                           difference.y * difference.y);
            float weight = 1.0f/(dist2* dist2);

            renderColor.red += (satellites[j].identifier.red *
                                weight /weights) * 3.0f;

            renderColor.green += (satellites[j].identifier.green *
                                  weight / weights) * 3.0f;

            renderColor.blue += (satellites[j].identifier.blue *
                                 weight / weights) * 3.0f;
         }
      }
      correctPixels[i] = renderColor;
    }
}

void sequentialPhysicsEngine(satellite *s){

   // double precision required for accumulation inside this routine,
   // but float storage is ok outside these loops.
   doublevector tmpPosition[SATELLITE_COUNT];
   doublevector tmpVelocity[SATELLITE_COUNT];

   for (int i = 0; i < SATELLITE_COUNT; ++i) {
       tmpPosition[i].x = s[i].position.x;
       tmpPosition[i].y = s[i].position.y;
       tmpVelocity[i].x = s[i].velocity.x;
       tmpVelocity[i].y = s[i].velocity.y;
   }

   // Physics iteration loop
   for(int physicsUpdateIndex = 0;
       physicsUpdateIndex < PHYSICSUPDATESPERFRAME;
      ++physicsUpdateIndex){

       // Physics satellite loop
      for(int i = 0; i < SATELLITE_COUNT; ++i){

         // Distance to the blackhole
         // (bit ugly code because C-struct cannot have member functions)
         doublevector positionToBlackHole = {.x = tmpPosition[i].x -
            HORIZONTAL_CENTER, .y = tmpPosition[i].y - VERTICAL_CENTER};
         double distToBlackHoleSquared =
            positionToBlackHole.x * positionToBlackHole.x +
            positionToBlackHole.y * positionToBlackHole.y;
         double distToBlackHole = sqrt(distToBlackHoleSquared);

         // Gravity force
         doublevector normalizedDirection = {
            .x = positionToBlackHole.x / distToBlackHole,
            .y = positionToBlackHole.y / distToBlackHole};
         double accumulation = GRAVITY / distToBlackHoleSquared;

         // Delta time is used to make velocity same despite different FPS
         // Update velocity based on force
         tmpVelocity[i].x -= accumulation * normalizedDirection.x *
            DELTATIME / PHYSICSUPDATESPERFRAME;
         tmpVelocity[i].y -= accumulation * normalizedDirection.y *
            DELTATIME / PHYSICSUPDATESPERFRAME;

         // Update position based on velocity
         tmpPosition[i].x +=
            tmpVelocity[i].x * DELTATIME / PHYSICSUPDATESPERFRAME;
         tmpPosition[i].y +=
            tmpVelocity[i].y * DELTATIME / PHYSICSUPDATESPERFRAME;
      }
   }

   // double precision required for accumulation inside this routine,
   // but float storage is ok outside these loops.
   // copy back the float storage.
   for (int i = 0; i < SATELLITE_COUNT; ++i) {
       s[i].position.x = tmpPosition[i].x;
       s[i].position.y = tmpPosition[i].y;
       s[i].velocity.x = tmpVelocity[i].x;
       s[i].velocity.y = tmpVelocity[i].y;
   }
}

// Just some value that barely passes for OpenCL example program
#define ALLOWED_FP_ERROR 0.08
// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void errorCheck(){
   for(unsigned int i=0; i < SIZE; ++i) {
      if(fabs(correctPixels[i].red - pixels[i].red) > ALLOWED_FP_ERROR ||
         fabs(correctPixels[i].green - pixels[i].green) > ALLOWED_FP_ERROR ||
         fabs(correctPixels[i].blue - pixels[i].blue) > ALLOWED_FP_ERROR) {
         printf("Buggy pixel at (x=%i, y=%i). Press enter to continue.\n", i % WINDOW_WIDTH, i / WINDOW_WIDTH);
         getchar();
         return;
       }
   }
   printf("Error check passed!\n");
}

void my_compute(void);
// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void compute(void){
   int timeSinceStart = glutGet(GLUT_ELAPSED_TIME);
   previousFrameTimeSinceStart = timeSinceStart;

   // Error check during first frames
   if (frameNumber < 2) {
      memcpy(backupSatelites, satellites, sizeof(satellite) * SATELLITE_COUNT);
      sequentialPhysicsEngine(backupSatelites);
   }
   parallelPhysicsEngine();

   if (frameNumber < 2) {
      for (int i = 0; i < SATELLITE_COUNT; i++) {
         if (memcmp (&satellites[i], &backupSatelites[i], sizeof(satellite))) {
            printf("Incorrect satellite data of satellite: %d\n", i);
            getchar();
         }
      }
   }

   int satelliteMovementMoment = glutGet(GLUT_ELAPSED_TIME);
   int satelliteMovementTime = satelliteMovementMoment  - timeSinceStart;

   // Decides the colors for the pixels
   parallelGraphicsEngine();

   int pixelColoringMoment = glutGet(GLUT_ELAPSED_TIME);
   int pixelColoringTime =  pixelColoringMoment - satelliteMovementMoment;

   // Sequential code is used to check possible errors in the parallel version
   if(frameNumber < 2){
      sequentialGraphicsEngine();
      errorCheck();
   }

   int finishTime = glutGet(GLUT_ELAPSED_TIME);
   // Print timings
   int totalTime = finishTime - previousFinishTime;
   previousFinishTime = finishTime;

   printf("Total frametime: %ims, satellite moving: %ims, space coloring: %ims.\n",
      totalTime, satelliteMovementTime, pixelColoringTime);

   // Render the frame
   glutPostRedisplay();
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Probably not the best random number generator
float randomNumber(float min, float max){
   return (rand() * (max - min) / RAND_MAX) + min;
}

// DO NOT EDIT THIS FUNCTION
void fixedInit(unsigned int seed){

   if(seed != 0){
     srand(seed);
   }

   // Init pixel buffer which is rendered to the widow
   pixels = (color*)malloc(sizeof(color) * SIZE);

   // Init pixel buffer which is used for error checking
   correctPixels = (color*)malloc(sizeof(color) * SIZE);

   backupSatelites = (satellite*)malloc(sizeof(satellite) * SATELLITE_COUNT);


   // Init satellites buffer which are moving in the space
   satellites = (satellite*)malloc(sizeof(satellite) * SATELLITE_COUNT);

   // Create random satellites
   for(int i = 0; i < SATELLITE_COUNT; ++i){

      // Random reddish color
      color id = {.red = randomNumber(0.f, 0.15f) + 0.1f,
                  .green = randomNumber(0.f, 0.14f) + 0.0f,
                  .blue = randomNumber(0.f, 0.16f) + 0.0f};
    
      // Random position with margins to borders
      floatvector initialPosition = {.x = HORIZONTAL_CENTER - randomNumber(50, 320),
                              .y = VERTICAL_CENTER - randomNumber(50, 320) };
      initialPosition.x = (i / 2 % 2 == 0) ?
         initialPosition.x : WINDOW_WIDTH - initialPosition.x;
      initialPosition.y = (i < SATELLITE_COUNT / 2) ?
         initialPosition.y : WINDOW_HEIGHT - initialPosition.y;

      // Randomize velocity tangential to the balck hole
      floatvector positionToBlackHole = {.x = initialPosition.x - HORIZONTAL_CENTER,
                                    .y = initialPosition.y - VERTICAL_CENTER};
      float distance = (0.06 + randomNumber(-0.01f, 0.01f))/ 
        sqrt(positionToBlackHole.x * positionToBlackHole.x + 
          positionToBlackHole.y * positionToBlackHole.y);
      floatvector initialVelocity = {.x = distance * -positionToBlackHole.y,
                                .y = distance * positionToBlackHole.x};

      // Every other orbits clockwise
      if(i % 2 == 0){
         initialVelocity.x = -initialVelocity.x;
         initialVelocity.y = -initialVelocity.y;
      }

      satellite tmpSatelite = {.identifier = id, .position = initialPosition,
                              .velocity = initialVelocity};
      satellites[i] = tmpSatelite;
   }
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void fixedDestroy(void){
   destroy();

   free(pixels);
   free(correctPixels);
   free(satellites);

   if(seed != 0){
     printf("Used seed: %i\n", seed);
   }
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Renders pixels-buffer to the window 
void render(void){
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_FLOAT, pixels);
   glutSwapBuffers();
   frameNumber++;
}

// DO NOT EDIT THIS FUNCTION
// Inits glut and start mainloop
int main(int argc, char** argv){
   if(argc > 1){
     seed = atoi(argv[1]);
     printf("Using seed: %i\n", seed);
   }

   // Init glut window
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
   glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
   glutCreateWindow("Parallelization excercise");
   glutDisplayFunc(render);
   atexit(fixedDestroy);
   previousFrameTimeSinceStart = glutGet(GLUT_ELAPSED_TIME);
   previousFinishTime = glutGet(GLUT_ELAPSED_TIME);
   glEnable(GL_DEPTH_TEST);
   glClearColor(0.0, 0.0, 0.0, 1.0);
   fixedInit(seed);
   init();

   // compute-function is called when everythin from last frame is ready
   //glutIdleFunc(compute);
   glutIdleFunc(my_compute);
   // Start main loop
   glutMainLoop();  
}

// implement my functions
void my_compute(void){
   int timeSinceStart = glutGet(GLUT_ELAPSED_TIME);
   previousFrameTimeSinceStart = timeSinceStart;

   // Error check during first frames
   if (frameNumber < 2) {
      memcpy(backupSatelites, satellites, sizeof(satellite) * SATELLITE_COUNT);
      sequentialPhysicsEngine(backupSatelites);
   }
   // satellites move
   parallelPhysicsEngine();

   if (frameNumber < 2) {
      for (int i = 0; i < SATELLITE_COUNT; i++) {
         if (memcmp (&satellites[i], &backupSatelites[i], sizeof(satellite))) {
            printf("Incorrect satellite data of satellite: %d\n", i);
            getchar();
         }
      }
   }

   int satelliteMovementMoment = glutGet(GLUT_ELAPSED_TIME);
   int satelliteMovementTime = satelliteMovementMoment  - timeSinceStart;

   // Decides the colors for the pixels
   parallelGraphicsEngine();

   int pixelColoringMoment = glutGet(GLUT_ELAPSED_TIME);
   int pixelColoringTime =  pixelColoringMoment - satelliteMovementMoment;

   // Sequential code is used to check possible errors in the parallel version
   if(frameNumber < 2){
      sequentialGraphicsEngine();
      errorCheck();
   }

   int finishTime = glutGet(GLUT_ELAPSED_TIME);
   // Print timings
   int totalTime = finishTime - previousFinishTime;
   previousFinishTime = finishTime;

   printf("Total frametime: %ims, satellite moving: %ims, space coloring: %ims.\n",
      totalTime, satelliteMovementTime, pixelColoringTime);

   // Render the frame
   glutPostRedisplay();

   // record the time
   if (frameNumber < 2) {return;}
   totalTime_allFrames += totalTime;
   physicsTime_allFrames += satelliteMovementTime;
   graphicsTime_allFrames += pixelColoringTime;

   // if running more than RUNNING_FRAMES frames, then stop the program
   // and print the average running time of the physics, graphics and total frame times
   if (!((frameNumber-1) % RUNNING_FRAMES)){
      int calculatedFrame = frameNumber -1;
      printf("Total frames: %i, Average Total frametime: %ims, Average satellite moving: %ims, Average space coloring: %ims.\n",
         calculatedFrame, totalTime_allFrames/calculatedFrame, physicsTime_allFrames/calculatedFrame, graphicsTime_allFrames/calculatedFrame);
   } 
}