/*
 * main_application.h
 *
 *  Created on: 28-Nov-2025
 *      Author: a5152874
 */

#ifndef MAIN_APPLICATION_H_
#define MAIN_APPLICATION_H_

// Expose a C friendly interface for main functions.
#ifdef __cplusplus
extern "C" {
#endif

void main_application(void);

void tflit_setup(void);

void tflite_run(void);

#ifdef __cplusplus
}
#endif


#endif /* MAIN_APPLICATION_H_ */
