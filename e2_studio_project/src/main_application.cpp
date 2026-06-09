/*
 * main_application.cpp
 *
 *  Created on: 28-Nov-2025
 *      Author: a5152874
 */

#include "main_application.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "weather_model.h"
#include <math.h>

//#include "log_disabled.h"
//#include "log_error.h"
//#include "log_warning.h"
//#include "log_info.h"
#include "log_debug.h"

float temp_data[] = {33.0f, 35.0f, 38.0f, 40.0f, 20.0f, 22.0f, 25.0f, 26.0f, 15.0f,\
                     17.0f, 18.0f, 19.0f, 27.0f, 28.0f, 29.0f, 31.0f, 24.0f, 23.0f,\
                     25.5f, 30.0f, 32.0f, 21.0f, 16.0f, 19.0f};
float humd_data[] = {31.0f, 40.0f, 43.0f, 35.0f, 88.0f, 85.0f, 90.0f, 82.0f, 94.0f,\
                     92.0f, 97.0f, 91.0f, 53.0f, 55.0f, 58.0f, 50.0f, 60.0f, 83.0f,\
                     70.0f,   62.0f, 48.0f, 80.0f, 60.0f, 97.0f};

const char * inference_data[] = {
                                 "Foggy",
                                 "Hot",
                                 "Rainy",
                                 "Sunny"
};

namespace
{
    tflite::ErrorReporter* error_reporter = nullptr;
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    int inference_count = 0;

    constexpr int kTensorArenaSize = 0x6400;
    uint8_t tensor_arena[kTensorArenaSize];
}

void tflit_setup(void)
{
    log_debug("TFLite Application Demo");

    tflite::InitializeTarget();

    // Set up logging. Google style is to avoid globals or statics because of
    // lifetime uncertainty, but since this has a trivial destructor it's okay.
    // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model = tflite::GetModel(g_weather_model_data);

    //TF_LITE_REPORT_ERROR(error_reporter, "Sample TFLIT Application New");

    // This pulls in all the operation implementations we need.
    // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::AllOpsResolver resolver;

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
      //TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
      return;
    }

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Keep track of how many inferences we have performed.
    inference_count = 0;
}

void tflite_run(void)
{
    float *scores;
    int i = 0;
    int j = 0;

    for (i = 0; i < sizeof(temp_data)/sizeof(temp_data[0]); i++)
    {
        float temp = temp_data[i];
        float hum  = humd_data[i];

        float norm_temp = (temp - 28.7254902f) / 7.33226973f;
        float norm_hum  = (hum  - 60.01960784f) / 23.07355567f;

        input->data.f[0] = norm_temp;
        input->data.f[1] = norm_hum;

        if (interpreter->Invoke() != kTfLiteOk)
        {
            return;
        }

        scores = output->data.f;

        // Choose max probability
        int predicted_index = 0;

        for (j = 1; j < 4; j++)
        {
            if (scores[j] > scores[predicted_index])
            {
                predicted_index = j;
            }
        }

        log_info("Prediction: %s", inference_data[predicted_index]);
    }

}

void main_application(void)
{
    tflit_setup();

    tflite_run();
}


