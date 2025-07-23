# TinyML Application demo

# To generate a model:

Run the following python script, 

```python
scripts/model_gen.py

Edit the script file for adding custom data

This script will generate the model in the '*.tflite' format.

# Convert the generated model into c array

tflite format model should be converted into c array, so that it can be integrated into embedded application.

Run the following command for the same:

```bash
xxd -i mode_name.tflite > model_name.cc
