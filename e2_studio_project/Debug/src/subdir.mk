################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/hal_entry.cpp \
../src/main_application.cpp 

CC_SRCS += \
../src/weather_model.cc 

C_SRCS += \
../src/hal_warmstart.c \
../src/syscalls.c 

SREC += \
tflite_weather_monitoring.srec 

C_DEPS += \
./src/hal_warmstart.d \
./src/syscalls.d 

CC_DEPS += \
./src/weather_model.d 

OBJS += \
./src/hal_entry.o \
./src/hal_warmstart.o \
./src/main_application.o \
./src/syscalls.o \
./src/weather_model.o 

MAP += \
tflite_weather_monitoring.map 

CPP_DEPS += \
./src/hal_entry.d \
./src/main_application.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	$(file > $@.in,-mcpu=cortex-m33 -mthumb -mfloat-abi=hard -mfpu=fpv5-sp-d16 -O2 -fmessage-length=0 -fsigned-char -ffunction-sections -fdata-sections -fno-strict-aliasing -Wunused -Wuninitialized -Wall -Wextra -Wmissing-declarations -Wconversion -Wpointer-arith -Wshadow -Wlogical-op -Waggregate-return -Wfloat-equal -g -D_RENESAS_RA_ -D_RA_CORE=CM33 -D_RA_ORDINAL=1 -I"C:/Project_data/works/personal/e2_studio_project/ra_gen" -I"C:/Project_data/works/personal/e2_studio_project/src/qc-middleware/common_utils" -I"C:/Project_data/works/personal/e2_studio_project/src/qc-middleware/SEGGER_RTT" -I"C:/Project_data/works/personal/e2_studio_project/src/third_party/gemmlowp" -I"C:/Project_data/works/personal/e2_studio_project/src/third_party/ruy" -I"C:/Project_data/works/personal/e2_studio_project/src/third_party/kissfft" -I"C:/Project_data/works/personal/e2_studio_project/src/third_party/flatbuffers/include" -I"." -I"C:/Project_data/works/personal/e2_studio_project/ra_cfg/fsp_cfg/bsp" -I"C:/Project_data/works/personal/e2_studio_project/ra_cfg/fsp_cfg" -I"C:/Project_data/works/personal/e2_studio_project/src" -I"C:/Project_data/works/personal/e2_studio_project/ra/fsp/inc" -I"C:/Project_data/works/personal/e2_studio_project/ra/fsp/inc/api" -I"C:/Project_data/works/personal/e2_studio_project/ra/fsp/inc/instances" -I"C:/Project_data/works/personal/e2_studio_project/ra/arm/CMSIS_6/CMSIS/Core/Include" -std=c++11 -fabi-version=0 -Wno-stringop-overflow -Wno-format-truncation --param=min-pagesize=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -c -o "$@" -x c++ "$<")
	@echo Building file: $< && arm-none-eabi-g++ @"$@.in"
src/%.o: ../src/%.c
	$(file > $@.in,-mcpu=cortex-m33 -mthumb -mfloat-abi=hard -mfpu=fpv5-sp-d16 -O2 -fmessage-length=0 -fsigned-char -ffunction-sections -fdata-sections -fno-strict-aliasing -Wunused -Wuninitialized -Wall -Wextra -Wmissing-declarations -Wconversion -Wpointer-arith -Wshadow -Wlogical-op -Waggregate-return -Wfloat-equal -g -D_RENESAS_RA_ -D_RA_CORE=CM33 -D_RA_ORDINAL=1 -I"C:/Project_data/works/personal/e2_studio_project/ra_gen" -I"C:/Project_data/works/personal/e2_studio_project/src/qc-middleware/common_utils" -I"C:/Project_data/works/personal/e2_studio_project/src/qc-middleware/SEGGER_RTT" -I"C:/Project_data/works/personal/e2_studio_project/src/third_party/gemmlowp" -I"C:/Project_data/works/personal/e2_studio_project/src/third_party/ruy" -I"C:/Project_data/works/personal/e2_studio_project/src/third_party/kissfft" -I"C:/Project_data/works/personal/e2_studio_project/src/third_party/flatbuffers/include" -I"." -I"C:/Project_data/works/personal/e2_studio_project/ra_cfg/fsp_cfg/bsp" -I"C:/Project_data/works/personal/e2_studio_project/ra_cfg/fsp_cfg" -I"C:/Project_data/works/personal/e2_studio_project/src" -I"C:/Project_data/works/personal/e2_studio_project/ra/fsp/inc" -I"C:/Project_data/works/personal/e2_studio_project/ra/fsp/inc/api" -I"C:/Project_data/works/personal/e2_studio_project/ra/fsp/inc/instances" -I"C:/Project_data/works/personal/e2_studio_project/ra/arm/CMSIS_6/CMSIS/Core/Include" -std=c99 -Wno-stringop-overflow -Wno-format-truncation --param=min-pagesize=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -c -o "$@" -x c "$<")
	@echo Building file: $< && arm-none-eabi-gcc @"$@.in"
src/%.o: ../src/%.cc
	$(file > $@.in,-mcpu=cortex-m33 -mthumb -mfloat-abi=hard -mfpu=fpv5-sp-d16 -O2 -fmessage-length=0 -fsigned-char -ffunction-sections -fdata-sections -fno-strict-aliasing -Wunused -Wuninitialized -Wall -Wextra -Wmissing-declarations -Wconversion -Wpointer-arith -Wshadow -Wlogical-op -Waggregate-return -Wfloat-equal -g -D_RENESAS_RA_ -D_RA_CORE=CM33 -D_RA_ORDINAL=1 -I"C:/Project_data/works/personal/e2_studio_project/ra_gen" -I"C:/Project_data/works/personal/e2_studio_project/src/qc-middleware/common_utils" -I"C:/Project_data/works/personal/e2_studio_project/src/qc-middleware/SEGGER_RTT" -I"C:/Project_data/works/personal/e2_studio_project/src/third_party/gemmlowp" -I"C:/Project_data/works/personal/e2_studio_project/src/third_party/ruy" -I"C:/Project_data/works/personal/e2_studio_project/src/third_party/kissfft" -I"C:/Project_data/works/personal/e2_studio_project/src/third_party/flatbuffers/include" -I"." -I"C:/Project_data/works/personal/e2_studio_project/ra_cfg/fsp_cfg/bsp" -I"C:/Project_data/works/personal/e2_studio_project/ra_cfg/fsp_cfg" -I"C:/Project_data/works/personal/e2_studio_project/src" -I"C:/Project_data/works/personal/e2_studio_project/ra/fsp/inc" -I"C:/Project_data/works/personal/e2_studio_project/ra/fsp/inc/api" -I"C:/Project_data/works/personal/e2_studio_project/ra/fsp/inc/instances" -I"C:/Project_data/works/personal/e2_studio_project/ra/arm/CMSIS_6/CMSIS/Core/Include" -std=c++11 -fabi-version=0 -Wno-stringop-overflow -Wno-format-truncation --param=min-pagesize=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -c -o "$@" -x c++ "$<")
	@echo Building file: $< && arm-none-eabi-g++ @"$@.in"

