#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "header.cuh"

// Вывод в консоль сообщения о том, что всё работает
void print_work_info()
{
    printf("\nI'm Work!!!\n\n");
}

// Проверка работоспособности CUDA на устройстве
int is_cuda_available()
{
    // Я так и не нашёл как енто определить, сижу плачу(((((
    return 0;
}