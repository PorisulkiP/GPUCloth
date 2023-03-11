#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "..\source\kernel\cloth.h"

#include "main.cuh"

/**
* Принцип работы симуляции:
* - Получаем данные из python файла
* - Запускаем просчёт
* - Возвращаем данные обратно в python
*/

// Сюда загружаются данные о ткани (Object, ClothModifierData)
int SIM_LoadClothOBJs()
{
    
    return 0;
}

// Сюда загружаются данные о объектах столкновений
// Depsgraph*, Object*, ClothModifierData*, float step, float dt
int SIM_SetCollisionOBJs() 
{
    //cloth_solve_collisions();
    return 0;
}

bool BuildClothSprings(ClothModifierData* clmd, Mesh* mesh)
{
    return cloth_build_springs(clmd, mesh);
}

// Запуск вычислений
// Depsgraph надо заменить на сцену
int SIM_solver(Depsgraph* depsgraph, Object* ob, ClothModifierData* clmd, Mesh* mesh, int frame)
{    
    return do_step_cloth(depsgraph, ob, clmd, mesh, frame);
}

// Вывод в консоль сообщения о том, что всё работает
void print_work_info()
{
    printf_s("\nDLL file is work!!!\n\n");

    system("chcp 1251"); //Подключаем русский язык

    cudaDeviceProp prop;
    int count;

    cudaGetDeviceCount(&count);    
    for (int i = 0; i < count; i++)
    {
        cudaGetDeviceProperties(&prop, i);
        printf_s("\n-----Общая информация об устройстве %d-----\n", i);
        printf_s("Кол-во устройств: %d\n", count);
        printf_s("Имя: %s\n", prop.name);
        printf_s("Вычислительные возможности: %d.%d\n", prop.major, prop.minor);
        printf_s("Тактовая частота: %d\n", prop.clockRate);
        printf_s("Перекрытие копирования:");
        if (prop.deviceOverlap) {
            printf_s("Перекрытие разрешено\n");
        }
        else {
            printf_s("Перекрытие запрещено\n");
        }
        printf_s("Тайм-аут выполнения ядра:");
        if (prop.kernelExecTimeoutEnabled) {
            printf_s("Перекрытие разрешено\n");
        }
        else {
            printf_s("Перекрытие запрещено\n");
        }

        printf_s("\n---Информация о памяти устройства %d---\n", i);
        printf_s("Всего глобальной памяти: %zd\n", prop.totalGlobalMem);
        printf_s("Всего константной памяти: %zd\n", prop.totalConstMem);
        printf_s("Максимальный шаг: %zd\n", prop.memPitch);
        printf_s("Выравнивание текстур: %zd\n", prop.textureAlignment);

        printf_s("\n---Информация о сультипроцессорах устройства %d---\n", i);
        printf_s("Количество мультипроцессоров: %d\n", prop.multiProcessorCount);
        printf_s("Разделяемая память на один МП: %zd\n", prop.sharedMemPerBlock);
        printf_s("Регистров на один МП: %d\n", prop.regsPerBlock);
        printf_s("Нитей в варпе: %d\n", prop.warpSize);
        printf_s("Макс. колличество нитей в блоке: %d\n", prop.maxThreadsPerBlock);
        printf_s("Макс.количество нитей по измерениям:(%d, %d, %d)\n", prop.maxThreadsDim[0],
            prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf_s("Макс. размер сетки:  (%d, %d, %d)\n", prop.maxGridSize[0],
            prop.maxGridSize[1], prop.maxGridSize[2]);
    }
}