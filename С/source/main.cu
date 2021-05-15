#include <iostream>
#include <thread>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "main.cuh"
#include "cache.cuh"
#include "collision.cuh"
#include "cloth_settings.cuh"
#include "SIM_mass_spring.cuh"

/**
* Принцип работы симуляции:
* - Получаем данные из python файла
* - Выполнить проверку на кэш
*       - Если кэш есть идём по нему
* - Если кэша нет, запускаем запись
* - Запускаем просчёт
* - Возвращаем данные обратно в python
*/
int start_sim()
{
    system("chcp 1251"); system("cls"); //Подключаем русский язык и чистим консоль




    
    return 0;
}

// Вывод в консоль сообщения о том, что всё работает
void print_work_info()
{
    printf("\nI'm Work!!!\n\n");
}