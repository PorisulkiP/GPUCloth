#pragma once
#include "cloth_settings.h"
#include "meshdata_types.cuh"
#include "SIM_mass_spring.h"

#include <vector>

#include <stdio.h>
#include <windows.h>  // В нем содержится описаниетипов APIENTRY, HINSTANCE и др.

#ifdef _WIN32
#  define DLLEXPORT __declspec(dllexport)
#else
#  define DLLEXPORT __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C"
{
#endif
// Вывод в консоль сообщения о том, что всё работает
DLLEXPORT void print_work_info();

// Загрузка данных
DLLEXPORT int SIM_LoadClothOBJs();
DLLEXPORT int SIM_SetCollisionOBJs();

// Запуск вычислений
DLLEXPORT bool BuildClothSprings(ClothModifierData* clmd, Mesh* mesh);
DLLEXPORT int SIM_solver(Depsgraph* depsgraph, Object* ob, ClothModifierData* clmd, Mesh* mesh, int frame, int countOfObj);

// TESTS
DLLEXPORT Mesh* Mesh_test(Mesh* mesh);
DLLEXPORT Depsgraph* Depsgraph_test(Depsgraph* depsgraph);
DLLEXPORT Object* Object_test(Object* obj);
DLLEXPORT ClothModifierData* ClothModifierData_test(ClothModifierData* cloth);

#pragma warning ( pop ) 
#ifdef __cplusplus
}
#endif