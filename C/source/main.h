#pragma once
#include "mesh_types.h"
#include "SIM_mass_spring.cuh"

extern "C"
{

// Вывод в консоль сообщения о том, что всё работает
__declspec(dllexport) void print_work_info();

// Загрузка данных
__declspec(dllexport) int SIM_LoadClothOBJs();
__declspec(dllexport) int SIM_SetCollisionOBJs();

// Запуск вычислений
__declspec(dllexport) bool BuildClothSprings(ClothModifierData* clmd, Mesh* mesh);
__declspec(dllexport) bool SIM_solver(const Depsgraph * depsgraph, const Object * ob, const ClothModifierData * clmd, const Mesh * mesh);


// TESTS
__declspec(dllexport) Mesh* Mesh_test(Mesh* mesh);
__declspec(dllexport) Depsgraph* Depsgraph_test(Depsgraph* depsgraph);
__declspec(dllexport) Object* Object_test(Object* obj);
__declspec(dllexport) ClothModifierData* ClothModifierData_test(ClothModifierData* cloth);

}