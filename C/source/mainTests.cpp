#include <iostream>
#include "main.cuh"
#include "depsgraph.h"

Mesh* Mesh_test(Mesh* mesh, uint trisCount)
{
    printf("\ntrisCount %d\n", trisCount);
    for (uint i = 0; i < trisCount; ++i)
    {
        printf("mvert %d: ", i);
        print_MVert(mesh->mvert[i]);
    }
    return mesh;
}

Depsgraph* Depsgraph_test(Depsgraph* depsgraph)
{
    print_Depsgraph(depsgraph);
    return depsgraph;
}

Object* Object_test(Object* obj)
{
    return obj;
}

ClothModifierData* ClothModifierData_test(ClothModifierData* cloth)
{
    printf("mvert_num: %d", cloth->clothObject->mvert_num);
    printf("numsprings: %d", cloth->clothObject->numsprings);
    printf("last_frame: %d", cloth->clothObject->last_frame);
    return cloth;
}