#include "main.h"
#include "depsgraph.cuh"
#include "meshdata_types.cuh"
#include "object_types.cuh"

Mesh* Mesh_test(Mesh* mesh)
{
    //printf("\ntrisCount %d\n", mesh->totvert);
    //for (int i = 0; i < mesh->totvert; ++i)
    //{
    //    printf("mvert %d: ", i);
    //    print_MVert(mesh->mvert[i]);
    //}
    return mesh;
}

Depsgraph* Depsgraph_test(Depsgraph* depsgraph)
{
    print_Depsgraph(depsgraph);
    return depsgraph;
}

Object* Object_test(Object* obj)
{
    printf("name %s: ", obj->id.name);
    return obj;
}

ClothModifierData* ClothModifierData_test(ClothModifierData* cloth)
{
    printf("mvert_num: %d", cloth->clothObject->mvert_num);
    printf("numsprings: %d", cloth->clothObject->numsprings);
    printf("last_frame: %d", cloth->clothObject->last_frame);


    std::cout << "Epsilon: " << cloth->coll_parms->epsilon << std::endl;
    std::cout << "Friction: " << cloth->coll_parms->friction << std::endl;

    return cloth;
}