#ifndef __SIM_MASS_SPRING__
#define __SIM_MASS_SPRING__

#include <vector>
#include "cloth.h"

#define PHYS_GLOBAL_GRAVITY 1

struct ClothModifierData;
struct Depsgraph;
struct Implicit_Data;
struct ListBase;
struct Object;

static float I3[3][3] = { {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0} };

typedef enum eMassSpringSolverStatus {
    SIM_SOLVER_SUCCESS = (1 << 0),
    SIM_SOLVER_NUMERICAL_ISSUE = (1 << 1),
    SIM_SOLVER_NO_CONVERGENCE = (1 << 2),
    SIM_SOLVER_INVALID_INPUT = (1 << 3),
} eMassSpringSolverStatus;

struct Implicit_Data* SIM_mass_spring_solver_create(int numverts, int numsprings);
void SIM_mass_spring_solver_free(struct Implicit_Data* id);

int SIM_cloth_solver_init(struct ClothModifierData* clmd);
void SIM_cloth_solver_free(struct ClothModifierData* clmd);
bool SIM_cloth_solve(struct Depsgraph* depsgraph, struct Object* ob, float frame, struct ClothModifierData* clmd, struct ListBase* effectors);
void SIM_cloth_solver_set_positions(struct ClothModifierData* clmd);
void SIM_cloth_solver_set_volume(struct ClothModifierData* clmd);

#endif /* __SIM_MASS_SPRING__ */