#pragma once

struct BoidSettings;
struct BoidState;
struct Object;
struct ParticleData;
struct ParticleSettings;
struct ParticleSimulationData;
struct RNG;

typedef struct BoidBrainData {
  struct ParticleSimulationData *sim;
  struct ParticleSettings *part;
  float timestep, cfra, dfra;
  float wanted_co[3], wanted_speed;

  /* Goal stuff */
  struct Object *goal_ob;
  float goal_co[3];
  float goal_nor[3];
  float goal_priority;

  struct RNG *rng;
} BoidBrainData;

void boids_precalc_rules(struct ParticleSettings *part, float cfra);
void boid_brain(BoidBrainData *bbd, int p, struct ParticleData *pa);
void boid_body(BoidBrainData *bbd, struct ParticleData *pa);
void boid_default_settings(struct BoidSettings *boids);
struct BoidRule *boid_new_rule(int type);
struct BoidState *boid_new_state(struct BoidSettings *boids);
struct BoidState *boid_duplicate_state(struct BoidSettings *boids, struct BoidState *state);
void boid_free_settings(BoidSettings *boids);
struct BoidSettings *boid_copy_settings(struct BoidSettings *boids);
struct BoidState *boid_get_current_state(struct BoidSettings *boids);
