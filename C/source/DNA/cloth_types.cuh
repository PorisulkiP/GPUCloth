#pragma once

#include "defs.cuh"
#include "modifier_types.cuh"

/*
 * Ёта структура содержит все глобальные данные, необходимые дл€ запуска моделировани€.
 * Ќа момент написани€ статьи эта структура содержит данные,
 * подход€щие дл€ выполнени€ моделировани€, как описано в разделе
 * ќграничени€ деформации в модели массы-пружины дл€ описани€ поведени€ жесткой ткани  савье ѕрово.
 * я попыталс€ сохранить похожие, если не точные имена переменных, как они представлены в статье.
 * “ам, где € немного изменил концепцию, как в stepsPerFrame по сравнению с временным шагом в статье,
 * € использовал переменные с разными именами, чтобы свести к минимуму путаницу.
*/
typedef struct ClothSimSettings {
	/** UNUSED atm. */
	/* */
	struct LinkNode* cache;
	/** See SB. */
	/* */
	float mingoal;
	/** Mechanical damping of springs. */
	/* */
	float Cdis;
	/** Viscous/fluid damping. */
	/* */
	float Cvi;
	/** Gravity/external force vector. */
	/* */
	float gravity[3];
	/** This is the duration of our time step, computed..   */
	/* */
	float dt;
	/** The mass of the entire cloth. */
	/* */
	float mass;
	/** Structural spring stiffness. */
	/* */
	float structural;
	/** Shear spring stiffness. */
	/* */
	float shear;
	/** Flexion spring stiffness. */
	/* */
	float bending;
	/** Max bending scaling value, min is "bending". */
	/* */
	float max_bend;
	/** Max structural scaling value, min is "structural". */
	/* */
	float max_struct;
	/** Max shear scaling value. */
	/* */
	float max_shear;
	/** Max sewing force. */
	/* */
	float max_sewing;
	/** Used for normalized springs. */
	/* */
	float avg_spring_len;
	/** Parameter how fast cloth runs. */
	/* */
	float timescale;
	/** Multiplies cloth speed. */
	/* */
	float time_scale;
	/** See SB. */
	/* */
	float maxgoal;
	/** Scaling of effector forces (see softbody_calc_forces)..*/
	/* */
	float eff_force_scale;
	/** Scaling of effector wind (see softbody_calc_forces)..   */
	/* */
	float eff_wind_scale;
	float sim_time_old;
	float defgoal;
	float goalspring;
	float goalfrict;
	/** Smoothing of velocities for hair. */
	/* */
	float velocity_smooth;
	/** Minimum density for hair. */
	/* */
	float density_target;
	/** Influence of hair density. */
	/* */
	float density_strength;
	/** Friction with colliders. */
	/* */
	float collider_friction;
	/** Damp the velocity to speed up getting to the resting position. */
	/* */
	float vel_damping;
	/** Min amount to shrink cloth by 0.0f (no shrink), 1.0f (shrink to nothing), -1.0f (double the
	 * edge length). */
	 /* */
	float shrink_min;
	/** Max amount to shrink cloth by 0.0f (no shrink), 1.0f (shrink to nothing), -1.0f (double the
	 * edge length). */
	 /* */
	float shrink_max;

	/* Air pressure */
	/* The uniform pressure that is constanty applied to the mesh. Can be negative */
	/* */
	float uniform_pressure_force;
	/* User set volume. This is the volume the mesh wants to expand to (the equilibrium volume). */
	/* */
	float target_volume;
	/* The scaling factor to apply to the actual pressure.
	 * pressure=( (current_volume/target_volume) - 1 + uniform_pressure_force) *
	 * pressure_factor */
	 /* */
	float pressure_factor;
	/* Density of the fluid inside or outside the object for use in the hydrostatic pressure
	 * gradient. */
	 /* */
	float fluid_density;
	short vgroup_pressure;
	//char _pad7[6];

	/* XXX various hair stuff
	 * should really be separate, this struct is a horrible mess already
	 */
	 /** Damping of bending springs. */
	float bending_damping;
	/** Size of voxel grid cells for continuum dynamics. */
	float voxel_cell_size;

	/** Number of time steps per frame. */
	int stepsPerFrame;
	/** Flags, see CSIMSETT_FLAGS enum above. */
	int flags;
	/** How many frames of simulation to do before we start. */
	int preroll;
	/** In percent!; if tearing enabled, a spring will get cut. */
	int maxspringlen;
	/** Which solver should be used? txold. */
	short solver_type;
	/** Vertex group for scaling bending stiffness. */
	short vgroup_bend;
	/** Optional vertexgroup name for assigning weight..*/
	short vgroup_mass;
	/** Vertex group for scaling structural stiffness. */
	short vgroup_struct;
	/** Vertex group for shrinking cloth. */
	short vgroup_shrink;
	/** Vertex group for scaling structural stiffness. */
	short shapekey_rest;
	/** Used for presets on GUI. */
	short presets;
	short reset;

	struct EffectorWeights* effector_weights;

	short bending_model;
	/** Vertex group for scaling structural stiffness. */
	short vgroup_shear;
	float tension;
	float compression;
	float max_tension;
	float max_compression;
	/** Mechanical damping of tension springs. */
	float tension_damp;
	/** Mechanical damping of compression springs. */
	float compression_damp;
	/** Mechanical damping of shear springs. */
	float shear_damp;

	/** The maximum lenght an internal spring can have during creation. */
	float internal_spring_max_length;
	/** How much the interal spring can diverge from the vertex normal during creation. */
	float internal_spring_max_diversion;
	/** Vertex group for scaling structural stiffness. */
	short vgroup_intern;
	//char _pad1[2];
	float internal_tension;
	float internal_compression;
	float max_internal_tension;
	float max_internal_compression;
	//char _pad0[4];

} ClothSimSettings;

/* SIMULATION FLAGS: goal flags,.. */
/* These are the bits used in SimSettings.flags. */
typedef enum {
  /** Object is only collision object, no cloth simulation is done. */
  CLOTH_SIMSETTINGS_FLAG_COLLOBJ = (1 << 2),
  /** DEPRECATED, for versioning only. */
  CLOTH_SIMSETTINGS_FLAG_GOAL = (1 << 3),
  /** True if tearing is enabled. */
  CLOTH_SIMSETTINGS_FLAG_TEARING = (1 << 4),
  /** True if pressure sim is enabled. */
  CLOTH_SIMSETTINGS_FLAG_PRESSURE = (1 << 5),
  /** Use the user defined target volume. */
  CLOTH_SIMSETTINGS_FLAG_PRESSURE_VOL = (1 << 6),
  /** True if internal spring generation is enabled. */
  CLOTH_SIMSETTINGS_FLAG_INTERNAL_SPRINGS = (1 << 7),
  /** DEPRECATED, for versioning only. */
  CLOTH_SIMSETTINGS_FLAG_SCALING = (1 << 8),
  /** Require internal springs to be created between points with opposite normals. */
  CLOTH_SIMSETTINGS_FLAG_INTERNAL_SPRINGS_NORMAL = (1 << 9),
  // ¬озможность редактировани€ кеша после симул€ции
  CLOTH_SIMSETTINGS_FLAG_CCACHE_EDIT = (1 << 12),
  /** Don't allow spring compression. */
  CLOTH_SIMSETTINGS_FLAG_RESIST_SPRING_COMPRESS = (1 << 13),
  /** Pull ends of loose edges together. */
  CLOTH_SIMSETTINGS_FLAG_SEW = (1 << 14),
  // Make simulation respect deformations in the base object.
  // —делайте симул€цию с учетом деформаций в базовом объекте.
  CLOTH_SIMSETTINGS_FLAG_DYNAMIC_BASEMESH = (1 << 15),
} CLOTH_SIMSETTINGS_FLAGS;

/* ClothSimSettings.bending_model. */
// “ип изменени€ ткани
typedef enum {
  CLOTH_BENDING_LINEAR = 0,
  CLOTH_BENDING_ANGULAR = 1,
} CLOTH_BENDING_MODEL;

typedef struct ClothCollSettings {
  /** E.g. pointer to temp memory for collisions. */
  struct LinkNode *collision_list;
  /** Min distance for collisions. */
  float epsilon;
  /** Fiction/damping with self contact. */
  float self_friction;
  /** Friction/damping applied on contact with other object. */
  float friction;
  /** Collision restitution on contact with other object. */
  float damping;
  /** For selfcollision. */
  float selfepsilon;
  float repel_force;
  float distance_repel;
  /** Collision flags defined in BKE_cloth.h. */
  int flags;
  /** How many iterations for the selfcollision loop. */
  short self_loop_count;
  /** How many iterations for the collision loop. */
  short loop_count;
  //char _pad[4];
  /** Only use colliders from this group of objects. */
  struct Collection *group;
  /** Vgroup to paint which vertices are not used for self collisions. */
  short vgroup_selfcol;
  /** Vgroup to paint which vertices are not used for object collisions. */
  short vgroup_objcol;
  //char _pad2[4];
  /** Impulse clamp for object collisions. */
  float clamp;
  /** Impulse clamp for self collisions. */
  float self_clamp;
} ClothCollSettings;

/* COLLISION FLAGS */
typedef enum {
  CLOTH_COLLSETTINGS_FLAG_ENABLED = (1 << 1), /* enables cloth - object collisions */
  CLOTH_COLLSETTINGS_FLAG_SELF = (1 << 2),    /* enables selfcollisions */
} CLOTH_COLLISIONSETTINGS_FLAGS;