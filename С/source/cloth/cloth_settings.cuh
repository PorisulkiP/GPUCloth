#pragma once
#include "DNA_meshdata_types.cuh"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * √лавна€ структура ткани.
 */
typedef struct Cloth {
	/* The vertices that represent this cloth. */
	/* */
	struct ClothVertex* verts;     
	/* The springs connecting the mesh. */
	/* */
	struct LinkNode* springs;
	/* The count of springs. */
	/* */
	unsigned int numsprings;
	/* The number of verts == m * n. */
	/* */
	unsigned int mvert_num;
	/* Number of triangles for cloth and edges for hair. */
	/* */
	unsigned int primitive_num;
	/* unused, only 1 solver here */
	/* */
	unsigned char old_solver_type; 

	unsigned char pad2;
	short pad3;
	//struct BVHTree* bvhtree;     /* collision tree for this cloth object */
	//struct BVHTree* bvhselftree; /* collision tree for this cloth object */

	struct MVertTri* tri;
	///* our implicit solver connects to this pointer */
	///* наш не€вный решатель подключаетс€ к этому указателю */
	//struct Implicit_Data* implicit; 
	/* used for selfcollisions */
	/* */
	struct EdgeSet* edgeset;
	int last_frame;
	/* Initial volume of the mesh. Used for pressure */
	/* */
	float initial_mesh_volume;
	/* Moving average of overall acceleration. */
	/* */
	float average_acceleration[3];
	/* Sewing edges represented using a GHash */
	/* */
	struct EdgeSet* sew_edge_graph; 
} Cloth;

/**
 * The definition of a cloth vertex.
 * ќпределение точек на ткани.
 */
typedef struct ClothVertex {
	/* General flags per vertex.        */
	/* ќбщие флаги дл€ каждой вершины.  */
	int flags;                  
	/* The velocity of the point.       */
	/* —корость точек.       */
	float v[3];                 
	/* constrained position         */
	/*  онстантна€ позици€         */
	float xconst[3];            
	/* The current position of this vertex. */
	/* “екущее положение этой вершины. */
	float x[3];                 
	/* The previous position of this vertex.*/
	float xold[3];              
	/* temporary position */
	float tx[3];                
	/* temporary old position */
	float txold[3];             
	/* temporary "velocity", mostly used as tv = tx-txold */
	float tv[3];                
	/* mass / weight of the vertex      */
	float mass;                 
	/* goal, from SB            */
	float goal;                 
	/* used in collision.c */
	float impulse[3];           
	/* rest position of the vertex */
	float xrest[3];             
	/* delta velocities to be applied by collision response */
	float dcvel[3];             
	/* same as above */
	unsigned int impulse_count; 
	/* average length of connected springs */
	float avg_spring_len;       

	float struct_stiff;
	float bend_stiff;
	float shear_stiff;
	int spring_count;      /* how many springs attached? */
	float shrink_factor;   /* how much to shrink this cloth */
	float internal_stiff;  /* internal spring stiffness scaling */
	float pressure_factor; /* how much pressure should affect this vertex */
} ClothVertex;

/**
 * The definition of a spring.
 * ќпределение пружин
 */
typedef struct ClothSpring {
	int ij;              /* Pij from the paper, one end of the spring.   */
	int kl;              /* Pkl from the paper, one end of the spring.   */
	int mn;              /* For hair springs: third vertex index; For bending springs: edge index; */
	int* pa;             /* Array of vert indices for poly a (for bending springs). */
	int* pb;             /* Array of vert indices for poly b (for bending springs). */
	int la;              /* Length of *pa. */
	int lb;              /* Length of *pb. */
	float restlen;       /* The original length of the spring. */
	float restang;       /* The original angle of the bending springs. */
	int type;            /* Types defined in BKE_cloth.h ("springType"). */
	int flags;           /* Defined in BKE_cloth.h, e.g. deactivated due to tearing. */
	float lin_stiffness; /* Linear stiffness factor from the vertex groups. */
	float ang_stiffness; /* Angular stiffness factor from the vertex groups. */
	float editrestlen;

	/* angular bending spring target and derivatives */
	float target[3];
} ClothSpring;

typedef struct ClothModifierData {
	//ModifierData modifier;

	/** The internal data structure for cloth. */
	/* */
	struct Cloth* clothObject;
	/** Definition is in DNA_cloth_types.h. */
	/* */
	struct ClothSimSettings* sim_parms;
	/** Definition is in DNA_cloth_types.h. */
	/* */
	struct ClothCollSettings* coll_parms;

	struct ClothSolverResult* solver_result;
} ClothModifierData;

/* Spring types as defined in the paper.*/
/* “ипы пружин из статьи. */
typedef enum {
	CLOTH_SPRING_TYPE_STRUCTURAL = (1 << 1),
	CLOTH_SPRING_TYPE_SHEAR = (1 << 2),
	CLOTH_SPRING_TYPE_BENDING = (1 << 3),
	CLOTH_SPRING_TYPE_GOAL = (1 << 4),
	CLOTH_SPRING_TYPE_SEWING = (1 << 5),
	CLOTH_SPRING_TYPE_BENDING_HAIR = (1 << 6),
	CLOTH_SPRING_TYPE_INTERNAL = (1 << 7),
} CLOTH_SPRING_TYPES;

/* SPRING FLAGS */
/* ‘лаги пружин */
typedef enum {
	CLOTH_SPRING_FLAG_DEACTIVATE = (1 << 1),
	// springs has values to be applied
	// пружины имеют значени€, которые должны быть применены
	CLOTH_SPRING_FLAG_NEEDED = (1 << 2),
} CLOTH_SPRINGS_FLAGS;

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
	char _pad7[6];

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

	//struct EffectorWeights* effector_weights;

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
	char _pad1[2];
	float internal_tension;
	float internal_compression;
	float max_internal_tension;
	float max_internal_compression;
	char _pad0[4];

} ClothSimSettings;

#ifdef __cplusplus
}
#endif
