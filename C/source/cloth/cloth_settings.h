#pragma once
#include "sys_types.cuh"

/**
 * The definition of a cloth vertex.
 * ����������� ����� �� �����.
 */
typedef struct ClothVertex {
	/* General flags per vertex.        */
	/* ����� ����� ��� ������ �������.  */
	int flags;
	/* The velocity of the point.       */
	/* �������� �����.       */
	float v[3];
	/* constrained position         */
	/* ����������� �������         */
	float xconst[3];
	/* The current position of this vertex. */
	/* ������� ��������� ���� �������. */
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
	// rest position of the vertex
	// �������� ��������� �������
	float xrest[3];
	/* delta velocities to be applied by collision response */
	float dcvel[3];
	/* same as above */
	uint impulse_count;
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
 * ����������� ������
 */
struct ClothSpring {
	int ij;              /* Pij from the paper, one end of the spring.   */
	int kl;              /* Pkl from the paper, one end of the spring.   */
	int mn;              /* For hair springs: third vertex index; For bending springs: edge index; */
	int* pa;             /* Array of vert indices for poly a (for bending springs). */
	int* pb;             /* Array of vert indices for poly b (for bending springs). */
	int la;              /* Length of *pa. */
	int lb;              /* Length of *pb. */
	float restlen;       /* The original length of the spring. */
	float restang;       /* The original angle of the bending springs. */
	int type;            /* Types defined in cloth.h ("springType"). */
	int flags;           /* Defined in cloth.h, e.g. deactivated due to tearing. */
	float lin_stiffness; /* Linear stiffness factor from the vertex groups. */
	float ang_stiffness; /* Angular stiffness factor from the vertex groups. */
	float editrestlen;

	/* angular bending spring target and derivatives */
	float target[3];
};

/**
 * ������� ��������� �����.
 */
typedef struct Cloth {
	// �������, ������� ������������ ��� �����.
	// The vertices that represent this cloth.
	struct ClothVertex* verts;
	// �������, ����������� �����.
	// The springs connecting the mesh.
	struct LinkNode* springs;
	// ���������� ������.
	// The count of springs.
	uint numsprings;
	// ���������� ������ == m * n
	// The number of verts = m*n
	uint mvert_num;
	// ���������� ������������� ��� ����� � ����� ��� �����.
	// Number of triangles for cloth and edges for hair.
	uint primitive_num;
	struct BVHTree* bvhtree;     /* collision tree for this cloth object */
	struct BVHTree* bvhselftree; /* collision tree for this cloth object */
	struct MVertTri* tri;
	struct Implicit_Data* implicit; /* our implicit solver connects to this pointer */
	struct EdgeSet* edgeset;        /* used for selfcollisions */
	// ��������� ���� ���������
	int last_frame;
	float initial_mesh_volume;      /* Initial volume of the mesh. Used for pressure */
	float average_acceleration[3];  /* Moving average of overall acceleration. */
	struct MEdge* edges;            /* Used for hair collisions. */
	struct EdgeSet* sew_edge_graph; /* Sewing edges represented using a GHash */
} Cloth;
