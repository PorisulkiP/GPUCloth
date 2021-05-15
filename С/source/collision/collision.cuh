

typedef struct CollisionModifierData {
	ModifierData modifier;

	/** Position at the beginning of the frame. */
	struct MVert* x;
	/** Position at the end of the frame. */
	struct MVert* xnew;
	/** Unused atm, but was discussed during sprint. */
	struct MVert* xold;
	/** New position at the actual inter-frame step. */
	struct MVert* current_xnew;
	/** Position at the actual inter-frame step. */
	struct MVert* current_x;
	/** (xnew - x) at the actual inter-frame step. */
	struct MVert* current_v;

	struct MVertTri* tri;

	unsigned int mvert_num;
	unsigned int tri_num;
	/** Cfra time of modifier. */
	float time_x, time_xnew;
	/** Collider doesn't move this frame, i.e. x[].co==xnew[].co. */
	char is_static;
	char _pad[7];

	/** Bounding volume hierarchy for this cloth object. */
	struct BVHTree* bvhtree;
} CollisionModifierData;

typedef struct ColDetectData {
	//ClothModifierData* clmd;
	//CollisionModifierData* collmd;
	//BVHTreeOverlap* overlap;
	//CollPair* collisions;
	bool culling;
	bool use_normal;
	bool collided;
} ColDetectData;

typedef struct SelfColDetectData {
	//ClothModifierData* clmd;
	//BVHTreeOverlap* overlap;
	//CollPair* collisions;
	bool collided;
} SelfColDetectData;