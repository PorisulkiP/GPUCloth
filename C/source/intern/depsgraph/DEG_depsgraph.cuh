#pragma once

#include "BLI_map.cuh"
#include "BLI_vector.hh"

#include "anim_types.h"
#include "modifier_types.cuh"
#include "customdata_types.cuh"
#include "ghash.cuh"

#include <string>
#include <functional>

struct ComponentNode;
struct Scene;
struct ViewLayer;
struct Depsgraph;
struct Relation;
struct OperationNode;

typedef enum eEvaluationMode {
    DAG_EVAL_VIEWPORT = 0, /* evaluate for OpenGL viewport */
    DAG_EVAL_RENDER = 1,   /* evaluate for render purposes */
} eEvaluationMode;

typedef enum ePhysicsRelationType {
    DEG_PHYSICS_EFFECTOR = 0,
    DEG_PHYSICS_COLLISION = 1,
    DEG_PHYSICS_SMOKE_COLLISION = 2,
    DEG_PHYSICS_DYNAMIC_BRUSH = 3,
    DEG_PHYSICS_RELATIONS_NUM = 4,
} ePhysicsRelationType;

enum eUpdateSource {
    /* Update is caused by a time change. */
    DEG_UPDATE_SOURCE_TIME = (1 << 0),
    /* Update caused by user directly or indirectly influencing the node. */
    DEG_UPDATE_SOURCE_USER_EDIT = (1 << 1),
    /* Update is happening as a special response for the relations update. */
    DEG_UPDATE_SOURCE_RELATIONS = (1 << 2),
    /* Update is happening due to visibility change. */
    DEG_UPDATE_SOURCE_VISIBILITY = (1 << 3),
};

enum eDepsNode_LinkedState_Type {
    /* Generic indirectly linked id node. */
    DEG_ID_LINKED_INDIRECTLY = 0,
    /* Id node present in the set (background) only. */
    DEG_ID_LINKED_VIA_SET = 1,
    /* Id node directly linked via the SceneLayer. */
    DEG_ID_LINKED_DIRECTLY = 2,
};

typedef enum eDepsSceneComponentType {
    /* Parameters Component - Default when nothing else fits
     * (i.e. just SDNA property setting). */
    DEG_SCENE_COMP_PARAMETERS,
    /* Animation Component
     * TODO(sergey): merge in with parameters? */
     DEG_SCENE_COMP_ANIMATION,
     /* Sequencer Component (Scene Only). */
     DEG_SCENE_COMP_SEQUENCER,
} eDepsSceneComponentType;

typedef enum eDepsObjectComponentType {
    /* Used in query API, to denote which component caller is interested in. */
    DEG_OB_COMP_ANY,

    /* Parameters Component - Default when nothing else fits
     * (i.e. just SDNA property setting). */
     DEG_OB_COMP_PARAMETERS,
     /* Animation Component.
      *
      * TODO(sergey): merge in with parameters? */
      DEG_OB_COMP_ANIMATION,
      /* Transform Component (Parenting/Constraints) */
      DEG_OB_COMP_TRANSFORM,
      /* Geometry Component (#Mesh / #DispList). */
      DEG_OB_COMP_GEOMETRY,

      /* Evaluation-Related Outer Types (with Sub-data) */

      /* Pose Component - Owner/Container of Bones Eval */
      DEG_OB_COMP_EVAL_POSE,
      /* Bone Component - Child/Sub-component of Pose */
      DEG_OB_COMP_BONE,

      /* Material Shading Component */
      DEG_OB_COMP_SHADING,
      /* Cache Component */
      DEG_OB_COMP_CACHE,
} eDepsObjectComponentType;

/* DagNode->eval_flags */
enum {
    /* Regardless to curve->path animation flag path is to be evaluated anyway,
     * to meet dependencies with such a things as curve modifier and other guys
     * who're using curve deform, where_on_path and so. */
    DAG_EVAL_NEED_CURVE_PATH = (1 << 0),
    /* A shrinkwrap modifier or constraint targeting this mesh needs information
     * about non-manifold boundary edges for the Target Normal Project mode. */
     DAG_EVAL_NEED_SHRINKWRAP_BOUNDARY = (1 << 1),
};


/* Evaluation Operation for atomic operation */
/* XXX: move this to another header that can be exposed? */
typedef std::function<void(struct ::Depsgraph*)> DepsEvalOperationCb;
typedef uint64_t IDComponentsMask;

/* Identifiers for common operations (as an enum). */
enum class OperationCode {
    /* Generic Operations. -------------------------------------------------- */

    /* Placeholder for operations which don't need special mention */
    OPERATION = 0,

    /* Generic parameters evaluation. */
    ID_PROPERTY,
    PARAMETERS_ENTRY,
    PARAMETERS_EVAL,
    PARAMETERS_EXIT,
    VISIBILITY,

    /* Animation, Drivers, etc. --------------------------------------------- */
    /* NLA + Action */
    ANIMATION_ENTRY,
    ANIMATION_EVAL,
    ANIMATION_EXIT,
    /* Driver */
    DRIVER,

    /* Scene related. ------------------------------------------------------- */
    SCENE_EVAL,
    AUDIO_ENTRY,
    AUDIO_VOLUME,

    /* Object related. ------------------------------------------------------ */
    OBJECT_FROM_LAYER_ENTRY,
    OBJECT_BASE_FLAGS,
    OBJECT_FROM_LAYER_EXIT,
    DIMENSIONS,

    /* Transform. ----------------------------------------------------------- */
    /* Transform entry point. */
    TRANSFORM_INIT,
    /* Local transforms only */
    TRANSFORM_LOCAL,
    /* Parenting */
    TRANSFORM_PARENT,
    /* Constraints */
    TRANSFORM_CONSTRAINTS,
    /* Handle object-level updates, mainly proxies hacks and recalc flags. */
    TRANSFORM_EVAL,
    /* Initializes transformation for simulation.
     * For example, ensures point cache is properly reset before doing rigid
     * body simulation. */
     TRANSFORM_SIMULATION_INIT,
     /* Transform exit point */
     TRANSFORM_FINAL,

     /* Rigid body. ---------------------------------------------------------- */
     /* Perform Simulation */
     RIGIDBODY_REBUILD,
     RIGIDBODY_SIM,
     /* Copy results to object */
     RIGIDBODY_TRANSFORM_COPY,

     /* Geometry. ------------------------------------------------------------ */

     /* Initialize evaluation of the geometry. Is an entry operation of geometry
      * component. */
      GEOMETRY_EVAL_INIT,
      /* Evaluate the whole geometry, including modifiers. */
      GEOMETRY_EVAL,
      /* Evaluation of geometry is completely done. */
      GEOMETRY_EVAL_DONE,
      /* Evaluation of a shape key.
       * NOTE: Currently only for object data data-blocks. */
       GEOMETRY_SHAPEKEY,

       /* Object data. --------------------------------------------------------- */
       LIGHT_PROBE_EVAL,
       SPEAKER_EVAL,
       SOUND_EVAL,
       ARMATURE_EVAL,

       /* Pose. ---------------------------------------------------------------- */
       /* Init pose, clear flags, etc. */
       POSE_INIT,
       /* Initialize IK solver related pose stuff. */
       POSE_INIT_IK,
       /* Pose is evaluated, and runtime data can be freed. */
       POSE_CLEANUP,
       /* Pose has been fully evaluated and ready to be used by others. */
       POSE_DONE,
       /* IK/Spline Solvers */
       POSE_IK_SOLVER,
       POSE_SPLINE_IK_SOLVER,

       /* Bone. ---------------------------------------------------------------- */
       /* Bone local transforms - entry point */
       BONE_LOCAL,
       /* Pose-space conversion (includes parent + restpose, */
       BONE_POSE_PARENT,
       /* Constraints */
       BONE_CONSTRAINTS,
       /* Bone transforms are ready
        *
        * - "READY"  This (internal, noop is used to signal that all pre-IK
        *            operations are done. Its role is to help mediate situations
        *            where cyclic relations may otherwise form (i.e. one bone in
        *            chain targeting another in same chain,
        *
        * - "DONE"   This noop is used to signal that the bone's final pose
        *            transform can be read by others. */
        /* TODO: deform mats could get calculated in the final_transform ops... */
        BONE_READY,
        BONE_DONE,
        /* B-Bone segment shape computation (after DONE) */
        BONE_SEGMENTS,

        /* Particle System. ----------------------------------------------------- */
        PARTICLE_SYSTEM_INIT,
        PARTICLE_SYSTEM_EVAL,
        PARTICLE_SYSTEM_DONE,

        /* Particle Settings. --------------------------------------------------- */
        PARTICLE_SETTINGS_INIT,
        PARTICLE_SETTINGS_EVAL,
        PARTICLE_SETTINGS_RESET,

        /* Point Cache. --------------------------------------------------------- */
        POINT_CACHE_RESET,

        /* File cache. ---------------------------------------------------------- */
        FILE_CACHE_UPDATE,

        /* Collections. --------------------------------------------------------- */
        VIEW_LAYER_EVAL,

        /* Copy on Write. ------------------------------------------------------- */
        COPY_ON_WRITE,

        /* Shading. ------------------------------------------------------------- */
        SHADING,
        MATERIAL_UPDATE,
        LIGHT_UPDATE,
        WORLD_UPDATE,

        /* Node Tree. ----------------------------------------------------------- */
        NTREE_OUTPUT,

        /* Batch caches. -------------------------------------------------------- */
        GEOMETRY_SELECT_UPDATE,

        /* Masks. --------------------------------------------------------------- */
        MASK_ANIMATION,
        MASK_EVAL,

        /* Movie clips. --------------------------------------------------------- */
        MOVIECLIP_EVAL,
        MOVIECLIP_SELECT_UPDATE,

        /* Images. -------------------------------------------------------------- */
        IMAGE_ANIMATION,

        /* Synchronization. ----------------------------------------------------- */
        SYNCHRONIZE_TO_ORIGINAL,

        /* Generic data-block --------------------------------------------------- */
        GENERIC_DATABLOCK_UPDATE,

        /* Sequencer. ----------------------------------------------------------- */

        SEQUENCES_EVAL,

        /* Duplication/instancing system. --------------------------------------- */
        DUPLI,

        /* Simulation. ---------------------------------------------------------- */
        SIMULATION_EVAL
};
const char* operationCodeAsString(OperationCode opcode);

/* Flags for Depsgraph Nodes.
 * NOTE: IS a bit shifts to allow usage as an accumulated. bitmask.
 */
enum OperationFlag {
    /* Node needs to be updated. */
    DEPSOP_FLAG_NEEDS_UPDATE = (1 << 0),

    /* Node was directly modified, causing need for update. */
    DEPSOP_FLAG_DIRECTLY_MODIFIED = (1 << 1),

    /* Node was updated due to user input. */
    DEPSOP_FLAG_USER_MODIFIED = (1 << 2),

    /* Node may not be removed, even when it has no evaluation callback and no outgoing relations.
     * This is for NO-OP nodes that are purely used to indicate a relation between components/IDs,
     * and not for connecting to an operation. */
     DEPSOP_FLAG_PINNED = (1 << 3),

     /* The operation directly or indirectly affects ID node visibility. */
     DEPSOP_FLAG_AFFECTS_VISIBILITY = (1 << 4),

     /* Set of flags which gets flushed along the relations. */
     DEPSOP_FLAG_FLUSH = (DEPSOP_FLAG_USER_MODIFIED),
};



/* Settings/Tags on Relationship.
 * NOTE: Is a bitmask, allowing accumulation. */
enum RelationFlag {
    /* "cyclic" link - when detecting cycles, this relationship was the one
     * which triggers a cyclic relationship to exist in the graph. */
    RELATION_FLAG_CYCLIC = (1 << 0),
    /* Update flush will not go through this relation. */
    RELATION_FLAG_NO_FLUSH = (1 << 1),
    /* Only flush along the relation is update comes from a node which was
     * affected by user input. */
     RELATION_FLAG_FLUSH_USER_EDIT_ONLY = (1 << 2),
     /* The relation can not be killed by the cyclic dependencies solver. */
     RELATION_FLAG_GODMODE = (1 << 4),
     /* Relation will check existence before being added. */
     RELATION_CHECK_BEFORE_ADD = (1 << 5),
};


#ifdef __cplusplus
extern "C" {
#endif

    /* ************************************************ */
    /* Depsgraph API */

    /* -------------------------------------------------------------------- */
    /** \name CRUD
     * \{ */

     /* Get main depsgraph instance from context! */

     /**
      * Create new Depsgraph instance.
      *
      * TODO: what arguments are needed here? What's the building-graph entry point?
      */
    //Depsgraph* DEG_graph_new(struct Main* bmain,
    //    struct Scene* scene,
    //    struct ViewLayer* view_layer,
    //    eEvaluationMode mode);

    ///**
    // * Replace the "owner" pointers (currently Main/Scene/ViewLayer) of this depsgraph.
    // * Used for:
    // * - Undo steps when we do want to re-use the old depsgraph data as much as possible.
    // * - Rendering where we want to re-use objects between different view layers.
    // */
    //void DEG_graph_replace_owners(struct Depsgraph* depsgraph,
    //    struct Main* bmain,
    //    struct Scene* scene,
    //    struct ViewLayer* view_layer);

    ///** Free graph's contents and graph itself. */
    //void DEG_graph_free(Depsgraph* graph);

    ///** \} */

    ///* -------------------------------------------------------------------- */
    ///** \name Node Types Registry
    // * \{ */

    // /** Register all node types. */
    //void DEG_register_node_types(void);

    ///** Free node type registry on exit. */
    //void DEG_free_node_types(void);

    ///** \} */

    ///* -------------------------------------------------------------------- */
    ///** \name Update Tagging
    // * \{ */

    // /** Tag dependency graph for updates when visible scenes/layers changes. */
    //void DEG_graph_tag_on_visible_update(Depsgraph* depsgraph, bool do_time);

    ///** Tag all dependency graphs for update when visible scenes/layers changes. */
    //void DEG_tag_on_visible_update(struct Main* bmain, bool do_time);

    ///**
    // * \note Will return NULL if the flag is not known, allowing to gracefully handle situations
    // * when recalc flag has been removed.
    // */
    //const char* DEG_update_tag_as_string(IDRecalcFlag flag);

    /** Tag given ID for an update in all the dependency graphs. */
    //void DEG_id_tag_update(struct ID* id, int flag);
    //void DEG_id_tag_update_ex(struct Main* bmain, struct ID* id, int flag);

    //void DEG_graph_id_tag_update(struct Main* bmain,
    //    struct Depsgraph* depsgraph,
    //    struct ID* id,
    //    int flag);

    ///** Tag all dependency graphs when time has changed. */
    //void DEG_time_tag_update(struct Main* bmain);

    ///** Tag a dependency graph when time has changed. */
    //void DEG_graph_time_tag_update(struct Depsgraph* depsgraph);

    ///**
    // * Mark a particular data-block type as having changing.
    // * This does not cause any updates but is used by external
    // * render engines to detect if for example a data-block was removed.
    // */
    //void DEG_graph_id_type_tag(struct Depsgraph* depsgraph, short id_type);
    //void DEG_id_type_tag(struct Main* bmain, short id_type);

    ///**
    // * Set a depsgraph to flush updates to editors. This would be done
    // * for viewport depsgraphs, but not render or export depsgraph for example.
    // */
    //void DEG_enable_editors_update(struct Depsgraph* depsgraph);

    ///** Check if something was changed in the database and inform editors about this. */
    //void DEG_editors_update(struct Depsgraph* depsgraph, bool time);

    ///** Clear recalc flags after editors or renderers have handled updates. */
    //void DEG_ids_clear_recalc(Depsgraph* depsgraph, bool backup);

    ///**
    // * Restore recalc flags, backed up by a previous call to #DEG_ids_clear_recalc.
    // * This also clears the backup.
    // */
    //void DEG_ids_restore_recalc(Depsgraph* depsgraph);

    /** \} */

    /* ************************************************ */
    /* Evaluation Engine API */

    /* -------------------------------------------------------------------- */
    /** \name Graph Evaluation
     * \{ */

     /**
      * Frame changed recalculation entry point.
      *
      * \note The frame-change happened for root scene that graph belongs to.
      */
    //void DEG_evaluate_on_framechange(Depsgraph* graph, float frame);

    /**
     * Data changed recalculation entry point.
     * Evaluate all nodes tagged for updating.
     */
    //void DEG_evaluate_on_refresh(Depsgraph* graph);

    /** \} */

    /* -------------------------------------------------------------------- */
    /** \name Editors Integration
     *
     * Mechanism to allow editors to be informed of depsgraph updates,
     * to do their own updates based on changes.
     * \{ */

    typedef struct DEGEditorUpdateContext {
        struct Main* bmain;
        struct Depsgraph* depsgraph;
        struct Scene* scene;
        struct ViewLayer* view_layer;
    } DEGEditorUpdateContext;

    typedef void (*DEG_EditorUpdateIDCb)(const DEGEditorUpdateContext* update_ctx, struct ID* id);
    typedef void (*DEG_EditorUpdateSceneCb)(const DEGEditorUpdateContext* update_ctx, bool updated);

    /** Set callbacks which are being called when depsgraph changes. */
    //void DEG_editors_set_update_cb(DEG_EditorUpdateIDCb id_func, DEG_EditorUpdateSceneCb scene_func);

    /** \} */

    /* -------------------------------------------------------------------- */
    /** \name Evaluation
     * \{ */

    //bool DEG_is_evaluating(const struct Depsgraph* depsgraph);

    //bool DEG_is_active(const struct Depsgraph* depsgraph);
    //void DEG_make_active(struct Depsgraph* depsgraph);
    //void DEG_make_inactive(struct Depsgraph* depsgraph);

    /** \} */

    /* -------------------------------------------------------------------- */
    /** \name Evaluation Debug
     * \{ */

    //void DEG_debug_print_begin(struct Depsgraph* depsgraph);

    //void DEG_debug_print_eval(struct Depsgraph* depsgraph,
    //    const char* function_name,
    //    const char* object_name,
    //    const void* object_address);

    //void DEG_debug_print_eval_subdata(struct Depsgraph* depsgraph,
    //    const char* function_name,
    //    const char* object_name,
    //    const void* object_address,
    //    const char* subdata_comment,
    //    const char* subdata_name,
    //    const void* subdata_address);

    //void DEG_debug_print_eval_subdata_index(struct Depsgraph* depsgraph,
    //    const char* function_name,
    //    const char* object_name,
    //    const void* object_address,
    //    const char* subdata_comment,
    //    const char* subdata_name,
    //    const void* subdata_address,
    //    int subdata_index);

    //void DEG_debug_print_eval_parent_typed(struct Depsgraph* depsgraph,
    //    const char* function_name,
    //    const char* object_name,
    //    const void* object_address,
    //    const char* parent_comment,
    //    const char* parent_name,
    //    const void* parent_address);

    //void DEG_debug_print_eval_time(struct Depsgraph* depsgraph,
    //    const char* function_name,
    //    const char* object_name,
    //    const void* object_address,
    //    float time);

    /** \} */


/* Metatype of Nodes - The general "level" in the graph structure
 * the node serves. */
    enum class NodeClass {
        /* Types generally unassociated with user-visible entities,
         * but needed for graph functioning. */
        GENERIC = 0,
        /* [Outer Node] An "aspect" of evaluating/updating an ID-Block, requiring
         * certain types of evaluation behavior. */
         COMPONENT = 1,
         /* [Inner Node] A glorified function-pointer/callback for scheduling up
          * evaluation operations for components, subject to relationship
          * requirements. */
          OPERATION = 2,
    };
    const char* nodeClassAsString(NodeClass node_class);

    /* Types of Nodes */
    enum class NodeType {
        /* Fallback type for invalid return value */
        UNDEFINED = 0,
        /* Inner Node (Operation) */
        OPERATION,

        /* **** Generic Types **** */

        /* Time-Source */
        TIMESOURCE,
        /* ID-Block reference - used as landmarks/collection point for components,
         * but not usually part of main graph. */
         ID_REF,

         /* **** Outer Types **** */

         /* Parameters Component - Default when nothing else fits
          * (i.e. just SDNA property setting). */
          PARAMETERS,
          /* Animation Component */
          ANIMATION,
          /* Transform Component (Parenting/Constraints) */
          TRANSFORM,
          /* Geometry Component (#Mesh / #DispList) */
          GEOMETRY,
          /* Sequencer Component (Scene Only) */
          SEQUENCER,
          /* Component which contains all operations needed for layer collections
           * evaluation. */
           LAYER_COLLECTIONS,
           /* Entry component of majority of ID nodes: prepares CoW pointers for
            * execution. */
            COPY_ON_WRITE,
            /* Used by all operations which are updating object when something is
             * changed in view layer. */
             OBJECT_FROM_LAYER,
             /* Audio-related evaluation. */
             AUDIO,
             ARMATURE,
             /* Un-interesting data-block, which is a part of dependency graph, but does
              * not have very distinctive update procedure. */
              GENERIC_DATABLOCK,

              /* Component which is used to define visibility relation between IDs, on the ID level.
               *
               * Consider two ID nodes NodeA and NodeB, with the relation between visibility components going
               * as NodeA -> NodeB. If NodeB is considered visible on screen, then the relation will ensure
               * that NodeA is also visible. The way how relation is oriented could be seen as a inverted from
               * visibility dependency point of view, but it follows the same direction as data dependency
               * which simplifies common algorithms which are dealing with relations and visibility.
               *
               * The fact that the visibility operates on the ID level basically means that all components in
               * the NodeA will be considered as affecting directly visible when NodeB's visibility is
               * affecting directly visible ID.
               *
               * This is the way to ensure objects needed for visualization without any actual data dependency
               * properly evaluated. Example of this is custom shapes for bones. */
               VISIBILITY,

               /* **** Evaluation-Related Outer Types (with Subdata) **** */

               /* Pose Component - Owner/Container of Bones Eval */
               EVAL_POSE,
               /* Bone Component - Child/Subcomponent of Pose */
               BONE,
               /* Particle Systems Component */
               PARTICLE_SYSTEM,
               PARTICLE_SETTINGS,
               /* Material Shading Component */
               SHADING,
               /* Point cache Component */
               POINT_CACHE,
               /* Image Animation Component */
               IMAGE_ANIMATION,
               /* Cache Component */
               /* TODO(sergey); Verify that we really need this. */
               CACHE,
               /* Batch Cache Component.
                * TODO(dfelinto/sergey): rename to make it more generic. */
                BATCH_CACHE,
                /* Duplication system. Used to force duplicated objects visible when
                 * when duplicator is visible. */
                 DUPLI,
                 /* Synchronization back to original datablock. */
                 SYNCHRONIZATION,
                 /* Simulation component. */
                 SIMULATION,
                 /* Node tree output component. */
                 NTREE_OUTPUT,

                 /* Total number of meaningful node types. */
                 NUM_TYPES
    };
    const char* nodeTypeAsString(NodeType type);

    NodeType nodeTypeFromSceneComponent(eDepsSceneComponentType component_type);
    eDepsSceneComponentType nodeTypeToSceneComponent(NodeType type);

    NodeType nodeTypeFromObjectComponent(eDepsObjectComponentType component_type);
    eDepsObjectComponentType nodeTypeToObjectComponent(NodeType type);

    /* All nodes in Depsgraph are descended from this. */
    struct Node {
        /* Helper class for static typeinfo in subclasses. */
        struct TypeInfo {
            TypeInfo(NodeType type, const char* type_name, int id_recalc_tag = 0);
            NodeType type;
            const char* type_name;
            int id_recalc_tag;
        };
        struct Stats {
            Stats();
            /* Reset all the counters. Including all stats needed for average
             * evaluation time calculation. */
            void reset();
            /* Reset counters needed for the current graph evaluation, does not
             * touch averaging accumulators. */
            void reset_current();
            /* Time spend on this node during current graph evaluation. */
            double current_time;
        };
        /* Relationships between nodes
         * The reason why all depsgraph nodes are descended from this type (apart
         * from basic serialization benefits - from the typeinfo) is that we can
         * have relationships between these nodes. */
        typedef blender::Vector<Relation*> Relations;

        std::string name;        /* Identifier - mainly for debugging purposes. */
        NodeType type;      /* Structural type of node. */
        Relations inlinks;  /* Nodes which this one depends on. */
        Relations outlinks; /* Nodes which depend on this one. */
        Stats stats;        /* Evaluation statistics. */

        /* Generic tags for traversal algorithms and such.
         *
         * Actual meaning of values depends on a specific area. Every area is to
         * clean this before use. */
        int custom_flags;

        /* Methods. */
        Node();
        virtual ~Node();

        /** Generic identifier for Depsgraph Nodes. */
        virtual std::string identifier() const;

        void init(const ID* /*id*/, const char* /*subdata*/);

        //void tag_update(Depsgraph* /*graph*/, eUpdateSource /*source*/);

        OperationNode* get_entry_operation(){ return (OperationNode*)nullptr; }
        OperationNode* get_exit_operation() { return (OperationNode*)nullptr; }

        NodeClass get_class() const;
    };

    struct OperationNode : public Node {
        OperationNode();

        //std::string identifier() const;
        //std::string full_identifier() const;

        //void tag_update(Depsgraph* graph, eUpdateSource source);

        bool is_noop() const
        {
            return (bool)evaluate == false;
        }

        OperationNode* get_entry_operation()
        {
            return this;
        }
        OperationNode* get_exit_operation()
        {
            return this;
        }

        /* Set this operation as component's entry/exit operation. */
        //void set_as_entry();
        //void set_as_exit();

        /* Component that contains the operation. */
        ComponentNode* owner;

        /* Callback for operation. */
        DepsEvalOperationCb evaluate;

        /* How many inlinks are we still waiting on before we can be evaluated. */
        uint32_t num_links_pending;
        bool scheduled;

        /* Identifier for the operation being performed. */
        OperationCode opcode;
        int name_tag;

        /* (OperationFlag) extra settings affecting evaluation. */
        int flag;
    };

    struct DEGCustomDataMeshMasks {
        uint64_t vert_mask;
        uint64_t edge_mask;
        uint64_t face_mask;
        uint64_t loop_mask;
        uint64_t poly_mask;

        DEGCustomDataMeshMasks() : vert_mask(0), edge_mask(0), face_mask(0), loop_mask(0), poly_mask(0)
        {
        }

        explicit DEGCustomDataMeshMasks(const CustomData_MeshMasks* other);

        DEGCustomDataMeshMasks& operator|=(const DEGCustomDataMeshMasks& other)
        {
            this->vert_mask |= other.vert_mask;
            this->edge_mask |= other.edge_mask;
            this->face_mask |= other.face_mask;
            this->loop_mask |= other.loop_mask;
            this->poly_mask |= other.poly_mask;
            return *this;
        }

        DEGCustomDataMeshMasks operator|(const DEGCustomDataMeshMasks& other) const
        {
            DEGCustomDataMeshMasks result;
            result.vert_mask = this->vert_mask | other.vert_mask;
            result.edge_mask = this->edge_mask | other.edge_mask;
            result.face_mask = this->face_mask | other.face_mask;
            result.loop_mask = this->loop_mask | other.loop_mask;
            result.poly_mask = this->poly_mask | other.poly_mask;
            return result;
        }

        bool operator==(const DEGCustomDataMeshMasks& other) const
        {
            return (this->vert_mask == other.vert_mask && this->edge_mask == other.edge_mask &&
                this->face_mask == other.face_mask && this->loop_mask == other.loop_mask &&
                this->poly_mask == other.poly_mask);
        }

        bool operator!=(const DEGCustomDataMeshMasks& other) const
        {
            return !(*this == other);
        }

        static DEGCustomDataMeshMasks MaskVert(const uint64_t vert_mask)
        {
            DEGCustomDataMeshMasks result;
            result.vert_mask = vert_mask;
            return result;
        }

        static DEGCustomDataMeshMasks MaskEdge(const uint64_t edge_mask)
        {
            DEGCustomDataMeshMasks result;
            result.edge_mask = edge_mask;
            return result;
        }

        static DEGCustomDataMeshMasks MaskFace(const uint64_t face_mask)
        {
            DEGCustomDataMeshMasks result;
            result.face_mask = face_mask;
            return result;
        }

        static DEGCustomDataMeshMasks MaskLoop(const uint64_t loop_mask)
        {
            DEGCustomDataMeshMasks result;
            result.loop_mask = loop_mask;
            return result;
        }

        static DEGCustomDataMeshMasks MaskPoly(const uint64_t poly_mask)
        {
            DEGCustomDataMeshMasks result;
            result.poly_mask = poly_mask;
            return result;
        }
    };

    /* B depends on A (A -> B) */
    struct Relation {
        Relation(Node* from, Node* to, const char* description)
            : from(from), to(to), name(description), flag(0)
        {
            from->outlinks.append(this);
            to->inlinks.append(this);
        }
        ~Relation() {}

        void unlink()
        {
            from->outlinks.remove_first_occurrence_and_reorder(this);
            to->inlinks.remove_first_occurrence_and_reorder(this);
        }

        /* the nodes in the relationship (since this is shared between the nodes) */
        Node* from; /* A */
        Node* to;   /* B */

        /* relationship attributes */
        const char* name; /* label for debugging */
        int flag;         /* Bitmask of RelationFlag) */
    };

    struct IDNode : public Node 
    {
        //struct ComponentIDKey 
        //{
        //    ComponentIDKey(NodeType type, const char* name = "");
        //    uint64_t hash() const;
        //    bool operator==(const ComponentIDKey& other) const;

        //    NodeType type;
        //    const char* name;
        //};

        /** Initialize 'id' node - from pointer data given. */
        void init(const ID* id);
        // void init_copy_on_write(ID* id_cow_hint = nullptr);
        ~IDNode() { destroy(); }
        void destroy();

        //virtual std::string identifier() const override;

        //ComponentNode* find_component(NodeType type, const char* name = "") const;
        //ComponentNode* add_component(NodeType type, const char* name = "");

        //void tag_update(Depsgraph* graph, eUpdateSource source);

        //void finalize_build(Depsgraph* graph);

        //IDComponentsMask get_visible_components_mask() const;

        /* Type of the ID stored separately, so it's possible to perform check whether CoW is needed
         * without de-referencing the id_cow (which is not safe when ID is NOT covered by CoW and has
         * been deleted from the main database.) */
        //ID_Type id_type;

        /* ID Block referenced. */
        ID* id_orig;

        /* Session-wide UUID of the id_orig.
         * Is used on relations update to map evaluated state from old nodes to the new ones, without
         * relying on pointers (which are not guaranteed to be unique) and without dereferencing id_orig
         * which could be "stale" pointer. */
        uint id_orig_session_uuid;

        /* Evaluated data-block.
         * Will be covered by the copy-on-write system if the ID Type needs it. */
        ID* id_cow;

        /* Hash to make it faster to look up components. */
        //blender::Map<ComponentIDKey, ComponentNode*> components;

        /* Additional flags needed for scene evaluation.
         * TODO(sergey): Only needed for until really granular updates
         * of all the entities. */
        //uint32_t eval_flags;
        //uint32_t previous_eval_flags;

        /* Extra customdata mask which needs to be evaluated for the mesh object. */
        //DEGCustomDataMeshMasks customdata_masks;
        //DEGCustomDataMeshMasks previous_customdata_masks;

        eDepsNode_LinkedState_Type linked_state;

        /* Indicates the data-block is to be considered visible in the evaluated scene.
         *
         * This flag is set during dependency graph build where check for an actual visibility might not
         * be available yet due to driven or animated restriction flags. So it is more of an intent or,
         * in other words, plausibility of the data-block to be visible. */
        //bool is_visible_on_build;

        /* Evaluated state of whether evaluation considered this data-block "enabled".
         *
         * For objects this is derived from the base restriction flags, which might be animated or
         * driven. It is set to `BASE_ENABLED_<VIEWPORT, RENDER>` (depending on the graph mode) after
         * the object's flags from layer were evaluated.
         *
         * For other data-types is currently always true. */
        //bool is_enabled_on_eval;

        /* For the collection type of ID, denotes whether collection was fully
         * recursed into. */
        //bool is_collection_fully_expanded;

        /* Is used to figure out whether object came to the dependency graph via a base. */
        //bool has_base;

        /* Accumulated flag from operation. Is initialized and used during updates flush. */
        //bool is_user_modified;

        /* Copy-on-Write component has been explicitly tagged for update. */
        //bool is_cow_explicitly_tagged;

        /* Accumulate recalc flags from multiple update passes. */
        //int id_cow_recalc_backup;

        //IDComponentsMask visible_components_mask;
        //IDComponentsMask previously_visible_components_mask;
    };

#ifdef __cplusplus
} /* extern "C" */
#endif
