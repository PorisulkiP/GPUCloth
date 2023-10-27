#pragma once

#include "BLI_iterator.h"

#include "depsgraph.cuh"

/* Needed for the instance iterator. */
#include "object_types.cuh"

struct BLI_Iterator;
struct CustomData_MeshMasks;
struct Depsgraph;
struct DupliObject;
struct ID;
struct ListBase;
struct PointerRNA;
struct Scene;
struct ViewLayer;

/* *********************** DEG input data ********************* */
//
///* Get scene that depsgraph was built for. */
//struct Scene *DEG_get_input_scene(const Depsgraph *graph);
//
///* Get view layer that depsgraph was built for. */
//struct ViewLayer *DEG_get_input_view_layer(const Depsgraph *graph);
//
///* Get bmain that depsgraph was built for. */
//struct Main *DEG_get_bmain(const Depsgraph *graph);
//
///* Get evaluation mode that depsgraph was built for. */
//eEvaluationMode DEG_get_mode(const Depsgraph *graph);
//
///* ********************* DEG evaluated data ******************* */
//
///* Check if given ID type was tagged for update. */
//bool DEG_id_type_updated(const struct Depsgraph *depsgraph, short id_type);
//bool DEG_id_type_any_updated(const struct Depsgraph *depsgraph);
//
///* Check if given ID type is present in the depsgraph */
//bool DEG_id_type_any_exists(const struct Depsgraph *depsgraph, short id_type);
//
///* Get additional evaluation flags for the given ID. */
//uint32_t DEG_get_eval_flags_for_id(const struct Depsgraph *graph, struct ID *id);
//
///* Get additional mesh CustomData_MeshMasks flags for the given object. */
//void DEG_get_customdata_mask_for_object(const struct Depsgraph *graph,
//                                        struct Object *object,
//                                        struct CustomData_MeshMasks *r_mask);

/* Get scene at its evaluated state.
 *
 * Technically, this is a copied-on-written and fully evaluated version of the input scene.
 * This function will check that the data-block has been expanded (and copied) from the original
 * one. Assert will happen if it's not. */
__host__ __device__ inline struct Scene* DEG_get_evaluated_scene(const Depsgraph* graph)
{
    const Depsgraph* deg_graph = reinterpret_cast<const Depsgraph*>(graph);
    Scene* scene_cow = deg_graph->scene_cow;
    /* TODO(sergey): Shall we expand data-block here? Or is it OK to assume
     * that caller is OK with just a pointer in case scene is not updated yet? */
    BLI_assert(scene_cow != nullptr && deg_copy_on_write_is_expanded(&scene_cow->id));
    return scene_cow;
}
//
///* Get view layer at its evaluated state.
// * This is a shortcut for accessing active view layer from evaluated scene. */
//struct ViewLayer *DEG_get_evaluated_view_layer(const struct Depsgraph *graph);
//
///* Get evaluated version of object for given original one. */
//struct Object *DEG_get_evaluated_object(const struct Depsgraph *depsgraph, struct Object *object);
//
///* Get evaluated version of given ID datablock. */
__host__ __device__ inline struct ID *DEG_get_evaluated_id(const struct Depsgraph *depsgraph, struct ID *id)
{
    if (id == nullptr) { return nullptr; }
    const IDNode* id_node = depsgraph->find_id_node(id);
    if (id_node == nullptr) { return id; }
    return id_node->id_cow;
}

///* Get evaluated version of data pointed to by RNA pointer */
//void DEG_get_evaluated_rna_pointer(const struct Depsgraph *depsgraph,
//                                   struct PointerRNA *ptr,
//                                   struct PointerRNA *r_ptr_eval);
//
///* Get original version of object for given evaluated one. */
//struct Object *DEG_get_original_object(struct Object *object);
//
///* Get original version of given evaluated ID datablock. */
//struct ID *DEG_get_original_id(struct ID *id);
//
///* Check whether given ID is an original,
// *
// * Original IDs are considered all the IDs which are not covered by copy-on-write system and are
// * not out-of-main localized data-blocks. */
//bool DEG_is_original_id(const struct ID *id);
//bool DEG_is_original_object(const struct Object *object);
//
///* Opposite of the above.
// *
// * If the data-block is not original it must be evaluated, and vice versa. */
//bool DEG_is_evaluated_id(const struct ID *id);
//bool DEG_is_evaluated_object(const struct Object *object);
//
///* Check whether depsgraph os fully evaluated. This includes the following checks:
// * - Relations are up-to-date.
// * - Nothing is tagged for update. */
//bool DEG_is_fully_evaluated(const struct Depsgraph *depsgraph);

/* ************************ DEG object iterators ********************* */

enum {
  DEG_ITER_OBJECT_FLAG_LINKED_DIRECTLY = (1 << 0),
  DEG_ITER_OBJECT_FLAG_LINKED_INDIRECTLY = (1 << 1),
  DEG_ITER_OBJECT_FLAG_LINKED_VIA_SET = (1 << 2),
  DEG_ITER_OBJECT_FLAG_VISIBLE = (1 << 3),
  DEG_ITER_OBJECT_FLAG_DUPLI = (1 << 4),
};

typedef struct DEGObjectIterData {
  struct Depsgraph *graph;
  int flag;

  struct Scene *scene;

  eEvaluationMode eval_mode;

  /* **** Iteration over geometry components **** */

  /* The object whose components we currently iterate over.
   * This might point to #temp_dupli_object. */
  struct Object *geometry_component_owner;
  /* Some identifier that is used to determine which geometry component should be returned next. */
  int geometry_component_id;
  /* Temporary storage for an object that is created from a component. */
  struct Object temp_geometry_component_object;

  /* **** Iteration over dupli-list. *** */

  /* Object which created the dupli-list. */
  struct Object *dupli_parent;
  /* List of duplicated objects. */
  struct ListBase *dupli_list;
  /* Next duplicated object to step into. */
  struct DupliObject *dupli_object_next;
  /* Corresponds to current object: current iterator object is evaluated from
   * this duplicated object. */
  struct DupliObject *dupli_object_current;
  /* Temporary storage to report fully populated DNA to the render engine or
   * other users of the iterator. */
  struct Object temp_dupli_object;

  /* **** Iteration over ID nodes **** */
  size_t id_node_index;
  size_t num_id_nodes;
} DEGObjectIterData;

//void DEG_iterator_objects_begin(struct BLI_Iterator *iter, DEGObjectIterData *data);
//void DEG_iterator_objects_next(struct BLI_Iterator *iter);
//void DEG_iterator_objects_end(struct BLI_Iterator *iter);

/**
 * Note: Be careful with DEG_ITER_OBJECT_FLAG_LINKED_INDIRECTLY objects.
 * Although they are available they have no overrides (collection_properties)
 * and will crash if you try to access it.
 */
#define DEG_OBJECT_ITER_BEGIN(graph_, instance_, flag_) { DEGObjectIterData data_ = { graph_, flag_, }; \
    ITER_BEGIN (DEG_iterator_objects_begin, DEG_iterator_objects_next, DEG_iterator_objects_end, &data_, Object *, instance_)
#define DEG_OBJECT_ITER_END ITER_END; } ((void)0)

/**
 * Depsgraph objects iterator for draw manager and final render
 */
#define DEG_OBJECT_ITER_FOR_RENDER_ENGINE_BEGIN(graph_, instance_) \
  DEG_OBJECT_ITER_BEGIN (graph_, instance_, DEG_ITER_OBJECT_FLAG_LINKED_DIRECTLY | \
                             DEG_ITER_OBJECT_FLAG_LINKED_VIA_SET | DEG_ITER_OBJECT_FLAG_VISIBLE | \
                             DEG_ITER_OBJECT_FLAG_DUPLI)

#define DEG_OBJECT_ITER_FOR_RENDER_ENGINE_END DEG_OBJECT_ITER_END

/* ************************ DEG ID iterators ********************* */

typedef struct DEGIDIterData {
  struct Depsgraph *graph;
  bool only_updated;

  size_t id_node_index;
  size_t num_id_nodes;
} DEGIDIterData;

//void DEG_iterator_ids_begin(struct BLI_Iterator *iter, DEGIDIterData *data);
//void DEG_iterator_ids_next(struct BLI_Iterator *iter);
//void DEG_iterator_ids_end(struct BLI_Iterator *iter);

