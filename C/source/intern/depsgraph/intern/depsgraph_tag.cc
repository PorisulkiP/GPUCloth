//#include "depsgraph_tag.h"
//
//#include <cstdio>
//#include <cstring> /* required for memset */
//#include <queue>
//
//#include "math_bits.cuh"
//#include "utildefines.h"
//#include "object_types.cuh"
//
//#include "anim_types.h"
//#include "scene.h"
//
//#include "DEG_depsgraph.h"
//#include "DEG_depsgraph_debug.h"
//#include "DEG_depsgraph_query.h"
//
//#include "builder/deg_builder.h"
//#include "depsgraph.h"
//#include "depsgraph_registry.h"
//#include "depsgraph_update.h"
//#include "eval/deg_eval_copy_on_write.h"
//#include "eval/deg_eval_flush.h"
//#include "node/deg_node.h"
//#include "node/deg_node_component.h"
//#include "node/deg_node_factory.h"
//#include "node/deg_node_id.h"
//#include "node/deg_node_operation.h"
//#include "node/deg_node_time.h"
//
//namespace deg = blender::deg;
//
//namespace blender::deg {
//
//    namespace {
//
//        void depsgraph_geometry_tag_to_component(const ID* id, NodeType* component_type)
//        {
//            const NodeType result = geometry_tag_to_component(id);
//            if (result != NodeType::UNDEFINED) {
//                *component_type = result;
//            }
//        }
//        const char* DEG_update_tag_as_string(IDRecalcFlag flag)
//        {
//            switch (flag) {
//            case ID_RECALC_TRANSFORM:
//                return "TRANSFORM";
//            case ID_RECALC_GEOMETRY:
//                return "GEOMETRY";
//            case ID_RECALC_ANIMATION:
//                return "ANIMATION";
//            case ID_RECALC_PSYS_REDO:
//                return "PSYS_REDO";
//            case ID_RECALC_PSYS_RESET:
//                return "PSYS_RESET";
//            case ID_RECALC_PSYS_CHILD:
//                return "PSYS_CHILD";
//            case ID_RECALC_PSYS_PHYS:
//                return "PSYS_PHYS";
//            case ID_RECALC_PSYS_ALL:
//                return "PSYS_ALL";
//            case ID_RECALC_COPY_ON_WRITE:
//                return "COPY_ON_WRITE";
//            case ID_RECALC_SHADING:
//                return "SHADING";
//            case ID_RECALC_SELECT:
//                return "SELECT";
//            case ID_RECALC_BASE_FLAGS:
//                return "BASE_FLAGS";
//            case ID_RECALC_POINT_CACHE:
//                return "POINT_CACHE";
//            case ID_RECALC_EDITORS:
//                return "EDITORS";
//            case ID_RECALC_SEQUENCER_STRIPS:
//                return "SEQUENCER_STRIPS";
//            case ID_RECALC_AUDIO_FPS:
//                return "AUDIO_FPS";
//            case ID_RECALC_AUDIO_VOLUME:
//                return "AUDIO_VOLUME";
//            case ID_RECALC_AUDIO_MUTE:
//                return "AUDIO_MUTE";
//            case ID_RECALC_AUDIO_LISTENER:
//                return "AUDIO_LISTENER";
//            case ID_RECALC_AUDIO:
//                return "AUDIO";
//            case ID_RECALC_PARAMETERS:
//                return "PARAMETERS";
//            case ID_RECALC_SOURCE:
//                return "SOURCE";
//            case ID_RECALC_ALL:
//                return "ALL";
//            case ID_RECALC_TAG_FOR_UNDO:
//                return "TAG_FOR_UNDO";
//            }
//            return nullptr;
//        }
//
//
//        OperationCode psysTagToOperationCode(IDRecalcFlag tag)
//        {
//            if (tag == ID_RECALC_PSYS_RESET) {
//                return OperationCode::PARTICLE_SETTINGS_RESET;
//            }
//            return OperationCode::OPERATION;
//        }
//        void depsgraph_update_editors_tag(Main* bmain, Depsgraph* graph, ID* id)
//        {
//            /* NOTE: We handle this immediately, without delaying anything, to be
//             * sure we don't cause threading issues with OpenGL. */
//             /* TODO(sergey): Make sure this works for CoW-ed datablocks as well. */
//            DEGEditorUpdateContext update_ctx = { nullptr };
//            update_ctx.bmain = bmain;
//            update_ctx.depsgraph = (::Depsgraph*)graph;
//            update_ctx.scene = graph->scene;
//            update_ctx.view_layer = graph->view_layer;
//        }
//
//        void depsgraph_id_tag_copy_on_write(Depsgraph* graph, IDNode* id_node, eUpdateSource update_source)
//        {
//            ComponentNode* cow_comp = id_node->find_component(NodeType::COPY_ON_WRITE);
//            if (cow_comp == nullptr) {
//                BLI_assert(!deg_copy_on_write_is_needed(GS(id_node->id_orig->name)));
//                return;
//            }
//            cow_comp->tag_update(graph, update_source);
//        }
//
//        void depsgraph_tag_component(Depsgraph* graph,
//            IDNode* id_node,
//            NodeType component_type,
//            OperationCode operation_code,
//            eUpdateSource update_source)
//        {
//            ComponentNode* component_node = id_node->find_component(component_type);
//            /* NOTE: Animation component might not be existing yet (which happens when adding new driver or
//             * adding a new keyframe), so the required copy-on-write tag needs to be taken care explicitly
//             * here. */
//            if (component_node == nullptr) {
//                if (component_type == NodeType::ANIMATION) {
//                    depsgraph_id_tag_copy_on_write(graph, id_node, update_source);
//                }
//                return;
//            }
//            if (operation_code == OperationCode::OPERATION) {
//                component_node->tag_update(graph, update_source);
//            }
//            else {
//                OperationNode* operation_node = component_node->find_operation(operation_code);
//                if (operation_node != nullptr) {
//                    operation_node->tag_update(graph, update_source);
//                }
//            }
//            /* If component depends on copy-on-write, tag it as well. */
//            if (component_node->need_tag_cow_before_update()) {
//                depsgraph_id_tag_copy_on_write(graph, id_node, update_source);
//            }
//        }
//
//        /* This is a tag compatibility with legacy code.
//         *
//         * Mainly, old code was tagging object with ID_RECALC_GEOMETRY tag to inform
//         * that object's data datablock changed. Now API expects that ID is given
//         * explicitly, but not all areas are aware of this yet. */
//        void graph_id_tag_update_single_flag(Main* bmain,
//            Depsgraph* graph,
//            ID* id,
//            IDNode* id_node,
//            IDRecalcFlag tag,
//            eUpdateSource update_source)
//        {
//            if (tag == ID_RECALC_EDITORS) {
//                if (graph != nullptr && graph->is_active) {
//                    depsgraph_update_editors_tag(bmain, graph, id);
//                }
//                return;
//            }
//            /* Get description of what is to be tagged. */
//            NodeType component_type;
//            OperationCode operation_code;
//            /* Check whether we've got something to tag. */
//            if (component_type == NodeType::UNDEFINED) {
//                /* Given ID does not support tag. */
//                /* TODO(sergey): Shall we raise some panic here? */
//                return;
//            }
//            /* Some sanity checks before moving forward. */
//            if (id_node == nullptr) {
//                /* Happens when object is tagged for update and not yet in the
//                 * dependency graph (but will be after relations update). */
//                return;
//            }
//            /* Tag ID recalc flag. */
//            DepsNodeFactory* factory = type_get_factory(component_type);
//            BLI_assert(factory != nullptr);
//            id_node->id_cow->recalc |= factory->id_recalc_tag();
//            /* Tag corresponding dependency graph operation for update. */
//            if (component_type == NodeType::ID_REF) {
//                id_node->tag_update(graph, update_source);
//            }
//            else {
//                depsgraph_tag_component(graph, id_node, component_type, operation_code, update_source);
//            }
//        }
//
//        const char* update_source_as_string(eUpdateSource source)
//        {
//            switch (source) {
//            case DEG_UPDATE_SOURCE_TIME:
//                return "TIME";
//            case DEG_UPDATE_SOURCE_USER_EDIT:
//                return "USER_EDIT";
//            case DEG_UPDATE_SOURCE_RELATIONS:
//                return "RELATIONS";
//            case DEG_UPDATE_SOURCE_VISIBILITY:
//                return "VISIBILITY";
//            }
//            BLI_assert(!"Should never happen.");
//            return "UNKNOWN";
//        }
//
//        int deg_recalc_flags_for_legacy_zero()
//        {
//            return ID_RECALC_ALL &
//                ~(ID_RECALC_PSYS_ALL | ID_RECALC_ANIMATION | ID_RECALC_SOURCE | ID_RECALC_EDITORS);
//        }
//
//        int deg_recalc_flags_effective(Depsgraph* graph, int flags)
//        {
//            if (graph != nullptr) {
//                if (!graph->is_active) {
//                    return 0;
//                }
//            }
//            if (flags == 0) {
//                return deg_recalc_flags_for_legacy_zero();
//            }
//            return flags;
//        }
//
//        /* Special tag function which tags all components which needs to be tagged
//         * for update flag=0.
//         *
//         * TODO(sergey): This is something to be avoid in the future, make it more
//         * explicit and granular for users to tag what they really need. */
//        void deg_graph_node_tag_zero(Main* bmain,
//            Depsgraph* graph,
//            IDNode* id_node,
//            eUpdateSource update_source)
//        {
//            if (id_node == nullptr) {
//                return;
//            }
//            ID* id = id_node->id_orig;
//            /* TODO(sergey): Which recalc flags to set here? */
//            id_node->id_cow->recalc |= deg_recalc_flags_for_legacy_zero();
//
//            for (ComponentNode* comp_node : id_node->components.values()) {
//                if (comp_node->type == NodeType::ANIMATION) {
//                    continue;
//                }
//                comp_node->tag_update(graph, update_source);
//            }
//        }
//
//    } /* namespace */
//
//    void DEG_graph_id_tag_update(struct Main* bmain,
//        struct Depsgraph* depsgraph,
//        struct ID* id,
//        int flag)
//    {
//        deg::Depsgraph* graph = (deg::Depsgraph*)depsgraph;
//    }
//
//    void DEG_time_tag_update(struct Main* bmain)
//    {
//        for (deg::Depsgraph* depsgraph : deg::get_all_registered_graphs(bmain)) {
//            DEG_graph_time_tag_update(reinterpret_cast<::Depsgraph*>(depsgraph));
//        }
//    }
//
//    void DEG_graph_time_tag_update(struct Depsgraph* depsgraph)
//    {
//        deg::Depsgraph* deg_graph = reinterpret_cast<deg::Depsgraph*>(depsgraph);
//        deg_graph->tag_time_source();
//    }
//
//    void DEG_id_type_tag(Main* bmain, short id_type)
//    {
//        for (deg::Depsgraph* depsgraph : deg::get_all_registered_graphs(bmain)) {
//            DEG_graph_id_type_tag(reinterpret_cast<::Depsgraph*>(depsgraph), id_type);
//        }
//    }
//
//    /* Update dependency graph when visible scenes/layers changes. */
//    void DEG_graph_on_visible_update(Main* bmain, Depsgraph* depsgraph, const bool do_time)
//    {
//        deg::Depsgraph* graph = (deg::Depsgraph*)depsgraph;
//    }
//
//    void DEG_on_visible_update(Main* bmain, const bool do_time)
//    {}
//}