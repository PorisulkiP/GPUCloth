#include "node/deg_node_id.h"

#include <cstdio>
#include <cstring> /* required for STREQ later on. */
#include "utildefines.h"

#include "ID.h"
#include "DEG_depsgraph.h"

//const char *linkedStateAsString(eDepsNode_LinkedState_Type linked_state)
//{
//  switch (linked_state) {
//    case DEG_ID_LINKED_INDIRECTLY:
//      return "INDIRECTLY";
//    case DEG_ID_LINKED_VIA_SET:
//      return "VIA_SET";
//    case DEG_ID_LINKED_DIRECTLY:
//      return "DIRECTLY";
//  }
//  BLI_assert(!"Unhandled linked state, should never happen.");
//  return "UNKNOWN";
//}

//IDNode::ComponentIDKey::ComponentIDKey(NodeType type, const char *name) : type(type), name(name)
//{
//}
//
//bool IDNode::ComponentIDKey::operator==(const ComponentIDKey &other) const
//{
//  return type == other.type && STREQ(name, other.name);
//}
//
//uint64_t IDNode::ComponentIDKey::hash() const
//{
//  const int type_as_int = static_cast<int>(type);
//  return BLI_ghashutil_combine_hash(BLI_ghashutil_uinthash(type_as_int),
//                                    BLI_ghashutil_strhash_p(name));
//}

/* Initialize 'id' node - from pointer data given. */
void IDNode::init(const ID *id)
{
  BLI_assert(id != nullptr);
  id_orig = (ID *)id;
  id_orig_session_uuid = id->session_uuid;
  //eval_flags = 0;
  //previous_eval_flags = 0;
  //customdata_masks = DEGCustomDataMeshMasks();
  //previous_customdata_masks = DEGCustomDataMeshMasks();
  //linked_state = DEG_ID_LINKED_INDIRECTLY;
  ////is_directly_visible = true;
  //is_collection_fully_expanded = false;
  //has_base = false;
  //is_user_modified = false;

  //visible_components_mask = 0;
  //previously_visible_components_mask = 0;
}

void IDNode::destroy()
{
    if (id_orig == nullptr) 
        return;

    //for (ComponentNode *comp_node : components.values()) 
    //{
    //    delete comp_node;
    //}

    id_orig = nullptr;
}

//std::string IDNode::identifier() const
//{
//  char orig_ptr[24], cow_ptr[24];
//  BLI_snprintf(orig_ptr, sizeof(orig_ptr), "%p", id_orig);
//  BLI_snprintf(cow_ptr, sizeof(cow_ptr), "%p", id_cow);
//  return string(nodeTypeAsString(type)) + " : " + name + " (orig: " + orig_ptr +
//         ", eval: " + cow_ptr + ", is_directly_visible " +
//         (is_directly_visible ? "true" : "false") + ")";
//}

//ComponentNode *IDNode::find_component(NodeType type, const char *name) const
//{
//  ComponentIDKey key(type, name);
//  return components.lookup_default(key, nullptr);
//}

//ComponentNode *IDNode::add_component(NodeType type, const char *name)
//{
//  ComponentNode *comp_node = find_component(type, name);
//  if (!comp_node) {
//    DepsNodeFactory *factory = type_get_factory(type);
//    comp_node = (ComponentNode *)factory->create_node(this->id_orig, "", name);
//
//    /* Register. */
//    ComponentIDKey key(type, name);
//    components.add_new(key, comp_node);
//    comp_node->owner = this;
//  }
//  return comp_node;
//}

//void IDNode::tag_update(Depsgraph *graph, eUpdateSource source)
//{
//  for (ComponentNode *comp_node : components.values()) {
//    /* Relations update does explicit animation update when needed. Here we ignore animation
//     * component to avoid loss of possible unkeyed changes. */
//    if (comp_node->type == NodeType::ANIMATION && source == DEG_UPDATE_SOURCE_RELATIONS) {
//      continue;
//    }
//    comp_node->tag_update(graph, source);
//  }
//}
//
//void IDNode::finalize_build(Depsgraph *graph)
//{
//  /* Finalize build of all components. */
//  for (ComponentNode *comp_node : components.values()) {
//    comp_node->finalize_build(graph);
//  }
//  visible_components_mask = get_visible_components_mask();
//}

//IDComponentsMask IDNode::get_visible_components_mask() const
//{
//  IDComponentsMask result = 0;
//  for (ComponentNode *comp_node : components.values()) 
//  {
//    if (comp_node->affects_directly_visible) 
//    {
//      const int component_type_as_int = static_cast<int>(comp_node->type);
//      BLI_assert(component_type_as_int < 64);
//      result |= (1ULL << component_type_as_int);
//    }
//  }
//  return result;
//}