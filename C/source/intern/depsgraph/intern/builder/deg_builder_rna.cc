#include "deg_builder_rna.h"

#include <cstring>

#include "MEM_guardedalloc.cuh"

#include "listbase.cuh"
#include "utildefines.h"

#include "object_types.cuh"

#include "builder/deg_builder.h"
#include "depsgraph.h"
#include "node/deg_node.h"
#include "node/deg_node_component.h"
#include "node/deg_node_id.h"
#include "node/deg_node_operation.h"

namespace blender::deg {

/* ********************************* ID Data ******************************** */

class RNANodeQueryIDData {
 public:
  explicit RNANodeQueryIDData(const ID *id) : id_(id)
  {
  }

  void ensure_constraint_to_pchan_map()
  {
  }

 protected:
  /* ID this data corresponds to. */
  const ID *id_;
 };

/* ***************************** Node Identifier **************************** */

RNANodeIdentifier::RNANodeIdentifier()
    : id(nullptr),
      type(NodeType::UNDEFINED),
      component_name(""),
      operation_code(OperationCode::OPERATION),
      operation_name(),
      operation_name_tag(-1)
{
}

bool RNANodeIdentifier::is_valid() const
{
  return id != nullptr && type != NodeType::UNDEFINED;
}

/* ********************************** Query ********************************* */

RNANodeQuery::RNANodeQuery(Depsgraph *depsgraph, DepsgraphBuilder *builder)
    : depsgraph_(depsgraph), builder_(builder)
{
}

RNANodeQuery::~RNANodeQuery()
{
}

Node *RNANodeQuery::find_node(const PointerRNA *ptr,
                              const PropertyRNA *prop,
                              RNAPointerSource source)
{
  const RNANodeIdentifier node_identifier = construct_node_identifier(ptr, prop, source);
  if (!node_identifier.is_valid()) {
    return nullptr;
  }
  IDNode *id_node = depsgraph_->find_id_node(node_identifier.id);
  if (id_node == nullptr) {
    return nullptr;
  }
  ComponentNode *comp_node = id_node->find_component(node_identifier.type,
                                                     node_identifier.component_name);
  if (comp_node == nullptr) {
    return nullptr;
  }
  if (node_identifier.operation_code == OperationCode::OPERATION) {
    return comp_node;
  }
  return comp_node->find_operation(node_identifier.operation_code,
                                   node_identifier.operation_name,
                                   node_identifier.operation_name_tag);
}

bool RNANodeQuery::contains(const char *prop_identifier, const char *rna_path_component)
{
  const char *substr = strstr(prop_identifier, rna_path_component);
  if (substr == nullptr) {
    return false;
  }

  // If substr != prop_identifier, it means that the substring is found further in prop_identifier,
  // and that thus index -1 is a valid memory location.
  const bool start_ok = substr == prop_identifier || substr[-1] == '.';
  if (!start_ok) {
    return false;
  }

  const size_t component_len = strlen(rna_path_component);
  const bool end_ok = ELEM(substr[component_len], '\0', '.', '[');
  return end_ok;
}

RNANodeQueryIDData *RNANodeQuery::ensure_id_data(const ID *id)
{
  unique_ptr<RNANodeQueryIDData> &id_data = id_data_map_.lookup_or_add_cb(
      id, [&]() { return std::make_unique<RNANodeQueryIDData>(id); });
  return id_data.get();
}

}  // namespace blender::deg
