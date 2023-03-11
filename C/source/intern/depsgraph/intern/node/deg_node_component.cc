#include "DEG_depsgraph.h"

#include <cstdio>
#include <cstring> /* required for STREQ later on. */

#include "utildefines.h"

#include "object_types.cuh"

namespace blender::deg {

/* *********** */
/* Outer Nodes */

/* Standard Component Methods ============================= */
    OperationNode *ComponentNode::find_operation(OperationIDKey key) const
{
  OperationNode *node = nullptr;
  if (operations_map != nullptr) {
    node = operations_map->lookup_default(key, nullptr);
  }
  else {
    for (OperationNode *op_node : operations) {
      if (op_node->opcode == key.opcode && op_node->name_tag == key.name_tag &&
          STREQ(op_node->name.c_str(), key.name)) {
        node = op_node;
        break;
      }
    }
  }
  return node;
}

OperationNode *ComponentNode::find_operation(OperationCode opcode,
                                             const char *name,
                                             int name_tag) const
{
  OperationIDKey key(opcode, name, name_tag);
  return find_operation(key);
}

OperationNode *ComponentNode::get_operation(OperationIDKey key) const
{
  OperationNode *node = find_operation(key);
  if (node == nullptr) {
    fprintf(stderr,
            "%s: find_operation(%s) failed\n",
            this->identifier().c_str(),
            key.identifier().c_str());
    BLI_assert(!"Request for non-existing operation, should not happen");
    return nullptr;
  }
  return node;
}

OperationNode *ComponentNode::get_operation(OperationCode opcode,
                                            const char *name,
                                            int name_tag) const
{
  OperationIDKey key(opcode, name, name_tag);
  return get_operation(key);
}

bool ComponentNode::has_operation(OperationIDKey key) const
{
  return find_operation(key) != nullptr;
}

bool ComponentNode::has_operation(OperationCode opcode, const char *name, int name_tag) const
{
  OperationIDKey key(opcode, name, name_tag);
  return has_operation(key);
}

OperationNode *ComponentNode::add_operation(const DepsEvalOperationCb &op,
                                            OperationCode opcode,
                                            const char *name,
                                            int name_tag)
{
  OperationNode *op_node = find_operation(opcode, name, name_tag);
  if (!op_node) {
    DepsNodeFactory *factory = type_get_factory(NodeType::OPERATION);
    op_node = (OperationNode *)factory->create_node(this->owner->id_orig, "", name);

    /* register opnode in this component's operation set */
    OperationIDKey key(opcode, name, name_tag);
    operations_map->add(key, op_node);

    /* set backlink */
    op_node->owner = this;
  }
  else {
    fprintf(stderr,
            "add_operation: Operation already exists - %s has %s at %p\n",
            this->identifier().c_str(),
            op_node->identifier().c_str(),
            op_node);
    BLI_assert(!"Should not happen!");
  }

  /* attach extra data */
  op_node->evaluate = op;
  op_node->opcode = opcode;
  op_node->name = name;
  op_node->name_tag = name_tag;

  return op_node;
}

void ComponentNode::set_entry_operation(OperationNode *op_node)
{
  BLI_assert(entry_operation == nullptr);
  entry_operation = op_node;
}

void ComponentNode::set_exit_operation(OperationNode *op_node)
{
  BLI_assert(exit_operation == nullptr);
  exit_operation = op_node;
}

void ComponentNode::clear_operations()
{
  if (operations_map != nullptr) {
    for (OperationNode *op_node : operations_map->values()) {
      delete op_node;
    }
    operations_map->clear();
  }
  for (OperationNode *op_node : operations) {
    delete op_node;
  }
  operations.clear();
}

void ComponentNode::tag_update(Depsgraph *graph, eUpdateSource source)
{
  OperationNode *entry_op = get_entry_operation();
  if (entry_op != nullptr && entry_op->flag & DEPSOP_FLAG_NEEDS_UPDATE) {
    return;
  }
  for (OperationNode *op_node : operations) {
    op_node->tag_update(graph, source);
  }
  // It is possible that tag happens before finalization.
  if (operations_map != nullptr) {
    for (OperationNode *op_node : operations_map->values()) {
      op_node->tag_update(graph, source);
    }
  }
}

OperationNode *ComponentNode::get_entry_operation()
{
  if (entry_operation) {
    return entry_operation;
  }
  if (operations_map != nullptr && operations_map->size() == 1) {
    OperationNode *op_node = nullptr;
    /* TODO(sergey): This is somewhat slow. */
    for (OperationNode *tmp : operations_map->values()) {
      op_node = tmp;
    }
    /* Cache for the subsequent usage. */
    entry_operation = op_node;
    return op_node;
  }
  if (operations.size() == 1) {
    return operations[0];
  }
  return nullptr;
}

OperationNode *ComponentNode::get_exit_operation()
{
  if (exit_operation) {
    return exit_operation;
  }
  if (operations_map != nullptr && operations_map->size() == 1) {
    OperationNode *op_node = nullptr;
    /* TODO(sergey): This is somewhat slow. */
    for (OperationNode *tmp : operations_map->values()) {
      op_node = tmp;
    }
    /* Cache for the subsequent usage. */
    exit_operation = op_node;
    return op_node;
  }
  if (operations.size() == 1) {
    return operations[0];
  }
  return nullptr;
}

void ComponentNode::finalize_build(Depsgraph * /*graph*/)
{
  operations.reserve(operations_map->size());
  for (OperationNode *op_node : operations_map->values()) {
    operations.append(op_node);
  }
  delete operations_map;
  operations_map = nullptr;
}

/* Bone Component ========================================= */

/* Initialize 'bone component' node - from pointer data given */
void BoneComponentNode::init(const ID *id, const char *subdata)
{
  /* generic component-node... */
  ComponentNode::init(id, subdata);

  /* name of component comes is bone name */
  /* TODO(sergey): This sets name to an empty string because subdata is
   * empty. Is it a bug? */
  // this->name = subdata;

  /* bone-specific node data */
  Object *object = (Object *)id;
}

}  // namespace blender::deg
