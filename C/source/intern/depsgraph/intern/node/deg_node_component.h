#pragma once

#include "node/deg_node.h"
#include "node/deg_node_operation.h"
#include "utildefines.h"

struct ID;
struct bPoseChannel;

namespace blender {
namespace deg {

struct BoneComponentNode;
struct Depsgraph;
struct IDNode;
struct OperationNode;

/* ID Component - Base type for all components */
struct ComponentNode : public Node {
  /* Key used to look up operations within a component */
  struct OperationIDKey {
    OperationCode opcode;
    const char *name;
    int name_tag;

    OperationIDKey();
    OperationIDKey(OperationCode opcode);
    OperationIDKey(OperationCode opcode, const char *name, int name_tag);

    bool operator==(const OperationIDKey &other) const;
    uint64_t hash() const;
  };

  /* Typedef for container of operations */
  ComponentNode();
  ~ComponentNode();

  void init(const ID *id, const char *subdata) override;

  /* Find an existing operation, if requested operation does not exist
   * nullptr will be returned. */
  OperationNode *find_operation(OperationIDKey key) const;
  OperationNode *find_operation(OperationCode opcode, const char *name, int name_tag) const;

  /* Find an existing operation, will throw an assert() if it does not exist. */
  OperationNode *get_operation(OperationIDKey key) const;
  OperationNode *get_operation(OperationCode opcode, const char *name, int name_tag) const;

  /* Check operation exists and return it. */
  bool has_operation(OperationIDKey key) const;
  bool has_operation(OperationCode opcode, const char *name, int name_tag) const;


  /* Entry/exit operations management.
   *
   * Use those instead of direct set since this will perform sanity checks. */
  void set_entry_operation(OperationNode *op_node);
  void set_exit_operation(OperationNode *op_node);

  void clear_operations();

  virtual OperationNode *get_entry_operation() override;
  virtual OperationNode *get_exit_operation() override;

  void finalize_build(Depsgraph *graph);

  IDNode *owner;
  OperationNode *entry_operation;
  OperationNode *exit_operation;

  virtual bool depends_on_cow()
  {
    return true;
  }

  /* Denotes whether COW component is to be tagged when this component
   * is tagged for update. */
  virtual bool need_tag_cow_before_update()
  {
    return true;
  }

  /* Denotes whether this component affects (possibly indirectly) on a
   * directly visible object. */
  bool affects_directly_visible;
};

/* ---------------------------------------- */

#define DEG_COMPONENT_NODE_DEFINE_TYPEINFO(NodeType, type_, type_name_, id_recalc_tag) \
  const Node::TypeInfo NodeType::typeinfo = Node::TypeInfo(type_, type_name_, id_recalc_tag)

#define DEG_COMPONENT_NODE_DECLARE DEG_DEPSNODE_DECLARE

#define DEG_COMPONENT_NODE_DEFINE(name, NAME, id_recalc_tag) \
  DEG_COMPONENT_NODE_DEFINE_TYPEINFO( \
      name##ComponentNode, NodeType::NAME, #name " Component", id_recalc_tag); \
  static DepsNodeFactoryImpl<name##ComponentNode> DNTI_##NAME

#define DEG_COMPONENT_NODE_DECLARE_GENERIC(name) \
  struct name##ComponentNode : public ComponentNode { \
    DEG_COMPONENT_NODE_DECLARE; \
  }

#define DEG_COMPONENT_NODE_DECLARE_NO_COW_TAG_ON_UPDATE(name) \
  struct name##ComponentNode : public ComponentNode { \
    DEG_COMPONENT_NODE_DECLARE; \
    virtual bool need_tag_cow_before_update() \
    { \
      return false; \
    } \
  }
/* Bone Component */
struct BoneComponentNode : public ComponentNode {
  void init(const ID *id, const char *subdata);

  struct bPoseChannel *pchan; /* the bone that this component represents */

  DEG_COMPONENT_NODE_DECLARE;
};

void deg_register_component_depsnodes();

}  // namespace deg
}  // namespace blender
