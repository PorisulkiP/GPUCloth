#include "builder/deg_builder_relations.h"

namespace blender::deg {

////////////////////////////////////////////////////////////////////////////////
// Time source.

TimeSourceKey::TimeSourceKey() : id(nullptr)
{
}

TimeSourceKey::TimeSourceKey(ID *id) : id(id)
{
}

string TimeSourceKey::identifier() const
{
  return string("TimeSourceKey");
}

////////////////////////////////////////////////////////////////////////////////
// Component.

ComponentKey::ComponentKey() : id(nullptr), type(NodeType::UNDEFINED), name("")
{
}

ComponentKey::ComponentKey(ID *id, NodeType type, const char *name)
    : id(id), type(type), name(name)
{
}

string ComponentKey::identifier() const
{
  const char *idname = (id) ? id->name : "<None>";
  string result = string("ComponentKey(");
  result += idname;
  result += ", " + string(nodeTypeAsString(type));
  if (name[0] != '\0') {
    result += ", '" + string(name) + "'";
  }
  result += ')';
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// Operation.

OperationKey::OperationKey()
    : id(nullptr),
      component_type(NodeType::UNDEFINED),
      component_name(""),
      opcode(OperationCode::OPERATION),
      name(""),
      name_tag(-1)
{
}

OperationKey::OperationKey(ID *id, NodeType component_type, const char *name, int name_tag)
    : id(id),
      component_type(component_type),
      component_name(""),
      opcode(OperationCode::OPERATION),
      name(name),
      name_tag(name_tag)
{
}

OperationKey::OperationKey(
    ID *id, NodeType component_type, const char *component_name, const char *name, int name_tag)
    : id(id),
      component_type(component_type),
      component_name(component_name),
      opcode(OperationCode::OPERATION),
      name(name),
      name_tag(name_tag)
{
}

OperationKey::OperationKey(ID *id, NodeType component_type, OperationCode opcode)
    : id(id),
      component_type(component_type),
      component_name(""),
      opcode(opcode),
      name(""),
      name_tag(-1)
{
}

OperationKey::OperationKey(ID *id,
                           NodeType component_type,
                           const char *component_name,
                           OperationCode opcode)
    : id(id),
      component_type(component_type),
      component_name(component_name),
      opcode(opcode),
      name(""),
      name_tag(-1)
{
}

OperationKey::OperationKey(
    ID *id, NodeType component_type, OperationCode opcode, const char *name, int name_tag)
    : id(id),
      component_type(component_type),
      component_name(""),
      opcode(opcode),
      name(name),
      name_tag(name_tag)
{
}

OperationKey::OperationKey(ID *id,
                           NodeType component_type,
                           const char *component_name,
                           OperationCode opcode,
                           const char *name,
                           int name_tag)
    : id(id),
      component_type(component_type),
      component_name(component_name),
      opcode(opcode),
      name(name),
      name_tag(name_tag)
{
}

string OperationKey::identifier() const
{
  string result = string("OperationKey(");
  result += "type: " + string(nodeTypeAsString(component_type));
  result += ", component name: '" + string(component_name) + "'";
  result += ", operation code: " + string(operationCodeAsString(opcode));
  if (name[0] != '\0') {
    result += ", '" + string(name) + "'";
  }
  result += ")";
  return result;
}
}  // namespace blender::deg
