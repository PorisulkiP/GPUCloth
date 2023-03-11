#pragma once

#include "depsgraph.h"

struct DepsNodeHandle;
struct Depsgraph;
struct EffectorWeights;
struct ListBase;
struct Object;

static const ID* get_original_id(const ID* id)
{
    if (id == nullptr) {
        return nullptr;
    }
    if (id->orig_id == nullptr) {
        return id;
    }
    BLI_assert((id->tag & LIB_TAG_COPIED_ON_WRITE) != 0);
    return (ID*)id->orig_id;
}

static ID* get_original_id(ID* id)
{
    const ID* const_id = id;
    return const_cast<ID*>(get_original_id(const_id));
}

static const ID* get_evaluated_id(const Depsgraph* deg_graph, const ID* id)
{
    if (id == nullptr) { return nullptr; }
    const IDNode* id_node = deg_graph->find_id_node(id);
    if (id_node == nullptr) { return id; }
    return id_node->id_cow;
}

template<class T> 
static ID* object_id_safe(T* object)
{
    if (object == nullptr) {  return nullptr; }
    return &object->id;
}

static ePhysicsRelationType modifier_to_relation_type(uint modifier_type)
{
    switch (modifier_type) {
    case eModifierType_Collision:       return DEG_PHYSICS_COLLISION;
    case eModifierType_Fluid:           return DEG_PHYSICS_SMOKE_COLLISION;
    case eModifierType_DynamicPaint:    return DEG_PHYSICS_DYNAMIC_BRUSH;
    }
    return DEG_PHYSICS_RELATIONS_NUM;
}

//ListBase* DEG_get_collision_relations(const Depsgraph* graph, Collection* collection, uint modifier_type)
//{
//    const ePhysicsRelationType type = modifier_to_relation_type(modifier_type);
//    //blender::Map<const ID*, ListBase*>* hash = graph->physics_relations[type];
//    //if (!hash) { return nullptr; }
//    ID* collection_orig = get_original_id(object_id_safe(collection));
//    return hash->lookup_default(collection_orig, nullptr);
//}