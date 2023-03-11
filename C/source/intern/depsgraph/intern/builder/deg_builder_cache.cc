#include "builder/deg_builder_cache.h"

#include "MEM_guardedalloc.cuh"

#include "utildefines.h"

namespace blender::deg {

/* Animated property storage. */

AnimatedPropertyID::AnimatedPropertyID() : data(nullptr)
{
}

AnimatedPropertyID::AnimatedPropertyID(const PointerRNA *pointer_rna,
                                       const PropertyRNA *property_rna)
    : AnimatedPropertyID(*pointer_rna, property_rna)
{
}

void AnimatedPropertyStorage::tagPropertyAsAnimated(const AnimatedPropertyID &property_id)
{
  animated_properties_set.add(property_id);
}

void AnimatedPropertyStorage::tagPropertyAsAnimated(const PointerRNA *pointer_rna,
                                                    const PropertyRNA *property_rna)
{
  tagPropertyAsAnimated(AnimatedPropertyID(pointer_rna, property_rna));
}

bool AnimatedPropertyStorage::isPropertyAnimated(const AnimatedPropertyID &property_id)
{
  return animated_properties_set.contains(property_id);
}

bool AnimatedPropertyStorage::isPropertyAnimated(const PointerRNA *pointer_rna,
                                                 const PropertyRNA *property_rna)
{
  return isPropertyAnimated(AnimatedPropertyID(pointer_rna, property_rna));
}

/* Builder cache itself. */

DepsgraphBuilderCache::DepsgraphBuilderCache()
{
}

DepsgraphBuilderCache::~DepsgraphBuilderCache()
{
  for (AnimatedPropertyStorage *animated_property_storage :
       animated_property_storage_map_.values()) {
    delete animated_property_storage;
  }
}
}  // namespace blender::deg
