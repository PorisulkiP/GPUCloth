#pragma once

#include "MEM_guardedalloc.cuh"

#include "depsgraph_type.h"

struct ID;
struct PointerRNA;
struct PropertyRNA;

namespace blender {
namespace deg {

class DepsgraphBuilderCache;

/* Identifier for animated property. */
class AnimatedPropertyID {
 public:
  AnimatedPropertyID();
  AnimatedPropertyID(const PointerRNA *pointer_rna, const PropertyRNA *property_rna = nullptr);
  AnimatedPropertyID(const PointerRNA &pointer_rna, const PropertyRNA *property_rna = nullptr);

  uint64_t hash() const;
  friend bool operator==(const AnimatedPropertyID &a, const AnimatedPropertyID &b);

  /* Corresponds to PointerRNA.data. */
  void *data;
};

class AnimatedPropertyStorage {
 public:
  AnimatedPropertyStorage();

  void initializeFromID(DepsgraphBuilderCache *builder_cache, ID *id = nullptr);

  void tagPropertyAsAnimated(const AnimatedPropertyID &property_id);
  void tagPropertyAsAnimated(const PointerRNA *pointer_rna, const PropertyRNA *property_rna = nullptr);

  bool isPropertyAnimated(const AnimatedPropertyID &property_id);
  bool isPropertyAnimated(const PointerRNA *pointer_rna, const PropertyRNA *property_rna = nullptr);

  /* The storage is fully initialized from all F-Curves from corresponding ID. */
  bool is_fully_initialized;

  /* indexed by PointerRNA.data. */
  Set<AnimatedPropertyID> animated_properties_set;
};

/* Cached data which can be re-used by multiple builders. */
class DepsgraphBuilderCache {
 public:
  DepsgraphBuilderCache();
  ~DepsgraphBuilderCache();

  /* Makes sure storage for animated properties exists and initialized for the given ID. */
  AnimatedPropertyStorage *ensureAnimatedPropertyStorage(ID *id);
  AnimatedPropertyStorage *ensureInitializedAnimatedPropertyStorage(ID *id);

  /* Shortcuts to go through ensureInitializedAnimatedPropertyStorage and its
   * isPropertyAnimated.
   *
   * NOTE: Avoid using for multiple subsequent lookups, query for the storage once, and then query
   * the storage.
   *
   * TODO(sergey): Technically, this makes this class something else than just a cache, but what is
   * the better name? */
  template<typename... Args> bool isPropertyAnimated(ID *id, Args... args)
  {
    AnimatedPropertyStorage *animated_property_storage = ensureInitializedAnimatedPropertyStorage(
        id);
    return animated_property_storage->isPropertyAnimated(args...);
  }

  Map<ID *, AnimatedPropertyStorage *> animated_property_storage_map_;
};

}  // namespace deg
}  // namespace blender
