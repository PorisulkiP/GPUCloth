#pragma once
#include "DEG_depsgraph_debug.h"

namespace blender {
namespace deg {

class DepsgraphDebug {
 public:
  DepsgraphDebug();

  bool do_time_debug() const;

  void begin_graph_evaluation();
  void end_graph_evaluation();

  /* NOTE: Corresponds to G_DEBUG_DEPSGRAPH_* flags. */
  int flags;

  /* Name of this dependency graph (is used for debug prints, helping to distinguish graphs
   * created for different view layer). */
  string name;

  /* Is true when dependency graph was evaluated at least once.
   * This is NOT an indication that depsgraph is at its evaluated state. */
  bool is_ever_evaluated;

 protected:
  /* Maximum number of counters used to calculate frame rate of depsgraph update. */
  static const constexpr int MAX_FPS_COUNTERS = 64;

  /* Point in time when last graph evaluation began.
   * Is initialized from begin_graph_evaluation() when time debug is enabled.
   */
  double graph_evaluation_start_time_;

  AveragedTimeSampler<MAX_FPS_COUNTERS> fps_samples_;
};
#define DEG_GLOBAL_DEBUG_PRINTF(type, ...) \
  do { \
    if (G.debug & G_DEBUG_DEPSGRAPH_##type) { \
      fprintf(stdout, __VA_ARGS__); \
    } \
  } while (0)

#define DEG_ERROR_PRINTF(...) \
  do { \
    fprintf(stderr, __VA_ARGS__); \
    fflush(stderr); \
  } while (0)

}  // namespace deg
}  // namespace blender
