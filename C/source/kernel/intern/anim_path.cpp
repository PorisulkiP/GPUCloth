#include "MEM_guardedalloc.cuh"

#include <float.h>

#include "DNA_curve_types.h"
#include "DNA_key_types.h"
#include "object_types.cuh"

#include "math_vector.cuh"

#include "BKE_anim_path.h"
#include "BKE_curve.h"
#include "BKE_key.h"

static int interval_test(const int min, const int max, int p1, const int cycl)
{
  if (cycl) {
    p1 = mod_i(p1 - min, (max - min + 1)) + min;
  }
  else {
    if (p1 < min) {
      p1 = min;
    }
    else if (p1 > max) {
      p1 = max;
    }
  }
  return p1;
}