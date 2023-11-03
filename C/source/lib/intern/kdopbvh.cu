#include "MEM_guardedalloc.cuh"

#include "kdopbvh.cuh"
#include "B_math.h"
#include "BLI_stack.h"
#include "mallocn_intern.cuh"
#include "task.cuh"
#include "utildefines.h"

typedef unsigned char axis_t;

enum
{
	MAX_TREETYPE = 32
};

#ifdef DEBUG
#  define KDOPBVH_THREAD_LEAF_THRESHOLD 0
#else
#  define KDOPBVH_THREAD_LEAF_THRESHOLD 1024
#endif

/**
 * Bounding Volume Hierarchy Definition
 *
 * Notes: From OBB until 26-DOP --> all bounding volumes possible, just choose type below
 * Notes: You have to choose the type at compile time ITM
 * Notes: You can choose the tree type --> binary, quad, octree, choose below
 */
__device__ constexpr float d_bvhtree_kdop_axes[13][3] = {
    {1.0, 0, 0},
    {0, 1.0, 0},
    {0, 0, 1.0},
    {1.0, 1.0, 1.0},
    {1.0, -1.0, 1.0},
    {1.0, 1.0, -1.0},
    {1.0, -1.0, -1.0},
    {1.0, 1.0, 0},
    {1.0, 0, 1.0},
    {0, 1.0, 1.0},
    {1.0, -1.0, 0},
    {1.0, 0, -1.0},
    {0, 1.0, -1.0},
};

constexpr float bvhtree_kdop_axes[13][3] = {
    {1.0, 0, 0},
    {0, 1.0, 0},
    {0, 0, 1.0},
    {1.0, 1.0, 1.0},
    {1.0, -1.0, 1.0},
    {1.0, 1.0, -1.0},
    {1.0, -1.0, -1.0},
    {1.0, 1.0, 0},
    {1.0, 0, 1.0},
    {0, 1.0, 1.0},
    {1.0, -1.0, 0},
    {1.0, 0, -1.0},
    {0, 1.0, -1.0},
};

/* Used to correct the epsilon and thus match the overlap distance. */
__device__ constexpr float d_bvhtree_kdop_axes_length[13] = {
    1.0f,
    1.0f,
    1.0f,
    1.7320508075688772f,
    1.7320508075688772f,
    1.7320508075688772f,
    1.7320508075688772f,
    1.4142135623730951f,
    1.4142135623730951f,
    1.4142135623730951f,
    1.4142135623730951f,
    1.4142135623730951f,
    1.4142135623730951f,
};

/* Used to correct the epsilon and thus match the overlap distance. */
static constexpr float bvhtree_kdop_axes_length[13] = {
    1.0f,
    1.0f,
    1.0f,
    1.7320508075688772f,
    1.7320508075688772f,
    1.7320508075688772f,
    1.7320508075688772f,
    1.4142135623730951f,
    1.4142135623730951f,
    1.4142135623730951f,
    1.4142135623730951f,
    1.4142135623730951f,
    1.4142135623730951f,
};

/* -------------------------------------------------------------------- */
/** \name Utility Functions
 * \{ */

__host__ __device__ axis_t min_axis(const axis_t a, const axis_t b)
{
  return (a < b) ? a : b;
}

/**
 * Intro-sort
 * with permission deriving from the following Java code:
 * http://ralphunden.net/content/tutorials/a-guide-to-introsort/
 * and he derived it from the SUN STL
 */

__host__ __device__ static void node_minmax_init(const BVHTree *tree, const BVHNode *node)
{
	const auto bv = reinterpret_cast<float(*)[2]>(node->bv);
#ifdef __CUDA_ARCH__
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tree->stop_axis && idx > tree->start_axis)
    {
        bv[idx][0] = FLT_MAX;
        bv[idx][1] = -FLT_MAX;
    }
#else
    for (axis_t axis_iter = tree->start_axis; axis_iter != tree->stop_axis; axis_iter++)
    {
        bv[axis_iter][0] = FLT_MAX;
        bv[axis_iter][1] = -FLT_MAX;
    }
#endif
}
/** \} */

/* -------------------------------------------------------------------- */
/** \name Balance Utility Functions
 * \{ */

/**
 * Insertion sort algorithm
 */
static void bvh_insertionsort(BVHNode **a, const int lo, const int hi, const int axis)
{
  int i, j;
  BVHNode *t;
  for (i = lo; i < hi; i++) {
    j = i;
    t = a[i];
    while ((j != lo) && (t->bv[axis] < (a[j - 1])->bv[axis])) {
      a[j] = a[j - 1];
      j--;
    }
    a[j] = t;
  }
}

static int bvh_partition(BVHNode **a, const int lo, const int hi, const BVHNode *x, const int axis)
{
  int i = lo, j = hi;
  while (1) {
    while (a[i]->bv[axis] < x->bv[axis]) {
      i++;
    }
    j--;
    while (x->bv[axis] < a[j]->bv[axis]) {
      j--;
    }
    if (!(i < j)) {
      return i;
    }
    SWAP(BVHNode *, a[i], a[j]);
    i++;
  }
}

/* returns Sortable */
__host__ __device__ static BVHNode *bvh_medianof3(BVHNode **a, const int lo, const int mid, const int hi, const int axis)
{
  if ((a[mid])->bv[axis] < (a[lo])->bv[axis]) {
    if ((a[hi])->bv[axis] < (a[mid])->bv[axis]) {
      return a[mid];
    }
    if ((a[hi])->bv[axis] < (a[lo])->bv[axis]) {
      return a[hi];
    }
    return a[lo];
  }

  if ((a[hi])->bv[axis] < (a[mid])->bv[axis]) {
    if ((a[hi])->bv[axis] < (a[lo])->bv[axis]) {
      return a[lo];
    }
    return a[hi];
  }
  return a[mid];
}

/**
 * \note after a call to this function you can expect one of:
 * - every node to left of a[n] are smaller or equal to it
 * - every node to the right of a[n] are greater or equal to it */
__host__ __device__ static void partition_nth_element(BVHNode** a, int begin, int end, const int n, const int axis)
{
	while (end - begin > 3)
	{
		const int cut = bvh_partition(
			a, begin, end, bvh_medianof3(a, begin, (begin + end) / 2, end - 1, axis), axis);
		if (cut <= n)
		{
			begin = cut;
		}
		else
		{
			end = cut;
		}
	}
	bvh_insertionsort(a, begin, end, axis);
}

/*
 * BVHTree bounding volumes functions
 */
__host__ __device__ static void create_kdop_hull(const BVHTree* tree, const BVHNode* node, const float* co,
                                                 const int numpoints, const int moving)
{
	float* bv = node->bv;

	/* Don't initialize bounds for the moving case */
	if (!moving)
	{
		node_minmax_init(tree, node);
	}

	for (int k = 0; k < numpoints; k++)
	{
		/* for all Axes. */
		for (axis_t axis_iter = tree->start_axis; axis_iter < tree->stop_axis; axis_iter++)
		{
#ifdef __CUDA_ARCH__
    	const float newminmax = dot_v3v3(&co[k * 3], d_bvhtree_kdop_axes[axis_iter]);
#else
			const float newminmax = dot_v3v3(&co[k * 3], bvhtree_kdop_axes[axis_iter]);
#endif
			if (newminmax < bv[2 * axis_iter])
			{
				bv[2 * axis_iter] = newminmax;
			}
			if (newminmax > bv[(2 * axis_iter) + 1])
			{
				bv[(2 * axis_iter) + 1] = newminmax;
			}
		}
	}
}

/**
 * \note depends on the fact that the BVH's for each face is already built
 */
static void refit_kdop_hull(const BVHTree *tree, const BVHNode *node, const int start, const int end)
{
  float newmin, newmax;
  float *__restrict bv = node->bv;
  int j;
  axis_t axis_iter;

  node_minmax_init(tree, node);

  for (j = start; j < end; j++) {
    float *__restrict node_bv = tree->nodes[j]->bv;

    /* for all Axes. */
    for (axis_iter = tree->start_axis; axis_iter < tree->stop_axis; axis_iter++) {
      newmin = node_bv[(2 * axis_iter)];
      if ((newmin < bv[(2 * axis_iter)])) {
        bv[(2 * axis_iter)] = newmin;
      }

      newmax = node_bv[(2 * axis_iter) + 1];
      if ((newmax > bv[(2 * axis_iter) + 1])) {
        bv[(2 * axis_iter) + 1] = newmax;
      }
    }
  }
}

/**
 * only supports x,y,z axis in the moment
 * but we should use a plain and simple function here for speed sake */
__host__ __device__ char get_largest_axis(const float* bv)
{
#ifdef __CUDA_ARCH__
    // GPU-specific code
    float middle_point[3];

    middle_point[0] = __fsub_rn(bv[1], bv[0]); // x axis
    middle_point[1] = __fsub_rn(bv[3], bv[2]); // y axis
    middle_point[2] = __fsub_rn(bv[5], bv[4]); // z axis

    if (__fadd_rn(middle_point[0], middle_point[1]) > 0)
    {
        if (__fadd_rn(middle_point[0], middle_point[2]) > 0)
        {
            return 1; // max x axis
        }
        return 5; // max z axis
    }
    if (__fadd_rn(middle_point[1], middle_point[2]) > 0)
    {
        return 3; // max y axis
    }
    return 5; // max z axis

#else
    // CPU-specific code
    float middle_point[3];

    middle_point[0] = (bv[1]) - (bv[0]); // x axis
    middle_point[1] = (bv[3]) - (bv[2]); // y axis
    middle_point[2] = (bv[5]) - (bv[4]); // z axis

    if (middle_point[0] > middle_point[1])
    {
        if (middle_point[0] > middle_point[2])
        {
            return 1; // max x axis
        }
        return 5; // max z axis
    }
    if (middle_point[1] > middle_point[2])
    {
        return 3; // max y axis
    }
    return 5; // max z axis
#endif
}


/**
 * bottom-up update of bvh node BV
 * join the children on the parent BV */
__host__ __device__ void node_join(const BVHTree* tree, const BVHNode* node)
{
#ifdef __CUDA_ARCH__
    // GPU-specific code
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    axis_t axis_iter = tree->start_axis + idx;

    if (idx < tree->tree_type && node->children[idx]) 
    {
        if (axis_iter >= tree->start_axis && axis_iter < tree->stop_axis) 
        {
            // update minimum
            if (node->children[idx]->bv[(2 * axis_iter)] < node->bv[(2 * axis_iter)]) 
            {
                node->bv[(2 * axis_iter)] = node->children[idx]->bv[(2 * axis_iter)];
            }
            // update maximum
            if (node->children[idx]->bv[(2 * axis_iter) + 1] > node->bv[(2 * axis_iter) + 1]) 
            {
                node->bv[(2 * axis_iter) + 1] = node->children[idx]->bv[(2 * axis_iter) + 1];
            }
        }
    }

#else
    // CPU-specific code

    for (int i = 0; i < tree->tree_type; i++) 
    {
        if (node->children[i]) 
        {
            for (axis_t axis_iter = tree->start_axis; axis_iter < tree->stop_axis; axis_iter++) 
            {
                // update minimum
                if (node->children[i]->bv[(2 * axis_iter)] < node->bv[(2 * axis_iter)]) {
                    node->bv[(2 * axis_iter)] = node->children[i]->bv[(2 * axis_iter)];
                }
                // update maximum
                if (node->children[i]->bv[(2 * axis_iter) + 1] > node->bv[(2 * axis_iter) + 1]) 
                {
                    node->bv[(2 * axis_iter) + 1] = node->children[i]->bv[(2 * axis_iter) + 1];
                }
            }
        }
        else {
            break;
        }
    }
#endif
}


#ifdef USE_PRINT_TREE

/**
 * Debug and information functions
 */

static void bvhtree_print_tree(BVHTree *tree, BVHNode *node, int depth)
{
  int i;
  axis_t axis_iter;

  for (i = 0; i < depth; i++) {
    printf(" ");
  }
  printf(" - %d (%ld): ", node->index, (long int)(node - tree->nodearray));
  for (axis_iter = (axis_t)(2 * tree->start_axis); axis_iter < (axis_t)(2 * tree->stop_axis);
       axis_iter++) {
    printf("%.3f ", node->bv[axis_iter]);
  }
  printf("\n");

  for (i = 0; i < tree->tree_type; i++) {
    if (node->children[i]) {
      bvhtree_print_tree(tree, node->children[i], depth + 1);
    }
  }
}

static void bvhtree_info(BVHTree *tree)
{
  printf("BVHTree Info: tree_type = %d, axis = %d, epsilon = %f\n",
         tree->tree_type,
         tree->axis,
         tree->epsilon);
  printf("nodes = %d, branches = %d, leafs = %d\n",
         tree->totbranch + tree->totleaf,
         tree->totbranch,
         tree->totleaf);
  printf(
      "Memory per node = %ubytes\n",
      (uint)(sizeof(BVHNode) + sizeof(BVHNode *) * tree->tree_type + sizeof(float) * tree->axis));
  printf("BV memory = %ubytes\n", (uint)MEM_lockfree_allocN_len(tree->nodebv));

  printf("Total memory = %ubytes\n",
         (uint)(sizeof(BVHTree) + MEM_lockfree_allocN_len(tree->nodes) + MEM_lockfree_allocN_len(tree->nodearray) +
                MEM_lockfree_allocN_len(tree->nodechild) + MEM_lockfree_allocN_len(tree->nodebv)));

  bvhtree_print_tree(tree, tree->nodes[tree->totleaf], 0);
}
#endif /* USE_PRINT_TREE */

#ifdef USE_VERIFY_TREE

static void bvhtree_verify(BVHTree *tree)
{
  int i, j, check = 0;

  /* check the pointer list */
  for (i = 0; i < tree->totleaf; i++) {
    if (tree->nodes[i]->parent == NULL) {
      printf("Leaf has no parent: %d\n", i);
    }
    else {
      for (j = 0; j < tree->tree_type; j++) {
        if (tree->nodes[i]->parent->children[j] == tree->nodes[i]) {
          check = 1;
        }
      }
      if (!check) {
        printf("Parent child relationship doesn't match: %d\n", i);
      }
      check = 0;
    }
  }

  /* check the leaf list */
  for (i = 0; i < tree->totleaf; i++) {
    if (tree->nodearray[i].parent == NULL) {
      printf("Leaf has no parent: %d\n", i);
    }
    else {
      for (j = 0; j < tree->tree_type; j++) {
        if (tree->nodearray[i].parent->children[j] == &tree->nodearray[i]) {
          check = 1;
        }
      }
      if (!check) {
        printf("Parent child relationship doesn't match: %d\n", i);
      }
      check = 0;
    }
  }

  printf("branches: %d, leafs: %d, total: %d\n",
         tree->totbranch,
         tree->totleaf,
         tree->totbranch + tree->totleaf);
}
#endif /* USE_VERIFY_TREE */

/* Helper data and structures to build a min-leaf generalized implicit tree
 * This code can be easily reduced
 * (basically this is only method to calculate pow(k, n) in O(1).. and stuff like that) */
typedef struct BVHBuildHelper {
  int tree_type;
  int totleafs;

  /** Min number of leafs that are achievable from a node at depth `N`. */
  int leafs_per_child[32];
  /** Number of nodes at depth `N (tree_type^N)`. */
  int branches_on_level[32];

  /** Number of leafs that are placed on the level that is not 100% filled */
  int remain_leafs;

} BVHBuildHelper;

__host__ __device__ void build_implicit_tree_helper(const BVHTree *tree, BVHBuildHelper *data)
{
  int depth = 0;

  data->totleafs = tree->totleaf;
  data->tree_type = tree->tree_type;

  /* Calculate the smallest tree_type^n such that tree_type^n >= num_leafs */
  for (data->leafs_per_child[0] = 1; data->leafs_per_child[0] < data->totleafs;
       data->leafs_per_child[0] *= data->tree_type) {
    /* pass */
  }

  data->branches_on_level[0] = 1;

  for (depth = 1; (depth < 32) && data->leafs_per_child[depth - 1]; depth++) {
    data->branches_on_level[depth] = data->branches_on_level[depth - 1] * data->tree_type;
    data->leafs_per_child[depth] = data->leafs_per_child[depth - 1] / data->tree_type;
  }

  const int remain = data->totleafs - data->leafs_per_child[1];
  const int nnodes = (remain + data->tree_type - 2) / (data->tree_type - 1);
  data->remain_leafs = remain + nnodes;
}

/**
 * Return the min index of all the leafs achievable with the given branch.
 */
static int implicit_leafs_index(const BVHBuildHelper *data, const int depth, const int child_index)
{
  int min_leaf_index = child_index * data->leafs_per_child[depth - 1];
  if (min_leaf_index <= data->remain_leafs) {
    return min_leaf_index;
  }
  if (data->leafs_per_child[depth]) {
    return data->totleafs -
           (data->branches_on_level[depth - 1] - child_index) * data->leafs_per_child[depth];
  }
  return data->remain_leafs;
}

/**
 * Generalized implicit tree build
 *
 * An implicit tree is a tree where its structure is implied,
 * thus there is no need to store child pointers or indexes.
 * It's possible to find the position of the child or the parent with simple maths
 * (multiplication and addition).
 * This type of tree is for example used on heaps..
 * where node N has its child at indices N*2 and N*2+1.
 *
 * Although in this case the tree type is general.. and not know until run-time.
 * tree_type stands for the maximum number of children that a tree node can have.
 * All tree types >= 2 are supported.
 *
 * Advantages of the used trees include:
 * - No need to store child/parent relations (they are implicit);
 * - Any node child always has an index greater than the parent;
 * - Brother nodes are sequential in memory;
 * Some math relations derived for general implicit trees:
 *
 *   K = tree_type, ( 2 <= K )
 *   ROOT = 1
 *   N child of node A = A * K + (2 - K) + N, (0 <= N < K)
 *
 * Util methods:
 *   TODO...
 *    (looping elements, knowing if its a leaf or not.. etc...)
 */

/* This functions returns the number of branches needed to have the requested number of leafs. */
__host__ __device__ static int implicit_needed_branches(const int tree_type, const int leafs)
{
  return max_ii(1, (leafs + tree_type - 3) / (tree_type - 1));
}

/**
 * This function handles the problem of "sorting" the leafs (along the split_axis).
 *
 * It arranges the elements in the given partitions such that:
 * - any element in partition N is less or equal to any element in partition N+1.
 * - if all elements are different all partition will get the same subset of elements
 *   as if the array was sorted.
 *
 * partition P is described as the elements in the range ( nth[P], nth[P+1] ]
 *
 * TODO: This can be optimized a bit by doing a specialized nth_element instead of K nth_elements
 */
__host__ __device__ static void split_leafs(BVHNode **leafs_array,
                        const int nth[],
                        const int partitions,
                        const int split_axis)
{
  int i;
  for (i = 0; i < partitions - 1; i++) {
    if (nth[i] >= nth[partitions]) {
      break;
    }

    partition_nth_element(leafs_array, nth[i], nth[partitions], nth[i + 1], split_axis);
  }
}

typedef struct BVHDivNodesData {
  const BVHTree *tree;
  BVHNode *branches_array;
  BVHNode **leafs_array;

  int tree_type;
  int tree_offset;

  const BVHBuildHelper *data;

  int depth;
  int i;
  int first_of_next_level;
} BVHDivNodesData;

static void non_recursive_bvh_div_nodes_task_cb(void *__restrict userdata, const int j)
{
  BVHDivNodesData *data = (BVHDivNodesData*)userdata;

  int k;
  const int parent_level_index = j - data->i;
  BVHNode *parent = &data->branches_array[j];
  int nth_positions[MAX_TREETYPE + 1];
  char split_axis;

  int parent_leafs_begin = implicit_leafs_index(data->data, data->depth, parent_level_index);
  int parent_leafs_end = implicit_leafs_index(data->data, data->depth, parent_level_index + 1);

  /* This calculates the bounding box of this branch
   * and chooses the largest axis as the axis to divide leafs */
  refit_kdop_hull(data->tree, parent, parent_leafs_begin, parent_leafs_end);
  split_axis = get_largest_axis(parent->bv);

  /* Save split axis (this can be used on ray-tracing to speedup the query time) */
  parent->main_axis = split_axis / 2;

  /* Split the children along the split_axis, note: its not needed to sort the whole leafs array
   * Only to assure that the elements are partitioned on a way that each child takes the elements
   * it would take in case the whole array was sorted.
   * Split_leafs takes care of that "sort" problem. */
  nth_positions[0] = parent_leafs_begin;
  nth_positions[data->tree_type] = parent_leafs_end;
  for (k = 1; k < data->tree_type; k++) {
    const int child_index = j * data->tree_type + data->tree_offset + k;
    /* child level index */
    const int child_level_index = child_index - data->first_of_next_level;
    nth_positions[k] = implicit_leafs_index(data->data, data->depth + 1, child_level_index);
  }

  split_leafs(data->leafs_array, nth_positions, data->tree_type, split_axis);

  /* Setup children and totnode counters
   * Not really needed but currently most of BVH code
   * relies on having an explicit children structure */
  for (k = 0; k < data->tree_type; k++) {
    const int child_index = j * data->tree_type + data->tree_offset + k;
    /* child level index */
    const int child_level_index = child_index - data->first_of_next_level;

    const int child_leafs_begin = implicit_leafs_index(
        data->data, data->depth + 1, child_level_index);
    const int child_leafs_end = implicit_leafs_index(
        data->data, data->depth + 1, child_level_index + 1);

    if (child_leafs_end - child_leafs_begin > 1) {
      parent->children[k] = &data->branches_array[child_index];
      parent->children[k]->parent = parent;
    }
    else if (child_leafs_end - child_leafs_begin == 1) {
      parent->children[k] = data->leafs_array[child_leafs_begin];
      parent->children[k]->parent = parent;
    }
    else {
      break;
    }
  }
  parent->totnode = (char)k;
}

/**
 * This functions builds an optimal implicit tree from the given leafs.
 * Where optimal stands for:
 * - The resulting tree will have the smallest number of branches;
 * - At most only one branch will have NULL children;
 * - All leafs will be stored at level N or N+1.
 *
 * This function creates an implicit tree on branches_array,
 * the leafs are given on the leafs_array.
 *
 * The tree is built per depth levels. First branches at depth 1.. then branches at depth 2.. etc..
 * The reason is that we can build level N+1 from level N without any data dependencies..
 * thus it allows to use multi-thread building.
 *
 * To archive this is necessary to find how much leafs are accessible from a certain branch,
 * #BVHBuildHelper, #implicit_needed_branches and #implicit_leafs_index
 * are auxiliary functions to solve that "optimal-split".
 */
__host__ __device__ void non_recursive_bvh_div_nodes(const BVHTree *tree, BVHNode *branches_array, BVHNode **leafs_array, const int num_leafs)
{
  int i;

  const int tree_type = tree->tree_type;
  /* this value is 0 (on binary trees) and negative on the others */
  const int tree_offset = 2 - tree->tree_type;

  const int num_branches = implicit_needed_branches(tree_type, num_leafs);

  BVHBuildHelper data;
  int depth;

  /* set parent from root node to NULL */
  BVHNode* root = &branches_array[1];
  root->parent = nullptr;

  /* Most of bvhtree code relies on 1-leaf trees having at least one branch
   * We handle that special case here */
  if (num_leafs == 1) 
  {
      refit_kdop_hull(tree, root, 0, num_leafs);
      root->main_axis = get_largest_axis(root->bv) / 2;
      root->totnode = 1;
      root->children[0] = leafs_array[0];
      root->children[0]->parent = root;
      return;
  }

  build_implicit_tree_helper(tree, &data);

  BVHDivNodesData cb_data = {
      .tree = tree,
      .branches_array = branches_array,
      .leafs_array = leafs_array,
      .tree_type = tree_type,
      .tree_offset = tree_offset,
      .data = &data,
      .first_of_next_level = 0
  };

  /* Loop tree levels (log N) loops */
  for (i = 1, depth = 1; i <= num_branches; i = i * tree_type + tree_offset, depth++) 
  {
    const int first_of_next_level = i * tree_type + tree_offset;
    /* index of last branch on this level */
    const int i_stop = min_ii(first_of_next_level, num_branches + 1);

    /* Loop all branches on this level */
    cb_data.first_of_next_level = first_of_next_level;
    cb_data.i = i;
    cb_data.depth = depth;

    if constexpr (true)
    {
      TaskParallelSettings settings;
      BLI_parallel_range_settings_defaults(&settings);
      settings.use_threading = (num_leafs > KDOPBVH_THREAD_LEAF_THRESHOLD);
      BLI_task_parallel_range(i, i_stop, &cb_data, reinterpret_cast<TaskParallelRangeFunc>(non_recursive_bvh_div_nodes_task_cb), &settings);
    }
    else 
    {
      /* Less hassle for debugging. */
      TaskParallelTLS tls = {nullptr};
      for (int i_task = i; i_task < i_stop; i_task++) {
        non_recursive_bvh_div_nodes_task_cb(&cb_data, i_task);
      }
    }
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name BLI_bvhtree API
 * \{ */

/**
 * \note many callers don't check for ``NULL`` return.
 */
__host__ __device__ BVHTree *BLI_bvhtree_new(const uint maxsize, float epsilon, const char tree_type, const char axis)
{
  BVHTree *tree;
  int numnodes, i;

  BLI_assert(tree_type >= 2 && tree_type <= MAX_TREETYPE);

  tree = (BVHTree*)MEM_lockfree_callocN(sizeof(BVHTree), "BVHTree");

  /* tree epsilon must be >= FLT_EPSILON
   * so that tangent rays can still hit a bounding volume..
   * this bug would show up when casting a ray aligned with a kdop-axis
   * and with an edge of 2 faces */
  epsilon = max_ff(FLT_EPSILON, epsilon);

  if (tree) {
    tree->epsilon = epsilon;
    tree->tree_type = tree_type;
    tree->axis = axis;

    if (axis == 26) {
      tree->start_axis = 0;
      tree->stop_axis = 13;
    }
    else if (axis == 18) {
      tree->start_axis = 7;
      tree->stop_axis = 13;
    }
    else if (axis == 14) {
      tree->start_axis = 0;
      tree->stop_axis = 7;
    }
    else if (axis == 8) { /* AABB */
      tree->start_axis = 0;
      tree->stop_axis = 4;
    }
    else if (axis == 6) { /* OBB */
      tree->start_axis = 0;
      tree->stop_axis = 3;
    }
    else {
      /* should never happen! */
      BLI_assert(0);

      goto fail;
    }

    /* Allocate arrays */
    numnodes = maxsize + implicit_needed_branches(tree_type, maxsize) + tree_type;

    tree->nodes = (BVHNode**)MEM_lockfree_callocN(sizeof(BVHNode *) * (size_t)numnodes, "BVHNodes");
    tree->nodebv = (float*)MEM_lockfree_callocN(sizeof(float) * (size_t)(axis * numnodes), "BVHNodeBV");
    tree->nodechild = (BVHNode**)MEM_lockfree_callocN(sizeof(BVHNode *) * (size_t)(tree_type * numnodes), "BVHNodeBV");
    tree->nodearray = (BVHNode*)MEM_lockfree_callocN(sizeof(BVHNode) * (size_t)numnodes, "BVHNodeArray");

    if (UNLIKELY((!tree->nodes) || (!tree->nodebv) || (!tree->nodechild) || (!tree->nodearray))) {
      goto fail;
    }

    /* link the dynamic bv and child links */
    for (i = 0; i < numnodes; i++) {
      tree->nodearray[i].bv = &tree->nodebv[i * axis];
      tree->nodearray[i].children = &tree->nodechild[i * tree_type];
    }
  }
  return tree;

fail:
  BLI_bvhtree_free(tree);
  return nullptr;
}

__host__ __device__ void BLI_bvhtree_free(BVHTree *tree)
{
  if (tree) {
    MEM_SAFE_FREE(tree->nodes);
    MEM_SAFE_FREE(tree->nodearray);
    MEM_SAFE_FREE(tree->nodebv);
    MEM_SAFE_FREE(tree->nodechild);
    MEM_lockfree_freeN(tree);
  }
}

__host__ __device__ void BLI_bvhtree_balance(BVHTree *tree)
{
  BVHNode **leafs_array = tree->nodes;

  /* This function should only be called once
   * (some big bug goes here if its being called more than once per tree) */
  BLI_assert(tree->totbranch == 0);

  /* Build the implicit tree */
  non_recursive_bvh_div_nodes(tree, tree->nodearray + (tree->totleaf - 1), leafs_array, tree->totleaf);

  /* current code expects the branches to be linked to the nodes array
   * we perform that linkage here */
  tree->totbranch = implicit_needed_branches(tree->tree_type, tree->totleaf);

#ifdef __CUDA_ARCH__
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < tree->totbranch)
  {
    tree->nodes[tree->totleaf + idx] = &tree->nodearray[tree->totleaf + idx];
  }
#else
  for (int i = 0; i < tree->totbranch; i++)
  {
      tree->nodes[tree->totleaf + i] = &tree->nodearray[tree->totleaf + i];
  }
#endif	

#ifdef USE_SKIP_LINKS
  build_skip_links(tree, tree->nodes[tree->totleaf], NULL, NULL);
#endif

#ifdef USE_VERIFY_TREE
  bvhtree_verify(tree);
#endif

#ifdef USE_PRINT_TREE
  bvhtree_info(tree);
#endif
}

__host__ __device__ void bvhtree_node_inflate(const BVHTree* tree, const BVHNode* node, const float dist)
{
#ifdef __CUDA_ARCH__
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tree->start_axis && idx < tree->stop_axis)
    {
        float dist_corrected = dist * d_bvhtree_kdop_axes_length[idx];
        node->bv[(2 * idx)] -= dist_corrected; /* minimum */
        node->bv[(2 * idx) + 1] += dist_corrected; /* maximum */
    }
#else
    for (axis_t axis_iter = tree->start_axis; axis_iter < tree->stop_axis; axis_iter++)
    {
		float dist_corrected = dist * bvhtree_kdop_axes_length[axis_iter];
        node->bv[(2 * axis_iter)] -= dist_corrected; /* minimum */
        node->bv[(2 * axis_iter) + 1] += dist_corrected; /* maximum */	
    }
#endif	
}

__host__ __device__ void BLI_bvhtree_insert(BVHTree* tree, const int index, const float co[3], const int numpoints)
{
	/* insert should only possible as long as tree->totbranch is 0 */
	BLI_assert(tree->totbranch <= 0);
	BLI_assert((size_t)tree->totleaf < MEM_lockfree_allocN_len(tree->nodes) / sizeof(*(tree->nodes)));

	BVHNode* node = tree->nodes[tree->totleaf] = &(tree->nodearray[tree->totleaf]);
	tree->totleaf++;

	create_kdop_hull(tree, node, co, numpoints, 0);
	node->index = index;

	/* inflate the bv with some epsilon */
	bvhtree_node_inflate(tree, node, tree->epsilon);
}

/* call before BLI_bvhtree_update_tree() */
__host__ __device__ bool BLI_bvhtree_update_node(const BVHTree *tree, const int index, const float co[3], const float co_moving[3], const int numpoints)
{
	/* check if index exists */
  if (index > tree->totleaf) {
    return false;
  }

  BVHNode* node = tree->nodearray + index;

  create_kdop_hull(tree, node, co, numpoints, 0);

  if (co_moving) {
    create_kdop_hull(tree, node, co_moving, numpoints, 1);
  }

  /* inflate the bv with some epsilon */
  bvhtree_node_inflate(tree, node, tree->epsilon);

  return true;
}
/**
 * Call #BLI_bvhtree_update_node() first for every node/point/triangle.
 */
__host__ __device__ void BLI_bvhtree_update_tree(BVHTree *tree)
{
  /* Update bottom=>top
   * TRICKY: the way we build the tree all the children have an index greater than the parent
   * This allows us todo a bottom up update by starting on the bigger numbered branch. */

  BVHNode **root = tree->nodes + tree->totleaf;
  BVHNode **index = tree->nodes + tree->totleaf + tree->totbranch - 1;

  for (; index >= root; index--) {
    node_join(tree, *index);
  }
}

/**
 * Number of times #BLI_bvhtree_insert has been called.
 * mainly useful for asserts functions to check we added the correct number.
 */
__host__ __device__ int BLI_bvhtree_get_len(const BVHTree *tree)
{
  return tree->totleaf;
}

/**
 * Maximum number of children that a node can have.
 */
__host__ __device__ int BLI_bvhtree_get_tree_type(const BVHTree *tree)
{
  return tree->tree_type;
}

__host__ __device__ float BLI_bvhtree_get_epsilon(const BVHTree *tree)
{
  return tree->epsilon;
}

/**
 * This function returns the bounding box of the BVH tree.
 */
void BLI_bvhtree_get_bounding_box(const BVHTree *tree, float r_bb_min[3], float r_bb_max[3])
{
  BVHNode *root = tree->nodes[tree->totleaf];
  if (root != nullptr) {
    const float bb_min[3] = {root->bv[0], root->bv[2], root->bv[4]};
    const float bb_max[3] = {root->bv[1], root->bv[3], root->bv[5]};
    copy_v3_v3(r_bb_min, bb_min);
    copy_v3_v3(r_bb_max, bb_max);
  }
  else {
    BLI_assert(false);
    zero_v3(r_bb_min);
    zero_v3(r_bb_max);
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name BLI_bvhtree_overlap
 * \{ */

/**
 * overlap - is it possible for 2 bv's to collide ?
 */
static bool tree_overlap_test(const BVHNode *node1,
                              const BVHNode *node2,
                              const axis_t start_axis,
                              const axis_t stop_axis)
{
  const float *bv1 = node1->bv + (start_axis << 1);
  const float *bv2 = node2->bv + (start_axis << 1);
  const float *bv1_end = node1->bv + (stop_axis << 1);

  /* test all axis if min + max overlap */
  for (; bv1 != bv1_end; bv1 += 2, bv2 += 2) {
    if ((bv1[0] > bv2[1]) || (bv2[0] > bv1[1])) {
      return 0;
    }
  }

  return 1;
}

__device__ static bool d_tree_overlap_test(const BVHNode* node1,
    const BVHNode* node2,
    const axis_t start_axis,
    const axis_t stop_axis)
{
    const float* bv1 = node1->bv + (start_axis << 1);
    const float* bv2 = node2->bv + (start_axis << 1);
    const float* bv1_end = node1->bv + (stop_axis << 1);

    /* test all axis if min + max overlap */
    for (; bv1 != bv1_end; bv1 += 2, bv2 += 2) {
        if ((bv1[0] > bv2[1]) || (bv2[0] > bv1[1])) {
            return false;
        }
    }

    return true;
}

static void tree_overlap_traverse(BVHOverlapData_Thread *data_thread,
                                  const BVHNode *node1,
                                  const BVHNode *node2)
{
  BVHOverlapData_Shared *data = (BVHOverlapData_Shared*)data_thread->shared;
  int j;

  if (tree_overlap_test(node1, node2, data->start_axis, data->stop_axis)) {
    /* check if node1 is a leaf */
    if (!node1->totnode) {
      /* check if node2 is a leaf */
      if (!node2->totnode) {
        BVHTreeOverlap *overlap;

        if (UNLIKELY(node1 == node2)) {
          return;
        }

        /* both leafs, insert overlap! */
        overlap = (BVHTreeOverlap*)BLI_stack_push_r(data_thread->overlap);
        overlap->indexA = node1->index;
        overlap->indexB = node2->index;
      }
      else {
        for (j = 0; j < data->tree2->tree_type; ++j) {
          if (node2->children[j]) {
            tree_overlap_traverse(data_thread, node1, node2->children[j]);
          }
        }
      }
    }
    else {
      for (j = 0; j < data->tree1->tree_type; ++j) {
        if (node1->children[j]) {
          tree_overlap_traverse(data_thread, node1->children[j], node2);
        }
      }
    }
  }
}

__device__ static void d_tree_overlap_traverse(BVHOverlapData_Thread* data_thread,
    const BVHNode* node1,
    const BVHNode* node2)
{
	const auto* data = data_thread->shared;
    int j;

    if (d_tree_overlap_test(node1, node2, data->start_axis, data->stop_axis)) {
        /* check if node1 is a leaf */
        if (!node1->totnode) {
            /* check if node2 is a leaf */
            if (!node2->totnode) {
	            if (UNLIKELY(node1 == node2)) {
                    return;
                }

                /* both leafs, insert overlap! */
	            auto overlap = static_cast<BVHTreeOverlap*>(BLI_stack_push_r(data_thread->overlap));
                overlap->indexA = node1->index;
                overlap->indexB = node2->index;
            }
            else {
                for (j = 0; j < data->tree2->tree_type; ++j) {
                    if (node2->children[j]) {
                        d_tree_overlap_traverse(data_thread, node1, node2->children[j]);
                    }
                }
            }
        }
        else {
            for (j = 0; j < data->tree1->tree_type; ++j) {
                if (node1->children[j]) {
                    d_tree_overlap_traverse(data_thread, node1->children[j], node2);
                }
            }
        }
    }
}

/**
 * a version of #tree_overlap_traverse that runs a callback to check if the nodes really intersect.
 */
static void tree_overlap_traverse_cb(BVHOverlapData_Thread *data_thread,
                                     const BVHNode *node1,
                                     const BVHNode *node2)
{
  BVHOverlapData_Shared *data = data_thread->shared;
  int j;

  if (tree_overlap_test(node1, node2, data->start_axis, data->stop_axis)) {
    /* check if node1 is a leaf */
    if (!node1->totnode) {
      /* check if node2 is a leaf */
      if (!node2->totnode) {
        BVHTreeOverlap *overlap;

        if (UNLIKELY(node1 == node2)) {
          return;
        }

        /* only difference to tree_overlap_traverse! */
        if (data->callback(data->userdata, node1->index, node2->index, data_thread->thread)) {
          /* both leafs, insert overlap! */
          overlap = (BVHTreeOverlap*)BLI_stack_push_r(data_thread->overlap);
          overlap->indexA = node1->index;
          overlap->indexB = node2->index;
        }
      }
      else {
        for (j = 0; j < data->tree2->tree_type; j++) {
          if (node2->children[j]) {
            tree_overlap_traverse_cb(data_thread, node1, node2->children[j]);
          }
        }
      }
    }
    else {
      for (j = 0; j < data->tree1->tree_type; j++) {
        if (node1->children[j]) {
          tree_overlap_traverse_cb(data_thread, node1->children[j], node2);
        }
      }
    }
  }
}

__device__ static void d_tree_overlap_traverse_cb(BVHOverlapData_Thread* data_thread,
    const BVHNode* node1,
    const BVHNode* node2)
{
	const BVHOverlapData_Shared* data = data_thread->shared;
    int j;

    if (d_tree_overlap_test(node1, node2, data->start_axis, data->stop_axis)) 
    {
        /* check if node1 is a leaf */
        if (!node1->totnode) 
        {
            /* check if node2 is a leaf */
            if (!node2->totnode) 
            {
	            if (UNLIKELY(node1 == node2)) 
                {
                    return;
                }

                /* only difference to tree_overlap_traverse! */
                if (data->callback(data->userdata, node1->index, node2->index, data_thread->thread)) 
                {
                    /* both leafs, insert overlap! */
                    auto* overlap = static_cast<BVHTreeOverlap*>(BLI_stack_push_r(data_thread->overlap));
                    overlap->indexA = node1->index;
                    overlap->indexB = node2->index;
                }
            }
            else
            {
                for (j = 0; j < data->tree2->tree_type; j++) 
                {
                    if (node2->children[j]) {
                        d_tree_overlap_traverse_cb(data_thread, node1, node2->children[j]);
                    }
                }
            }
        }
        else 
        {
            for (j = 0; j < data->tree1->tree_type; j++) 
            {
                if (node1->children[j]) 
                {
                    d_tree_overlap_traverse_cb(data_thread, node1->children[j], node2);
                }
            }
        }
    }
}


/**
 * a version of #tree_overlap_traverse_cb that that break on first true return.
 */
static bool tree_overlap_traverse_num(BVHOverlapData_Thread *data_thread,
                                      const BVHNode *node1,
                                      const BVHNode *node2)
{
  BVHOverlapData_Shared *data = data_thread->shared;
  int j;

  if (tree_overlap_test(node1, node2, data->start_axis, data->stop_axis)) {
    /* check if node1 is a leaf */
    if (!node1->totnode) {
      /* check if node2 is a leaf */
      if (!node2->totnode) {
        BVHTreeOverlap *overlap;

        if (UNLIKELY(node1 == node2)) {
          return false;
        }

        /* only difference to tree_overlap_traverse! */
        if (!data->callback ||
            data->callback(data->userdata, node1->index, node2->index, data_thread->thread)) {
          /* both leafs, insert overlap! */
          if (data_thread->overlap) {
            overlap = (BVHTreeOverlap*)BLI_stack_push_r(data_thread->overlap);
            overlap->indexA = node1->index;
            overlap->indexB = node2->index;
          }
          return (--data_thread->max_interactions) == 0;
        }
      }
      else {
        for (j = 0; j < node2->totnode; j++) {
          if (tree_overlap_traverse_num(data_thread, node1, node2->children[j])) {
            return true;
          }
        }
      }
    }
    else {
      const uint max_interactions = data_thread->max_interactions;
      for (j = 0; j < node1->totnode; j++) {
        if (tree_overlap_traverse_num(data_thread, node1->children[j], node2)) {
          data_thread->max_interactions = max_interactions;
        }
      }
    }
  }
  return false;
}

__device__ static bool d_tree_overlap_traverse_num(BVHOverlapData_Thread* data_thread,
    const BVHNode* node1,
    const BVHNode* node2)
{
	const BVHOverlapData_Shared* data = data_thread->shared;
    int j;

    if (d_tree_overlap_test(node1, node2, data->start_axis, data->stop_axis)) {
        /* check if node1 is a leaf */
        if (!node1->totnode) {
            /* check if node2 is a leaf */
            if (!node2->totnode) {
	            if (UNLIKELY(node1 == node2)) {
                    return false;
                }

                /* only difference to tree_overlap_traverse! */
                if (!data->callback ||
                    data->callback(data->userdata, node1->index, node2->index, data_thread->thread)) {
                    /* both leafs, insert overlap! */
                    if (data_thread->overlap) {
	                    auto* overlap = static_cast<BVHTreeOverlap*>(BLI_stack_push_r(data_thread->overlap));
                        overlap->indexA = node1->index;
                        overlap->indexB = node2->index;
                    }
                    return (--data_thread->max_interactions) == 0;
                }
            }
            else {
                for (j = 0; j < node2->totnode; j++) {
                    if (d_tree_overlap_traverse_num(data_thread, node1, node2->children[j])) {
                        return true;
                    }
                }
            }
        }
        else {
            const uint max_interactions = data_thread->max_interactions;
            for (j = 0; j < node1->totnode; j++) {
                if (d_tree_overlap_traverse_num(data_thread, node1->children[j], node2)) {
                    data_thread->max_interactions = max_interactions;
                }
            }
        }
    }
    return false;
}

/**
 * Use to check the total number of threads #BLI_bvhtree_overlap will use.
 *
 * \warning Must be the first tree passed to #BLI_bvhtree_overlap!
 */
__host__ __device__ int BLI_bvhtree_overlap_thread_num(const BVHTree *tree)
{
  return (int)MIN2(tree->tree_type, tree->nodes[tree->totleaf]->totnode);
}

__host__ __device__ static void bvhtree_overlap_task_cb(void *__restrict userdata, const int j)
{
  BVHOverlapData_Thread *data = &((BVHOverlapData_Thread *)userdata)[j];
  BVHOverlapData_Shared *data_shared = data->shared;

  if (data->max_interactions) {
    tree_overlap_traverse_num(data,
                              data_shared->tree1->nodes[data_shared->tree1->totleaf]->children[j],
                              data_shared->tree2->nodes[data_shared->tree2->totleaf]);
  }
  else if (data_shared->callback) {
    tree_overlap_traverse_cb(data,
                             data_shared->tree1->nodes[data_shared->tree1->totleaf]->children[j],
                             data_shared->tree2->nodes[data_shared->tree2->totleaf]);
  }
  else {
    tree_overlap_traverse(data,
                          data_shared->tree1->nodes[data_shared->tree1->totleaf]->children[j],
                          data_shared->tree2->nodes[data_shared->tree2->totleaf]);
  }
}

__host__ __device__ BVHTreeOverlap *BLI_bvhtree_overlap_ex(
    const BVHTree *tree1,
    const BVHTree *tree2,
    uint *r_overlap_tot,
    /* optional callback to test the overlap before adding (must be thread-safe!) */
    const BVHTree_OverlapCallback callback,
    void *userdata,
    const uint max_interactions,
    const int flag)
{
  bool overlap_pairs = (flag & BVH_OVERLAP_RETURN_PAIRS) != 0;
  bool use_threading = (flag & BVH_OVERLAP_USE_THREADING) != 0 &&
                       (tree1->totleaf > KDOPBVH_THREAD_LEAF_THRESHOLD);

  /* 'RETURN_PAIRS' was not implemented without 'max_interactions'. */
  BLI_assert(overlap_pairs || max_interactions);

  const int root_node_len = BLI_bvhtree_overlap_thread_num(tree1);
  const int thread_num = use_threading ? root_node_len : 1;
  int j;
  size_t total = 0;
  BVHTreeOverlap *overlap = nullptr, *to = nullptr;
  BVHOverlapData_Shared data_shared;
  BVHOverlapData_Thread *data = static_cast<BVHOverlapData_Thread*>(BLI_array_alloca(data, static_cast<size_t>(thread_num)));
  axis_t start_axis, stop_axis;

  /* check for compatibility of both trees (can't compare 14-DOP with 18-DOP) */
  if (UNLIKELY((tree1->axis != tree2->axis) && (tree1->axis == 14 || tree2->axis == 14) &&
               (tree1->axis == 18 || tree2->axis == 18))) {
    BLI_assert(0);
    return nullptr;
  }

  const BVHNode *root1 = tree1->nodes[tree1->totleaf];
  const BVHNode *root2 = tree2->nodes[tree2->totleaf];

  start_axis = min_axis(tree1->start_axis, tree2->start_axis);
  stop_axis = min_axis(tree1->stop_axis, tree2->stop_axis);

  /* fast check root nodes for collision before doing big splitting + traversal */
  if (!tree_overlap_test(root1, root2, start_axis, stop_axis)) {
    return nullptr;
  }

  data_shared.tree1 = tree1;
  data_shared.tree2 = tree2;
  data_shared.start_axis = start_axis;
  data_shared.stop_axis = stop_axis;

  /* can be NULL */
  data_shared.callback = callback;
  data_shared.userdata = userdata;

  for (j = 0; j < thread_num; j++) {
    /* init BVHOverlapData_Thread */
    data[j].shared = &data_shared;
    data[j].overlap = overlap_pairs ? BLI_stack_new(sizeof(BVHTreeOverlap), __func__) : nullptr;
    data[j].max_interactions = max_interactions;

    /* for callback */
    data[j].thread = j;
  }

  if (use_threading) {
    TaskParallelSettings settings;
    BLI_parallel_range_settings_defaults(&settings);
    settings.min_iter_per_thread = 1;
    BLI_task_parallel_range(0, root_node_len, data, reinterpret_cast<TaskParallelRangeFunc>(bvhtree_overlap_task_cb), &settings);
  }
  else {
    if (max_interactions) {
      tree_overlap_traverse_num(data, root1, root2);
    }
    else if (callback) {
      tree_overlap_traverse_cb(data, root1, root2);
    }
    else {
      tree_overlap_traverse(data, root1, root2);
    }
  }

  if (overlap_pairs) {
    for (j = 0; j < thread_num; j++) {
      total += BLI_stack_count(data[j].overlap);
    }

    to = overlap = static_cast<BVHTreeOverlap*>(MEM_lockfree_mallocN(sizeof(BVHTreeOverlap) * total, "BVHTreeOverlap"));

    for (j = 0; j < thread_num; j++) {
      uint count = (uint)BLI_stack_count(data[j].overlap);
      BLI_stack_pop_n(data[j].overlap, to, count);
      BLI_stack_free(data[j].overlap);
      to += count;
    }
    *r_overlap_tot = (uint)total;
  }

  return overlap;
}

__host__ __device__ BVHTreeOverlap *BLI_bvhtree_overlap(
    const BVHTree *tree1,
    const BVHTree *tree2,
    uint *r_overlap_tot,
    /* optional callback to test the overlap before adding (must be thread-safe!) */
    const BVHTree_OverlapCallback callback,
    void *userdata)
{
  return BLI_bvhtree_overlap_ex(tree1,
                                tree2,
                                r_overlap_tot,
                                callback,
                                userdata,
                                0,
                                BVH_OVERLAP_USE_THREADING | BVH_OVERLAP_RETURN_PAIRS);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name BLI_bvhtree_intersect_plane
 * \{ */

static bool tree_intersect_plane_test(const float *bv, const float plane[4])
{
  /* TODO(germano): Support other kdop geometries. */
  const float bb_min[3] = {bv[0], bv[2], bv[4]};
  const float bb_max[3] = {bv[1], bv[3], bv[5]};
  float bb_near[3], bb_far[3];
  aabb_get_near_far_from_plane(plane, bb_min, bb_max, bb_near, bb_far);
  if ((plane_point_side_v3(plane, bb_near) > 0.0f) !=
      (plane_point_side_v3(plane, bb_far) > 0.0f)) {
    return true;
  }

  return false;
}
//
//static void bvhtree_intersect_plane_dfs_recursive(BVHIntersectPlaneData *__restrict data,
//                                                  const BVHNode *node)
//{
//  if (tree_intersect_plane_test(node->bv, data->plane)) {
//    /* check if node is a leaf */
//    if (!node->totnode) {
//      int *intersect = (int*)BLI_stack_push_r(data->intersect);
//      *intersect = node->index;
//    }
//    else {
//      for (int j = 0; j < data->tree->tree_type; j++) {
//        if (node->children[j]) {
//          bvhtree_intersect_plane_dfs_recursive(data, node->children[j]);
//        }
//      }
//    }
//  }
//}
//
//int *BLI_bvhtree_intersect_plane(BVHTree *tree, float plane[4], uint *r_intersect_tot)
//{
//  int *intersect = nullptr;
//  size_t total = 0;
//
//  if (tree->totleaf) {
//    BVHIntersectPlaneData data;
//    data.tree = tree;
//    copy_v4_v4(data.plane, plane);
//    data.intersect = (BLI_Stack*)BLI_stack_new(sizeof(int), __func__);
//
//    BVHNode *root = tree->nodes[tree->totleaf];
//    bvhtree_intersect_plane_dfs_recursive(&data, root);
//
//    total = BLI_stack_count(data.intersect);
//    if (total) {
//      intersect =(int*)MEM_lockfree_mallocN(sizeof(int) * total, __func__);
//      BLI_stack_pop_n(data.intersect, intersect, (uint)total);
//    }
//    BLI_stack_free(data.intersect);
//  }
//  *r_intersect_tot = (uint)total;
//  return intersect;
//}

/** \} */

/* -------------------------------------------------------------------- */
/** \name BLI_bvhtree_find_nearest
 * \{ */

/* Determines the nearest point of the given node BV.
 * Returns the squared distance to that point. */
static float calc_nearest_point_squared(const float proj[3], const BVHNode *node, float nearest[3])
{
  int i;
  const float *bv = node->bv;

  /* nearest on AABB hull */
  for (i = 0; i != 3; i++, bv += 2) {
    float val = proj[i];
    if (bv[0] > val) {
      val = bv[0];
    }
    if (bv[1] < val) {
      val = bv[1];
    }
    nearest[i] = val;
  }

  return len_squared_v3v3(proj, nearest);
}

/* Depth first search method */
static void dfs_find_nearest_dfs(BVHNearestData *data, BVHNode *node)
{
  if (node->totnode == 0) {
    if (data->callback) {
      data->callback(data->userdata, node->index, data->co, &data->nearest);
    }
    else {
      data->nearest.index = node->index;
      data->nearest.dist_sq = calc_nearest_point_squared(data->proj, node, data->nearest.co);
    }
  }
  else {
    /* Better heuristic to pick the closest node to dive on */
    int i;
    float nearest[3];

    if (data->proj[node->main_axis] <= node->children[0]->bv[node->main_axis * 2 + 1]) {

      for (i = 0; i != node->totnode; i++) {
        if (calc_nearest_point_squared(data->proj, node->children[i], nearest) >=
            data->nearest.dist_sq) {
          continue;
        }
        dfs_find_nearest_dfs(data, node->children[i]);
      }
    }
    else {
      for (i = node->totnode - 1; i >= 0; i--) {
        if (calc_nearest_point_squared(data->proj, node->children[i], nearest) >=
            data->nearest.dist_sq) {
          continue;
        }
        dfs_find_nearest_dfs(data, node->children[i]);
      }
    }
  }
}

static void dfs_find_nearest_begin(BVHNearestData *data, BVHNode *node)
{
  float nearest[3], dist_sq;
  dist_sq = calc_nearest_point_squared(data->proj, node, nearest);
  if (dist_sq >= data->nearest.dist_sq) {
    return;
  }
  dfs_find_nearest_dfs(data, node);
}

/* Priority queue method */
//static void heap_find_nearest_inner(BVHNearestData *data, HeapSimple *heap, BVHNode *node)
//{
//  if (node->totnode == 0) {
//    if (data->callback) {
//      data->callback(data->userdata, node->index, data->co, &data->nearest);
//    }
//    else {
//      data->nearest.index = node->index;
//      data->nearest.dist_sq = calc_nearest_point_squared(data->proj, node, data->nearest.co);
//    }
//  }
//  else {
//    float nearest[3];
//
//    for (int i = 0; i != node->totnode; i++) {
//      float dist_sq = calc_nearest_point_squared(data->proj, node->children[i], nearest);
//
//      //if (dist_sq < data->nearest.dist_sq) {
//      //  BLI_heapsimple_insert(heap, dist_sq, node->children[i]);
//      //}
//    }
//  }
//}

static void heap_find_nearest_begin(const BVHNearestData *data, BVHNode *root)
{
  float nearest[3];
  float dist_sq = calc_nearest_point_squared(data->proj, root, nearest);

  if (dist_sq < data->nearest.dist_sq) {
    //HeapSimple *heap = BLI_heapsimple_new_ex(32);

    //heap_find_nearest_inner(data, heap, root);

    //while (!BLI_heapsimple_is_empty(heap) && BLI_heapsimple_top_value(heap) < data->nearest.dist_sq) 
    //{
    //    BVHNode* node = (BVHNode*)BLI_heapsimple_pop_min(heap);
    //    heap_find_nearest_inner(data, heap, node);
    //}

    //BLI_heapsimple_free(heap, NULL);
  }
}

int BLI_bvhtree_find_nearest_ex(const BVHTree *tree,
                                const float co[3],
                                BVHTreeNearest *nearest,
                                const BVHTree_NearestPointCallback callback,
                                void *userdata,
                                const int flag)
{
  axis_t axis_iter;

  BVHNearestData data;
  BVHNode *root = tree->nodes[tree->totleaf];

  /* init data to search */
  data.tree = tree;
  data.co = co;

  data.callback = callback;
  data.userdata = userdata;

  for (axis_iter = data.tree->start_axis; axis_iter != data.tree->stop_axis; axis_iter++) 
  {
#ifdef __CUDA_ARCH__
      data.proj[axis_iter] = dot_v3v3(data.co, d_bvhtree_kdop_axes[axis_iter]);
#else
      data.proj[axis_iter] = dot_v3v3(data.co, bvhtree_kdop_axes[axis_iter]);
#endif
    
  }

  if (nearest) {
    memcpy(&data.nearest, nearest, sizeof(*nearest));
  }
  else {
    data.nearest.index = -1;
    data.nearest.dist_sq = FLT_MAX;
  }

  /* dfs search */
  if (root) {
    if (flag & BVH_NEAREST_OPTIMAL_ORDER) {
      heap_find_nearest_begin(&data, root);
    }
    else {
      dfs_find_nearest_begin(&data, root);
    }
  }

  /* copy back results */
  if (nearest) {
    memcpy(nearest, &data.nearest, sizeof(*nearest));
  }

  return data.nearest.index;
}

int BLI_bvhtree_find_nearest(BVHTree *tree,
                             const float co[3],
                             BVHTreeNearest *nearest,
                             const BVHTree_NearestPointCallback callback,
                             void *userdata)
{
  return BLI_bvhtree_find_nearest_ex(tree, co, nearest, callback, userdata, 0);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name BLI_bvhtree_find_nearest_first
 * \{ */

static bool isect_aabb_v3(const BVHNode *node, const float co[3])
{
  const BVHTreeAxisRange *bv = (const BVHTreeAxisRange *)node->bv;

  if (co[0] > bv[0].min && co[0] < bv[0].max && co[1] > bv[1].min && co[1] < bv[1].max &&
      co[2] > bv[2].min && co[2] < bv[2].max) {
    return true;
  }

  return false;
}

static bool dfs_find_duplicate_fast_dfs(BVHNearestData *data, BVHNode *node)
{
  if (node->totnode == 0) {
    if (isect_aabb_v3(node, data->co)) {
      if (data->callback) {
        const float dist_sq = data->nearest.dist_sq;
        data->callback(data->userdata, node->index, data->co, &data->nearest);
        return (data->nearest.dist_sq < dist_sq);
      }
      data->nearest.index = node->index;
      return true;
    }
  }
  else {
    /* Better heuristic to pick the closest node to dive on */
    int i;

    if (data->proj[node->main_axis] <= node->children[0]->bv[node->main_axis * 2 + 1]) {
      for (i = 0; i != node->totnode; i++) {
        if (isect_aabb_v3(node->children[i], data->co)) {
          if (dfs_find_duplicate_fast_dfs(data, node->children[i])) {
            return true;
          }
        }
      }
    }
    else {
      for (i = node->totnode; i--;) {
        if (isect_aabb_v3(node->children[i], data->co)) {
          if (dfs_find_duplicate_fast_dfs(data, node->children[i])) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

/**
 * Find the first node nearby.
 * Favors speed over quality since it doesn't find the best target node.
 */
int BLI_bvhtree_find_nearest_first(const BVHTree *tree,
                                   const float co[3],
                                   const float dist_sq,
                                   const BVHTree_NearestPointCallback callback,
                                   void *userdata)
{
  BVHNearestData data;
  BVHNode *root = tree->nodes[tree->totleaf];

  /* init data to search */
  data.tree = tree;
  data.co = co;

  data.callback = callback;
  data.userdata = userdata;
  data.nearest.index = -1;
  data.nearest.dist_sq = dist_sq;

  /* dfs search */
  if (root) {
    dfs_find_duplicate_fast_dfs(&data, root);
  }

  return data.nearest.index;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name BLI_bvhtree_ray_cast
 *
 * raycast is done by performing a DFS on the BVHTree and saving the closest hit.
 *
 * \{ */

/* Determines the distance that the ray must travel to hit the bounding volume of the given node */
static float ray_nearest_hit(const BVHRayCastData *data, const float bv[6])
{
  int i;

  float low = 0, upper = data->hit.dist;

  for (i = 0; i != 3; i++, bv += 2) {
    if (data->ray_dot_axis[i] == 0.0f) {
      /* axis aligned ray */
      if (data->ray.origin[i] < bv[0] - data->ray.radius ||
          data->ray.origin[i] > bv[1] + data->ray.radius) {
        return FLT_MAX;
      }
    }
    else {
      float ll = (bv[0] - data->ray.radius - data->ray.origin[i]) / data->ray_dot_axis[i];
      float lu = (bv[1] + data->ray.radius - data->ray.origin[i]) / data->ray_dot_axis[i];

      if (data->ray_dot_axis[i] > 0.0f) {
        if (ll > low) {
          low = ll;
        }
        if (lu < upper) {
          upper = lu;
        }
      }
      else {
        if (lu > low) {
          low = lu;
        }
        if (ll < upper) {
          upper = ll;
        }
      }

      if (low > upper) {
        return FLT_MAX;
      }
    }
  }
  return low;
}

/**
 * Determines the distance that the ray must travel to hit the bounding volume of the given node
 * Based on Tactical Optimization of Ray/Box Intersection, by Graham Fyffe
 * [http://tog.acm.org/resources/RTNews/html/rtnv21n1.html#art9]
 *
 * TODO this doesn't take data->ray.radius into consideration */
static float fast_ray_nearest_hit(const BVHRayCastData *data, const BVHNode *node)
{
  const float *bv = node->bv;

  float t1x = (bv[data->index[0]] - data->ray.origin[0]) * data->idot_axis[0];
  float t2x = (bv[data->index[1]] - data->ray.origin[0]) * data->idot_axis[0];
  float t1y = (bv[data->index[2]] - data->ray.origin[1]) * data->idot_axis[1];
  float t2y = (bv[data->index[3]] - data->ray.origin[1]) * data->idot_axis[1];
  float t1z = (bv[data->index[4]] - data->ray.origin[2]) * data->idot_axis[2];
  float t2z = (bv[data->index[5]] - data->ray.origin[2]) * data->idot_axis[2];

  if ((t1x > t2y || t2x < t1y || t1x > t2z || t2x < t1z || t1y > t2z || t2y < t1z) ||
      (t2x < 0.0f || t2y < 0.0f || t2z < 0.0f) ||
      (t1x > data->hit.dist || t1y > data->hit.dist || t1z > data->hit.dist)) {
    return FLT_MAX;
  }
  return max_fff(t1x, t1y, t1z);
}

static void dfs_raycast(BVHRayCastData *data, const BVHNode *node)
{
  int i;

  /* ray-bv is really fast.. and simple tests revealed its worth to test it
   * before calling the ray-primitive functions */
  /* XXX: temporary solution for particles until fast_ray_nearest_hit supports ray.radius */
  float dist = (data->ray.radius == 0.0f) ? fast_ray_nearest_hit(data, node) :
                                            ray_nearest_hit(data, node->bv);
  if (dist >= data->hit.dist) {
    return;
  }

  if (node->totnode == 0) {
    if (data->callback) {
      data->callback(data->userdata, node->index, &data->ray, &data->hit);
    }
    else {
      data->hit.index = node->index;
      data->hit.dist = dist;
      madd_v3_v3v3fl(data->hit.co, data->ray.origin, data->ray.direction, dist);
    }
  }
  else {
    /* pick loop direction to dive into the tree (based on ray direction and split axis) */
    if (data->ray_dot_axis[node->main_axis] > 0.0f) {
      for (i = 0; i != node->totnode; i++) {
        dfs_raycast(data, node->children[i]);
      }
    }
    else {
      for (i = node->totnode - 1; i >= 0; i--) {
        dfs_raycast(data, node->children[i]);
      }
    }
  }
}

/**
 * A version of #dfs_raycast with minor changes to reset the index & dist each ray cast.
 */
static void dfs_raycast_all(BVHRayCastData *data, const BVHNode *node)
{
  int i;

  /* ray-bv is really fast.. and simple tests revealed its worth to test it
   * before calling the ray-primitive functions */
  /* XXX: temporary solution for particles until fast_ray_nearest_hit supports ray.radius */
  float dist = (data->ray.radius == 0.0f) ? fast_ray_nearest_hit(data, node) :
                                            ray_nearest_hit(data, node->bv);
  if (dist >= data->hit.dist) {
    return;
  }

  if (node->totnode == 0) {
    /* no need to check for 'data->callback' (using 'all' only makes sense with a callback). */
    dist = data->hit.dist;
    data->callback(data->userdata, node->index, &data->ray, &data->hit);
    data->hit.index = -1;
    data->hit.dist = dist;
  }
  else {
    /* pick loop direction to dive into the tree (based on ray direction and split axis) */
    if (data->ray_dot_axis[node->main_axis] > 0.0f) {
      for (i = 0; i != node->totnode; i++) {
        dfs_raycast_all(data, node->children[i]);
      }
    }
    else {
      for (i = node->totnode - 1; i >= 0; i--) {
        dfs_raycast_all(data, node->children[i]);
      }
    }
  }
}

static void bvhtree_ray_cast_data_precalc(BVHRayCastData *data, int flag)
{
  int i;

  for (i = 0; i < 3; i++) {
#ifdef __CUDA_ARCH__
      data->ray_dot_axis[i] = dot_v3v3(data->ray.direction, d_bvhtree_kdop_axes[i]);
#else
      data->ray_dot_axis[i] = dot_v3v3(data->ray.direction, bvhtree_kdop_axes[i]);
#endif

    if (fabsf(data->ray_dot_axis[i]) < FLT_EPSILON) {
      data->ray_dot_axis[i] = 0.0f;
      /* Sign is not important in this case, `data->index` is adjusted anyway. */
      data->idot_axis[i] = FLT_MAX;
    }
    else {
      data->idot_axis[i] = 1.0f / data->ray_dot_axis[i];
    }

    data->index[2 * i] = data->idot_axis[i] < 0.0f ? 1 : 0;
    data->index[2 * i + 1] = 1 - data->index[2 * i];
    data->index[2 * i] += 2 * i;
    data->index[2 * i + 1] += 2 * i;
  }

#ifdef USE_KDOPBVH_WATERTIGHT
  if (flag & BVH_RAYCAST_WATERTIGHT) {
    isect_ray_tri_watertight_v3_precalc(&data->isect_precalc, data->ray.direction);
    data->ray.isect_precalc = &data->isect_precalc;
  }
  else {
    data->ray.isect_precalc = nullptr;
  }
#else
  UNUSED_VARS(flag);
#endif
}

int BLI_bvhtree_ray_cast_ex(const BVHTree *tree,
                            const float co[3],
                            const float dir[3],
                            const float radius,
                            BVHTreeRayHit *hit,
                            const BVHTree_RayCastCallback callback,
                            void *userdata,
                            const int flag)
{
  BVHRayCastData data;
  BVHNode *root = tree->nodes[tree->totleaf];

  BLI_ASSERT_UNIT_V3(dir);

  data.tree = tree;

  data.callback = callback;
  data.userdata = userdata;

  copy_v3_v3(data.ray.origin, co);
  copy_v3_v3(data.ray.direction, dir);
  data.ray.radius = radius;

  bvhtree_ray_cast_data_precalc(&data, flag);

  if (hit) {
    memcpy(&data.hit, hit, sizeof(*hit));
  }
  else {
    data.hit.index = -1;
    data.hit.dist = BVH_RAYCAST_DIST_MAX;
  }

  if (root) {
    dfs_raycast(&data, root);
    //      iterative_raycast(&data, root);
  }

  if (hit) {
    memcpy(hit, &data.hit, sizeof(*hit));
  }

  return data.hit.index;
}

int BLI_bvhtree_ray_cast(BVHTree *tree,
                         const float co[3],
                         const float dir[3],
                         const float radius,
                         BVHTreeRayHit *hit,
                         const BVHTree_RayCastCallback callback,
                         void *userdata)
{
  return BLI_bvhtree_ray_cast_ex(
      tree, co, dir, radius, hit, callback, userdata, BVH_RAYCAST_DEFAULT);
}

float BLI_bvhtree_bb_raycast(const float bv[6],
                             const float light_start[3],
                             const float light_end[3],
                             float pos[3])
{
  BVHRayCastData data;
  float dist;

  data.hit.dist = BVH_RAYCAST_DIST_MAX;

  /* get light direction */
  sub_v3_v3v3(data.ray.direction, light_end, light_start);

  data.ray.radius = 0.0;

  copy_v3_v3(data.ray.origin, light_start);

  normalize_v3(data.ray.direction);
  copy_v3_v3(data.ray_dot_axis, data.ray.direction);

  dist = ray_nearest_hit(&data, bv);

  madd_v3_v3v3fl(pos, light_start, data.ray.direction, dist);

  return dist;
}

/**
 * Calls the callback for every ray intersection
 *
 * \note Using a \a callback which resets or never sets the #BVHTreeRayHit index & dist works too,
 * however using this function means existing generic callbacks can be used from custom callbacks
 * without having to handle resetting the hit beforehand.
 * It also avoid redundant argument and return value which aren't meaningful
 * when collecting multiple hits.
 */
void BLI_bvhtree_ray_cast_all_ex(const BVHTree *tree,
                                 const float co[3],
                                 const float dir[3],
                                 const float radius,
                                 const float hit_dist,
                                 const BVHTree_RayCastCallback callback,
                                 void *userdata,
                                 const int flag)
{
  BVHRayCastData data;
  BVHNode *root = tree->nodes[tree->totleaf];

  BLI_ASSERT_UNIT_V3(dir);
  BLI_assert(callback != NULL);

  data.tree = tree;

  data.callback = callback;
  data.userdata = userdata;

  copy_v3_v3(data.ray.origin, co);
  copy_v3_v3(data.ray.direction, dir);
  data.ray.radius = radius;

  bvhtree_ray_cast_data_precalc(&data, flag);

  data.hit.index = -1;
  data.hit.dist = hit_dist;

  if (root) {
    dfs_raycast_all(&data, root);
  }
}

void BLI_bvhtree_ray_cast_all(BVHTree *tree,
                              const float co[3],
                              const float dir[3],
                              const float radius,
                              const float hit_dist,
                              const BVHTree_RayCastCallback callback,
                              void *userdata)
{
  BLI_bvhtree_ray_cast_all_ex(
      tree, co, dir, radius, hit_dist, callback, userdata, BVH_RAYCAST_DEFAULT);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name BLI_bvhtree_range_query
 *
 * Allocates and fills an array with the indices of node that are on the given spherical range
 * (center, radius).
 * Returns the size of the array.
 *
 * \{ */

typedef struct RangeQueryData {
  BVHTree *tree;
  const float *center;
  float radius_sq; /* squared radius */

  int hits;

  BVHTree_RangeQuery callback;
  void *userdata;
} RangeQueryData;

static void dfs_range_query(RangeQueryData *data, BVHNode *node)
{
  if (node->totnode == 0) {
#if 0 /*UNUSED*/
    /* Calculate the node min-coords
     * (if the node was a point then this is the point coordinates) */
    float co[3];
    co[0] = node->bv[0];
    co[1] = node->bv[2];
    co[2] = node->bv[4];
#endif
  }
  else {
    int i;
    for (i = 0; i != node->totnode; i++) {
      float nearest[3];
      float dist_sq = calc_nearest_point_squared(data->center, node->children[i], nearest);
      if (dist_sq < data->radius_sq) {
        /* Its a leaf.. call the callback */
        if (node->children[i]->totnode == 0) {
          data->hits++;
          data->callback(data->userdata, node->children[i]->index, data->center, dist_sq);
        }
        else {
          dfs_range_query(data, node->children[i]);
        }
      }
    }
  }
}

int BLI_bvhtree_range_query(
    BVHTree *tree, const float co[3], const float radius, const BVHTree_RangeQuery callback, void *userdata)
{
  BVHNode *root = tree->nodes[tree->totleaf];

  RangeQueryData data;
  data.tree = tree;
  data.center = co;
  data.radius_sq = radius * radius;
  data.hits = 0;

  data.callback = callback;
  data.userdata = userdata;

  if (root != nullptr) {
    float nearest[3];
    float dist_sq = calc_nearest_point_squared(data.center, root, nearest);
    if (dist_sq < data.radius_sq) {
      /* Its a leaf.. call the callback */
      if (root->totnode == 0) {
        data.hits++;
        data.callback(data.userdata, root->index, co, dist_sq);
      }
      else {
        dfs_range_query(&data, root);
      }
    }
  }

  return data.hits;
}

/** \} */
//
///* -------------------------------------------------------------------- */
///** \name BLI_bvhtree_nearest_projected
// * \{ */
//
//static void bvhtree_nearest_projected_dfs_recursive(BVHNearestProjectedData *__restrict data,
//                                                    const BVHNode *node)
//{
//  if (node->totnode == 0) {
//    if (data->callback) {
//      data->callback(data->userdata, node->index, &data->precalc, nullptr, 0, &data->nearest);
//    }
//    else {
//      data->nearest.index = node->index;
//      data->nearest.dist_sq = dist_squared_to_projected_aabb(
//          &data->precalc,
//          (float[3])(node->bv[0], node->bv[2], node->bv[4]),
//          (float[3])(node->bv[1], node->bv[3], node->bv[5]),
//          data->closest_axis);
//    }
//  }
//  else {
//    /* First pick the closest node to recurse into */
//    if (data->closest_axis[node->main_axis]) {
//      for (int i = 0; i != node->totnode; i++) {
//        const float *bv = node->children[i]->bv;
//
//        if (dist_squared_to_projected_aabb(&data->precalc,
//                                           (float[3])(bv[0], bv[2], bv[4]),
//                                           (float[3])(bv[1], bv[3], bv[5]),
//                                           data->closest_axis) <= data->nearest.dist_sq) {
//          bvhtree_nearest_projected_dfs_recursive(data, node->children[i]);
//        }
//      }
//    }
//    else {
//      for (int i = node->totnode; i--;) {
//        const float *bv = node->children[i]->bv;
//
//        if (dist_squared_to_projected_aabb(&data->precalc,
//                                           (float[3])(bv[0], bv[2], bv[4]),
//                                           (float[3])(bv[1], bv[3], bv[5]),
//                                           data->closest_axis) <= data->nearest.dist_sq) {
//          bvhtree_nearest_projected_dfs_recursive(data, node->children[i]);
//        }
//      }
//    }
//  }
//}
//
//static void bvhtree_nearest_projected_with_clipplane_test_dfs_recursive(
//    BVHNearestProjectedData *__restrict data, const BVHNode *node)
//{
//  if (node->totnode == 0) {
//    if (data->callback) {
//      data->callback(data->userdata,
//                     node->index,
//                     &data->precalc,
//                     data->clip_plane,
//                     data->clip_plane_len,
//                     &data->nearest);
//    }
//    else {
//      data->nearest.index = node->index;
//      data->nearest.dist_sq = dist_squared_to_projected_aabb(
//          &data->precalc,
//          (float[3])(node->bv[0], node->bv[2], node->bv[4]),
//          (float[3])(node->bv[1], node->bv[3], node->bv[5]),
//          data->closest_axis);
//    }
//  }
//  else {
//    /* First pick the closest node to recurse into */
//    if (data->closest_axis[node->main_axis]) {
//      for (int i = 0; i != node->totnode; i++) {
//        const float *bv = node->children[i]->bv;
//        const float bb_min[3] = {bv[0], bv[2], bv[4]};
//        const float bb_max[3] = {bv[1], bv[3], bv[5]};
//
//        int isect_type = isect_aabb_planes_v3(
//            data->clip_plane, data->clip_plane_len, bb_min, bb_max);
//
//        if ((isect_type != ISECT_AABB_PLANE_BEHIND_ANY) &&
//            dist_squared_to_projected_aabb(&data->precalc, bb_min, bb_max, data->closest_axis) <=
//                data->nearest.dist_sq) {
//          if (isect_type == ISECT_AABB_PLANE_CROSS_ANY) {
//            bvhtree_nearest_projected_with_clipplane_test_dfs_recursive(data, node->children[i]);
//          }
//          else {
//            /* ISECT_AABB_PLANE_IN_FRONT_ALL */
//            bvhtree_nearest_projected_dfs_recursive(data, node->children[i]);
//          }
//        }
//      }
//    }
//    else {
//      for (int i = node->totnode; i--;) {
//        const float *bv = node->children[i]->bv;
//        const float bb_min[3] = {bv[0], bv[2], bv[4]};
//        const float bb_max[3] = {bv[1], bv[3], bv[5]};
//
//        int isect_type = isect_aabb_planes_v3(
//            data->clip_plane, data->clip_plane_len, bb_min, bb_max);
//
//        if (isect_type != ISECT_AABB_PLANE_BEHIND_ANY &&
//            dist_squared_to_projected_aabb(&data->precalc, bb_min, bb_max, data->closest_axis) <=
//                data->nearest.dist_sq) {
//          if (isect_type == ISECT_AABB_PLANE_CROSS_ANY) {
//            bvhtree_nearest_projected_with_clipplane_test_dfs_recursive(data, node->children[i]);
//          }
//          else {
//            /* ISECT_AABB_PLANE_IN_FRONT_ALL */
//            bvhtree_nearest_projected_dfs_recursive(data, node->children[i]);
//          }
//        }
//      }
//    }
//  }
//}
//
//int BLI_bvhtree_find_nearest_projected(BVHTree *tree,
//                                       float projmat[4][4],
//                                       float winsize[2],
//                                       float mval[2],
//                                       float clip_plane[6][4],
//                                       int clip_plane_len,
//                                       BVHTreeNearest *nearest,
//                                       BVHTree_NearestProjectedCallback callback,
//                                       void *userdata)
//{
//  BVHNode *root = tree->nodes[tree->totleaf];
//  if (root != nullptr) {
//    BVHNearestProjectedData data;
//    dist_squared_to_projected_aabb_precalc(&data.precalc, projmat, winsize, mval);
//
//    data.callback = callback;
//    data.userdata = userdata;
//
//    if (clip_plane) {
//      data.clip_plane_len = clip_plane_len;
//      for (int i = 0; i < data.clip_plane_len; i++) {
//        copy_v4_v4(data.clip_plane[i], clip_plane[i]);
//      }
//    }
//    else {
//      data.clip_plane_len = 1;
//      planes_from_projmat(projmat, nullptr, nullptr, nullptr, nullptr, data.clip_plane[0], nullptr);
//    }
//
//    if (nearest) {
//      memcpy(&data.nearest, nearest, sizeof(*nearest));
//    }
//    else {
//      data.nearest.index = -1;
//      data.nearest.dist_sq = FLT_MAX;
//    }
//    {
//      const float bb_min[3] = {root->bv[0], root->bv[2], root->bv[4]};
//      const float bb_max[3] = {root->bv[1], root->bv[3], root->bv[5]};
//
//      int isect_type = isect_aabb_planes_v3(data.clip_plane, data.clip_plane_len, bb_min, bb_max);
//
//      if (isect_type != 0 &&
//          dist_squared_to_projected_aabb(&data.precalc, bb_min, bb_max, data.closest_axis) <=
//              data.nearest.dist_sq) {
//        if (isect_type == 1) {
//          bvhtree_nearest_projected_with_clipplane_test_dfs_recursive(&data, root);
//        }
//        else {
//          bvhtree_nearest_projected_dfs_recursive(&data, root);
//        }
//      }
//    }
//
//    if (nearest) {
//      memcpy(nearest, &data.nearest, sizeof(*nearest));
//    }
//
//    return data.nearest.index;
//  }
//  return -1;
//}
//
///** \} */

/* -------------------------------------------------------------------- */
/** \name BLI_bvhtree_walk_dfs
 * \{ */

typedef struct BVHTree_WalkData {
  BVHTree_WalkParentCallback walk_parent_cb;
  BVHTree_WalkLeafCallback walk_leaf_cb;
  BVHTree_WalkOrderCallback walk_order_cb;
  void *userdata;
} BVHTree_WalkData;

/**
 * Runs first among nodes children of the first node before going
 * to the next node in the same layer.
 *
 * \return false to break out of the search early.
 */
static bool bvhtree_walk_dfs_recursive(BVHTree_WalkData *walk_data, const BVHNode *node)
{
  if (node->totnode == 0) {
    return walk_data->walk_leaf_cb(
        (const BVHTreeAxisRange *)node->bv, node->index, walk_data->userdata);
  }

  /* First pick the closest node to recurse into */
  if (walk_data->walk_order_cb(
          (const BVHTreeAxisRange *)node->bv, node->main_axis, walk_data->userdata)) {
    for (int i = 0; i != node->totnode; i++) {
      if (walk_data->walk_parent_cb((const BVHTreeAxisRange *)node->children[i]->bv,
                                    walk_data->userdata)) {
        if (!bvhtree_walk_dfs_recursive(walk_data, node->children[i])) {
          return false;
        }
      }
    }
  }
  else {
    for (int i = node->totnode - 1; i >= 0; i--) {
      if (walk_data->walk_parent_cb((const BVHTreeAxisRange *)node->children[i]->bv,
                                    walk_data->userdata)) {
        if (!bvhtree_walk_dfs_recursive(walk_data, node->children[i])) {
          return false;
        }
      }
    }
  }
  return true;
}

/**
 * This is a generic function to perform a depth first search on the #BVHTree
 * where the search order and nodes traversed depend on callbacks passed in.
 *
 * \param tree: Tree to walk.
 * \param walk_parent_cb: Callback on a parents bound-box to test if it should be traversed.
 * \param walk_leaf_cb: Callback to test leaf nodes, callback must store its own result,
 * returning false exits early.
 * \param walk_order_cb: Callback that indicates which direction to search,
 * either from the node with the lower or higher K-DOP axis value.
 * \param userdata: Argument passed to all callbacks.
 */
void BLI_bvhtree_walk_dfs(const BVHTree *tree,
                          const BVHTree_WalkParentCallback walk_parent_cb,
                          const BVHTree_WalkLeafCallback walk_leaf_cb,
                          const BVHTree_WalkOrderCallback walk_order_cb,
                          void *userdata)
{
  const BVHNode *root = tree->nodes[tree->totleaf];
  if (root != nullptr) {
    BVHTree_WalkData walk_data = {walk_parent_cb, walk_leaf_cb, walk_order_cb, userdata};
    /* first make sure the bv of root passes in the test too */
    if (walk_parent_cb((const BVHTreeAxisRange *)root->bv, userdata)) {
      bvhtree_walk_dfs_recursive(&walk_data, root);
    }
  }
}

/** \} */
