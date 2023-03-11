#pragma once

#include <stdio.h>


struct Depsgraph;
struct Scene;
struct ViewLayer;

/* ------------------------------------------------ */

/* NOTE: Those flags are same bitmask as G.debug_flags */

void DEG_debug_flags_set(struct Depsgraph *depsgraph, int flags);
int DEG_debug_flags_get(const struct Depsgraph *depsgraph);

void DEG_debug_name_set(struct Depsgraph *depsgraph, const char *name);
const char *DEG_debug_name_get(struct Depsgraph *depsgraph);

/* ------------------------------------------------ */

void DEG_stats_simple(const struct Depsgraph *graph,
                      size_t *r_outer,
                      size_t *r_operations,
                      size_t *r_relations);

/* ************************************************ */
/* Diagram-Based Graph Debugging */

void DEG_debug_relations_graphviz(const struct Depsgraph *graph, FILE *fp, const char *label);

void DEG_debug_stats_gnuplot(const struct Depsgraph *graph,
                             FILE *fp,
                             const char *label,
                             const char *output_filename);

/* ************************************************ */

/* Compare two dependency graphs. */
bool DEG_debug_compare(const struct Depsgraph *graph1, const struct Depsgraph *graph2);

/* Check that dependencies in the graph are really up to date. */
bool DEG_debug_graph_relations_validate(struct Depsgraph *graph,
                                        struct Main *bmain,
                                        struct Scene *scene,
                                        struct ViewLayer *view_layer);

/* Perform consistency check on the graph. */
bool DEG_debug_consistency_check(struct Depsgraph *graph);


