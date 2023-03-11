#include <cstdarg>

#include "BLI_dot_export.hh"
#include "utildefines.h"

#include "listBase.h"

#include "DEG_depsgraph.h"
#include "DEG_depsgraph_debug.h"

#include "depsgraph.h"

#include "node/deg_node_component.h"
#include "node/deg_node_id.h"
#include "node/deg_node_operation.h"
#include "node/deg_node_time.h"

namespace deg = blender::deg;
namespace dot = blender::dot;

/* ****************** */
/* Graphviz Debugging */

namespace blender::deg {

/* Only one should be enabled, defines whether graphviz nodes
 * get colored by individual types or classes.
 */
#define COLOR_SCHEME_NODE_CLASS 1
//#define COLOR_SCHEME_NODE_TYPE  2

static const char *deg_debug_graphviz_fontname = "helvetica";
static float deg_debug_graphviz_graph_label_size = 20.0f;
static float deg_debug_graphviz_node_label_size = 14.0f;
static const int deg_debug_max_colors = 12;
#ifdef COLOR_SCHEME_NODE_TYPE
static const char *deg_debug_colors[] = {
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928",
    "#ff00ff",
};
#endif
static const char *deg_debug_colors_light[] = {
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
    "#ff00ff",
};

#ifdef COLOR_SCHEME_NODE_TYPE
static const int deg_debug_node_type_color_map[][2] = {
    {NodeType::TIMESOURCE, 0},
    {NodeType::ID_REF, 1},

    /* Outer Types */
    {NodeType::PARAMETERS, 2},
    {NodeType::PROXY, 3},
    {NodeType::ANIMATION, 4},
    {NodeType::TRANSFORM, 5},
    {NodeType::GEOMETRY, 6},
    {NodeType::SEQUENCER, 7},
    {NodeType::SHADING, 8},
    {NodeType::SHADING_PARAMETERS, 9},
    {NodeType::CACHE, 10},
    {NodeType::POINT_CACHE, 11},
    {NodeType::LAYER_COLLECTIONS, 12},
    {NodeType::COPY_ON_WRITE, 13},
    {-1, 0},
};
#endif

static int deg_debug_node_color_index(const Node *node)
{
#ifdef COLOR_SCHEME_NODE_CLASS
  /* Some special types. */
  switch (node->type) {
    case NodeType::ID_REF:
      return 5;
    case NodeType::OPERATION: {
      OperationNode *op_node = (OperationNode *)node;
      if (op_node->is_noop()) {
        if (op_node->flag & OperationFlag::DEPSOP_FLAG_PINNED) {
          return 7;
        }
        return 8;
      }
      break;
    }

    default:
      break;
  }
  /* Do others based on class. */
  switch (node->get_class()) {
    case NodeClass::OPERATION:
      return 4;
    case NodeClass::COMPONENT:
      return 1;
    default:
      return 9;
  }
#endif

#ifdef COLOR_SCHEME_NODE_TYPE
  const int(*pair)[2];
  for (pair = deg_debug_node_type_color_map; (*pair)[0] >= 0; pair++) {
    if ((*pair)[0] == node->type) {
      return (*pair)[1];
    }
  }
  return -1;
#endif
}

struct DotExportContext {
  bool show_tags;
  dot::DirectedGraph &digraph;
  Map<const Node *, dot::Node *> nodes_map;
  Map<const Node *, dot::Cluster *> clusters_map;
};

static void deg_debug_graphviz_legend_color(const char *name,
                                            const char *color,
                                            std::stringstream &ss)
{

  ss << "<TR>";
  ss << "<TD>" << name << "</TD>";
  ss << "<TD BGCOLOR=\"" << color << "\"></TD>";
  ss << "</TR>";
}

static void deg_debug_graphviz_legend(DotExportContext &ctx)
{
  dot::Node &legend_node = ctx.digraph.new_node("");
  legend_node.attributes.set("rank", "sink");
  legend_node.attributes.set("shape", "none");
  legend_node.attributes.set("margin", 0);

  std::stringstream ss;
  ss << "<";
  ss << R"(<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">)";
  ss << "<TR><TD COLSPAN=\"2\"><B>Legend</B></TD></TR>";

#ifdef COLOR_SCHEME_NODE_CLASS
  const char **colors = deg_debug_colors_light;
  deg_debug_graphviz_legend_color("Operation", colors[4], ss);
  deg_debug_graphviz_legend_color("Component", colors[1], ss);
  deg_debug_graphviz_legend_color("ID Node", colors[5], ss);
  deg_debug_graphviz_legend_color("NOOP", colors[8], ss);
  deg_debug_graphviz_legend_color("Pinned OP", colors[7], ss);
#endif

#ifdef COLOR_SCHEME_NODE_TYPE
  const int(*pair)[2];
  for (pair = deg_debug_node_type_color_map; (*pair)[0] >= 0; pair++) {
    DepsNodeFactory *nti = type_get_factory((NodeType)(*pair)[0]);
    deg_debug_graphviz_legend_color(
        ctx, nti->tname().c_str(), deg_debug_colors_light[(*pair)[1] % deg_debug_max_colors], ss);
  }
#endif

  ss << "</TABLE>";
  ss << ">";
  legend_node.attributes.set("label", ss.str());
  legend_node.attributes.set("fontname", deg_debug_graphviz_fontname);
}

static void deg_debug_graphviz_node_color(DotExportContext &ctx,
                                          const Node *node,
                                          dot::Attributes &dot_attributes)
{
  const char *color_default = "black";
  const char *color_modified = "orangered4";
  const char *color_update = "dodgerblue3";
  const char *color = color_default;
  if (ctx.show_tags) {
    if (node->get_class() == NodeClass::OPERATION) {
      OperationNode *op_node = (OperationNode *)node;
      if (op_node->flag & DEPSOP_FLAG_DIRECTLY_MODIFIED) {
        color = color_modified;
      }
      else if (op_node->flag & DEPSOP_FLAG_NEEDS_UPDATE) {
        color = color_update;
      }
    }
  }
  dot_attributes.set("color", color);
}

static void deg_debug_graphviz_node_penwidth(DotExportContext &ctx,
                                             const Node *node,
                                             dot::Attributes &dot_attributes)
{
  float penwidth_default = 1.0f;
  float penwidth_modified = 4.0f;
  float penwidth_update = 4.0f;
  float penwidth = penwidth_default;
  if (ctx.show_tags) {
    if (node->get_class() == NodeClass::OPERATION) {
      OperationNode *op_node = (OperationNode *)node;
      if (op_node->flag & DEPSOP_FLAG_DIRECTLY_MODIFIED) {
        penwidth = penwidth_modified;
      }
      else if (op_node->flag & DEPSOP_FLAG_NEEDS_UPDATE) {
        penwidth = penwidth_update;
      }
    }
  }
  dot_attributes.set("penwidth", penwidth);
}

static void deg_debug_graphviz_node_fillcolor(const Node *node, dot::Attributes &dot_attributes)
{
  const char *defaultcolor = "gainsboro";
  int color_index = deg_debug_node_color_index(node);
  const char *fillcolor = color_index < 0 ?
                              defaultcolor :
                              deg_debug_colors_light[color_index % deg_debug_max_colors];
  dot_attributes.set("fillcolor", fillcolor);
}

static void deg_debug_graphviz_node_style(DotExportContext &ctx,
                                          const Node *node,
                                          dot::Attributes &dot_attributes)
{
  StringRef base_style = "filled"; /* default style */
  if (ctx.show_tags) {
    if (node->get_class() == NodeClass::OPERATION) {
      OperationNode *op_node = (OperationNode *)node;
      if (op_node->flag & (DEPSOP_FLAG_DIRECTLY_MODIFIED | DEPSOP_FLAG_NEEDS_UPDATE)) {
        base_style = "striped";
      }
    }
  }
  switch (node->get_class()) {
    case NodeClass::GENERIC:
      dot_attributes.set("style", base_style);
      break;
    case NodeClass::COMPONENT:
      dot_attributes.set("style", base_style);
      break;
    case NodeClass::OPERATION:
      dot_attributes.set("style", base_style + ",rounded");
      break;
  }
}

static void deg_debug_graphviz_graph_nodes(DotExportContext &ctx, const Depsgraph *graph);
static void deg_debug_graphviz_graph_relations(DotExportContext &ctx, const Depsgraph *graph);

}  // namespace blender::deg

void DEG_debug_relations_graphviz(const Depsgraph *graph, FILE *fp, const char *label)
{
  if (!graph) {
    return;
  }

  const deg::Depsgraph *deg_graph = reinterpret_cast<const deg::Depsgraph *>(graph);

  dot::DirectedGraph digraph;
  deg::DotExportContext ctx{false, digraph};

  digraph.set_rankdir(dot::Attr_rankdir::LeftToRight);
  digraph.attributes.set("compound", "true");
  digraph.attributes.set("labelloc", "t");
  digraph.attributes.set("fontsize", deg::deg_debug_graphviz_graph_label_size);
  digraph.attributes.set("fontname", deg::deg_debug_graphviz_fontname);
  digraph.attributes.set("label", label);
  digraph.attributes.set("splines", "ortho");
  digraph.attributes.set("overlap", "scalexy");

  deg::deg_debug_graphviz_graph_nodes(ctx, deg_graph);
  deg::deg_debug_graphviz_graph_relations(ctx, deg_graph);

  deg::deg_debug_graphviz_legend(ctx);

  std::string dot_string = digraph.to_dot_string();
  fprintf(fp, "%s", dot_string.c_str());
}
