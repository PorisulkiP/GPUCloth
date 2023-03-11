/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The Original Code is Copyright (C) 2020 Blender Foundation.
 * All rights reserved.
 */

#include "pipeline.h"

#include "scene_types.cuh"

#include "deg_builder_cycle.h"
#include "deg_builder_nodes.h"
#include "deg_builder_relations.h"
#include "deg_builder_transitive.h"

namespace blender::deg {

AbstractBuilderPipeline::AbstractBuilderPipeline(::Depsgraph *graph)
    : deg_graph_(reinterpret_cast<Depsgraph *>(graph)),
      bmain_(deg_graph_->bmain),
      scene_(deg_graph_->scene),
      view_layer_(deg_graph_->view_layer)
{
}

AbstractBuilderPipeline::~AbstractBuilderPipeline()
{
}

void AbstractBuilderPipeline::build()
{
  double start_time = 0.0;
  build_step_sanity_check();
  build_step_nodes();
  build_step_relations();
}

void AbstractBuilderPipeline::build_step_sanity_check()
{
  BLI_assert(BLI_findindex(&scene_->view_layers, view_layer_) != -1);
  BLI_assert(deg_graph_->scene == scene_);
  BLI_assert(deg_graph_->view_layer == view_layer_);
}

void AbstractBuilderPipeline::build_step_nodes()
{
  /* Generate all the nodes in the graph first */
  unique_ptr<DepsgraphNodeBuilder> node_builder = construct_node_builder();
  node_builder->begin_build();
  build_nodes(*node_builder);
  node_builder->end_build();
}

void AbstractBuilderPipeline::build_step_relations()
{
  /* Hook up relationships between operations - to determine evaluation order. */
  unique_ptr<DepsgraphRelationBuilder> relation_builder = construct_relation_builder();
  relation_builder->begin_build();
  build_relations(*relation_builder);
  relation_builder->build_copy_on_write_relations();
  relation_builder->build_driver_relations();
}


unique_ptr<DepsgraphNodeBuilder> AbstractBuilderPipeline::construct_node_builder()
{
  return std::make_unique<DepsgraphNodeBuilder>(bmain_, deg_graph_, &builder_cache_);
}

unique_ptr<DepsgraphRelationBuilder> AbstractBuilderPipeline::construct_relation_builder()
{
  return std::make_unique<DepsgraphRelationBuilder>(bmain_, deg_graph_, &builder_cache_);
}

}  // namespace blender::deg
