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
 * The Original Code is Copyright (C) 2013 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup depsgraph
 *
 * Methods for constructing depsgraph relations for drivers.
 */

#include "builder/deg_builder_relations_drivers.h"

#include <cstring>

#include "builder/deg_builder_relations.h"
#include "depsgraph_relation.h"
#include "node/deg_node.h"

namespace blender::deg {

DriverDescriptor::DriverDescriptor(PointerRNA *id_ptr, FCurve *fcu)
    : id_ptr_(id_ptr),
      fcu_(fcu),
      driver_relations_needed_(false),
      property_rna_(nullptr),
      is_array_(false)
{
  driver_relations_needed_ = determine_relations_needed();
  split_rna_path();
}

bool DriverDescriptor::determine_relations_needed()
{
  if (!resolve_rna()) {
    /* Properties that don't exist can't cause threading issues either. */
    return false;
  }
  return true;
}

bool DriverDescriptor::driver_relations_needed() const
{
  return driver_relations_needed_;
}

bool DriverDescriptor::is_array() const
{
  return is_array_;
}

/* Assumes that 'other' comes from the same RNA group, that is, has the same RNA path prefix. */
bool DriverDescriptor::is_same_array_as(const DriverDescriptor &other) const
{
  if (!is_array_ || !other.is_array_) {
    return false;
  }
  return rna_suffix == other.rna_suffix;
}

static bool is_reachable(const Node *const from, const Node *const to)
{
  if (from == to) {
    return true;
  }

  // Perform a graph walk from 'to' towards its incoming connections.
  // Walking from 'from' towards its outgoing connections is 10x slower on the Spring rig.
  deque<const Node *> queue;
  Set<const Node *> seen;
  queue.push_back(to);
  while (!queue.empty()) {
    // Visit the next node to inspect.
    const Node *visit = queue.back();
    queue.pop_back();

    if (visit == from) {
      return true;
    }

    // Queue all incoming relations that we haven't seen before.
    for (Relation *relation : visit->inlinks) {
      const Node *prev_node = relation->from;
      if (seen.add(prev_node)) {
        queue.push_back(prev_node);
      }
    }
  }
  return false;
}

/* **** DepsgraphRelationBuilder functions **** */

void DepsgraphRelationBuilder::build_driver_relations()
{
  for (IDNode *id_node : graph_->id_nodes) {
    build_driver_relations(id_node);
  }
}
}  // namespace blender::deg
