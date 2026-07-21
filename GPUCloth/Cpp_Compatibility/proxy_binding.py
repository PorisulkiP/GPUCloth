"""Validation and ownership rules for proxy cloth simulation."""

from __future__ import annotations


class ProxyBindingError(ValueError):
    pass


def expected_grid_vertex_count(cells_x, cells_y, sheet_count):
    cells_x = int(cells_x)
    cells_y = int(cells_y)
    sheet_count = int(sheet_count)
    if cells_x <= 0 or cells_y <= 0 or sheet_count <= 0:
        raise ProxyBindingError(
            "proxy grid cell counts and sheet count must be positive")
    return (cells_x + 1) * (cells_y + 1) * sheet_count


def _mesh_vertex_count(obj, role):
    if obj is None:
        raise ProxyBindingError(f"{role} object is not set")
    if getattr(obj, "type", None) != "MESH":
        raise ProxyBindingError(f"{role} object must be a mesh")
    mesh = getattr(obj, "data", None)
    vertices = getattr(mesh, "vertices", None)
    if vertices is None:
        raise ProxyBindingError(f"{role} object has no mesh vertices")
    return len(vertices)


def validate_proxy_binding(render_obj, settings):
    """Return immutable binding facts or raise before native allocation."""
    if not bool(getattr(settings, "use_proxy", False)):
        render_count = _mesh_vertex_count(render_obj, "render")
        return {
            "render_object": render_obj,
            "simulation_object": render_obj,
            "render_vertex_count": render_count,
            "simulation_vertex_count": render_count,
            "uses_proxy": False,
        }

    proxy_obj = getattr(settings, "proxy_object", None)
    if proxy_obj is render_obj:
        raise ProxyBindingError("proxy object must differ from render object")

    render_count = _mesh_vertex_count(render_obj, "render")
    proxy_count = _mesh_vertex_count(proxy_obj, "proxy")
    sheets = int(getattr(settings, "num_sheets", 0))
    expected_render = expected_grid_vertex_count(
        getattr(settings, "hi_nx", 0), getattr(settings, "hi_ny", 0), sheets)
    expected_proxy = expected_grid_vertex_count(
        getattr(settings, "proxy_nx", 0),
        getattr(settings, "proxy_ny", 0),
        sheets,
    )

    errors = []
    if render_count != expected_render:
        errors.append(
            f"render topology has {render_count} vertices; expected "
            f"{expected_render} from hi_nx/hi_ny/num_sheets")
    if proxy_count != expected_proxy:
        errors.append(
            f"proxy topology has {proxy_count} vertices; expected "
            f"{expected_proxy} from proxy_nx/proxy_ny/num_sheets")
    if errors:
        raise ProxyBindingError("; ".join(errors))

    return {
        "render_object": render_obj,
        "simulation_object": proxy_obj,
        "render_vertex_count": render_count,
        "simulation_vertex_count": proxy_count,
        "uses_proxy": True,
    }
