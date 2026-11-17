"""Light cluster assignment and culling for the 2D renderer.

The viewport is split into square tiles ("clusters").  Every light is written
only to the tiles its screen-space bounding box actually overlaps instead of
being replicated across a full row/column, which keeps a large light from
blowing past the per-tile light budget.

A light is any mapping (or object) exposing ``x``/``y``/``w``/``h``.  The
following optional fields are understood:

``rotation`` / ``angle`` (degrees) or ``rotation_rad`` / ``angle_rad``
    Rotates the box around its centre before the tile coverage is computed.
``scale`` (number, pair or ``{"x": ..., "y": ...}``), ``sx``/``sy``
    Scales the box dimensions.
``intensity`` / ``brightness`` and ``distance``
    Used to rank lights per tile; brighter/closer lights survive truncation.
``layers`` / ``layer``
    Restricts a light to a render layer (for example the transparent pass).
    Lights without a layer apply to every layer.
``static`` / ``is_static``
    Marks a light as static so :class:`ClusterCache` can reuse its clusters.

All public helpers are pure: the caller owns the light dictionaries and the
returned cluster dictionary.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Tile = Tuple[int, int]
TileRange = Tuple[int, int, int, int]
Screen = Tuple[float, float]

DEFAULT_MAX_LIGHTS = 8
_EPS = 1e-9
_MISSING = object()


# ---------------------------------------------------------------------------
# Field access helpers
# ---------------------------------------------------------------------------


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    """Return the first non-``None`` field from a mapping or an object."""
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj and obj[name] is not None:
                return obj[name]
        elif hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _scale_factors(box: Any) -> Tuple[float, float]:
    """Read the light's scale as independent ``(sx, sy)`` factors."""
    scale = _field(box, "scale", default=_MISSING)
    if scale is not _MISSING:
        if isinstance(scale, Mapping):
            sx = float(_field(scale, "x", "sx", default=1.0))
            sy = float(_field(scale, "y", "sy", default=1.0))
        elif isinstance(scale, Sequence) and not isinstance(scale, (str, bytes)):
            values = list(scale)
            if not values:
                sx = sy = 1.0
            else:
                sx = float(values[0])
                sy = float(values[1]) if len(values) > 1 else sx
        else:
            sx = sy = float(scale)
    else:
        sx = float(_field(box, "sx", "scale_x", default=1.0))
        sy = float(_field(box, "sy", "scale_y", default=1.0))
    return abs(sx), abs(sy)


def _rotation_degrees(box: Any) -> float:
    """Return the rotation in degrees (radian fields are converted)."""
    radians = _field(box, "rotation_rad", "angle_rad", "radians", default=_MISSING)
    if radians is not _MISSING:
        return math.degrees(float(radians))
    return float(_field(box, "rotation", "angle", default=0.0))


def box_bounds(box: Any) -> Tuple[float, float, float, float]:
    """Return the axis-aligned ``(left, top, right, bottom)`` screen bounds.

    Rotation is applied around the box centre and scale multiplies the box
    dimensions before rotation, so a rotated or scaled light gets an
    enclosing box that fully covers the light instead of leaking outside the
    unrotated rectangle.
    """
    left = float(_field(box, "x", "left", default=0.0))
    top = float(_field(box, "y", "top", default=0.0))
    width = float(_field(box, "w", "width", default=0.0))
    height = float(_field(box, "h", "height", default=0.0))

    sx, sy = _scale_factors(box)
    width *= sx
    height *= sy

    # Normalise negative extents so callers get a proper rectangle.
    if width < 0:
        left += width
        width = -width
    if height < 0:
        top += height
        height = -height

    angle = _rotation_degrees(box)
    if angle % 360.0 == 0.0:
        return left, top, left + width, top + height

    radians = math.radians(angle)
    cos_a = abs(math.cos(radians))
    sin_a = abs(math.sin(radians))
    half_w = width / 2.0
    half_h = height / 2.0
    center_x = left + half_w
    center_y = top + half_h

    # Extents of a rotated rectangle projected onto the screen axes.
    extent_x = half_w * cos_a + half_h * sin_a
    extent_y = half_w * sin_a + half_h * cos_a
    return (
        center_x - extent_x,
        center_y - extent_y,
        center_x + extent_x,
        center_y + extent_y,
    )


# ---------------------------------------------------------------------------
# Core culling / sorting / clustering
# ---------------------------------------------------------------------------


def tile_range(box: Any, tile_size: float, screen: Screen) -> TileRange:
    """Return ``(x0, y0, x1, y1)`` tiles covered by ``box``.

    The returned range uses exclusive upper bounds, is clipped to the screen,
    and accounts for rotation/scale.  A box ending exactly on a tile boundary
    does not spill into the next tile.  A fully offscreen box yields an empty
    range.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    screen_w, screen_h = float(screen[0]), float(screen[1])
    left, top, right, bottom = box_bounds(box)
    if right <= left or bottom <= top:
        return (0, 0, 0, 0)

    columns = int(math.ceil(screen_w / tile_size)) if screen_w > 0 else 0
    rows = int(math.ceil(screen_h / tile_size)) if screen_h > 0 else 0

    # floor() for the origin, floor(last_covered_pixel) + 1 for the exclusive
    # end.  The epsilon keeps a box that ends exactly on a boundary (or one
    # produced by floating-point rotation) from opening an extra tile.
    x0 = max(0, int(math.floor(left / tile_size + _EPS)))
    y0 = max(0, int(math.floor(top / tile_size + _EPS)))
    x1 = int(math.floor((right - _EPS) / tile_size)) + 1
    y1 = int(math.floor((bottom - _EPS) / tile_size)) + 1

    x0 = min(x0, columns)
    y0 = min(y0, rows)
    x1 = max(x0, min(x1, columns))
    y1 = max(y0, min(y1, rows))
    return (x0, y0, x1, y1)


def sort_lights(lights: Iterable[Any]) -> List[Any]:
    """Rank lights by intensity (desc) then distance (asc).

    Python's sort is stable, so lights with identical scores keep their input
    order.  This makes the truncation result deterministic and independent of
    the order in which lights happened to be created.
    """
    return sorted(
        lights,
        key=lambda light: (
            -float(_field(light, "intensity", "brightness", default=1.0)),
            float(_field(light, "distance", "range", default=1.0)),
        ),
    )


def cull_offscreen(lights: Iterable[Any], screen: Screen) -> List[Any]:
    """Drop lights whose bounds do not overlap the screen at all.

    Lights that merely touch a screen edge are kept; only fully offscreen
    boxes are removed.
    """
    screen_w, screen_h = float(screen[0]), float(screen[1])
    visible: List[Any] = []
    for light in lights:
        left, top, right, bottom = box_bounds(light)
        if right <= left or bottom <= top:
            continue
        if right <= 0 or bottom <= 0 or left >= screen_w or top >= screen_h:
            continue
        visible.append(light)
    return visible


def _light_matches_layer(light: Any, layer: Optional[Hashable]) -> bool:
    """Return whether ``light`` contributes to ``layer``.

    Lights with no layer information affect every layer.  Transparent-pass
    clusters therefore still receive unlayered scene lights while
    layer-restricted lights stay on their own layer.
    """
    if layer is None:
        return True
    raw = _field(light, "layers", "layer", default=None)
    if raw is None:
        return True
    if isinstance(raw, (set, frozenset, list, tuple)):
        return layer in raw
    return raw == layer


def _is_static(light: Any) -> bool:
    return bool(_field(light, "static", "is_static", default=False))


def _clusters_from(
    ordered_lights: Iterable[Any],
    tile_size: float,
    screen: Screen,
    max_lights: Optional[int],
) -> Dict[Tile, List[Any]]:
    """Assign already-ranked lights to their covered tiles.

    Because ``ordered_lights`` is in score order, every tile bucket is also in
    score order; truncation can happen once at the end without re-sorting each
    bucket.
    """
    clusters: Dict[Tile, List[Any]] = {}
    for light in ordered_lights:
        x0, y0, x1, y1 = tile_range(light, tile_size, screen)
        for tile_y in range(y0, y1):
            for tile_x in range(x0, x1):
                clusters.setdefault((tile_x, tile_y), []).append(light)

    if max_lights is not None:
        limit = max(0, int(max_lights))
        for tile, bucket in clusters.items():
            del bucket[limit:]
    return clusters


def assign_lights(
    lights: Iterable[Any],
    tile_size: float,
    screen: Screen,
    max_lights: Optional[int] = DEFAULT_MAX_LIGHTS,
    layer: Optional[Hashable] = None,
) -> Dict[Tile, List[Any]]:
    """Assign each light to every tile its screen bounds overlap.

    Args:
        lights: Light mappings/objects.
        tile_size: Edge length of a cluster in pixels.  Use
            :func:`adaptive_tile_size` to keep the cluster count stable across
            resolutions.
        screen: ``(width, height)`` of the camera viewport in pixels.
        max_lights: Maximum number of lights kept per tile, chosen by
            :func:`sort_lights`.  ``None`` disables the per-tile cap.
        layer: Optional render layer (e.g. ``"transparent"``).  Lights with a
            matching layer or with no layer restriction are clustered.

    Returns:
        A mapping of ``(tile_x, tile_y)`` to the lights affecting that tile,
        brightest first.
    """
    if max_lights is not None:
        max_lights = int(max_lights)

    eligible = [light for light in lights if _light_matches_layer(light, layer)]
    visible = cull_offscreen(eligible, screen)
    return _clusters_from(sort_lights(visible), tile_size, screen, max_lights)


def adaptive_tile_size(
    screen: Screen,
    target_columns: int = 60,
    min_size: int = 8,
    max_size: int = 64,
) -> int:
    """Choose a tile size that keeps the cluster grid roughly constant.

    ``tile_size`` is derived from the longer screen axis so 1080p and 4K
    viewports share a similar tile count instead of quadrupling the cluster
    count at 4K (and wasting precision on tiny distant tiles).
    """
    if target_columns <= 0:
        raise ValueError("target_columns must be positive")
    longest = max(float(screen[0]), float(screen[1]))
    if longest <= 0:
        return int(min_size)
    size = int(round(longest / target_columns))
    size = max(int(min_size), min(int(max_size), size))
    return size


# ---------------------------------------------------------------------------
# Static-light / multi-camera cache
# ---------------------------------------------------------------------------


def _layer_signature(raw: Any) -> Any:
    if isinstance(raw, (set, frozenset)):
        return tuple(sorted(raw, key=repr))
    if isinstance(raw, (list, tuple)):
        return tuple(raw)
    return raw


def _light_signature(light: Any) -> Any:
    """Build a hashable signature for cache invalidation.

    The signature is derived from the resolved bounds (so moving, rotating or
    scaling a light invalidates the cached clusters and its shadow data in the
    same step) plus the ranking and layer fields.
    """
    return (
        _field(light, "name", "id", default=None),
        box_bounds(light),
        float(_field(light, "intensity", "brightness", default=1.0)),
        float(_field(light, "distance", "range", default=1.0)),
        _layer_signature(_field(light, "layers", "layer", default=None)),
    )


@dataclass
class _CameraCache:
    screen: Screen
    tile_size: float
    layer: Optional[Hashable]
    static_signature: Tuple[Any, ...]
    static_clusters: Dict[Tile, List[Any]] = field(default_factory=dict)


class ClusterCache:
    """Per-camera cluster cache with static-light reuse.

    Each camera owns its own entry, so split-screen or picture-in-picture
    cameras no longer share or overwrite one cluster table.  Static lights are
    rebuilt only when their bounds, ranking or layer changes; the cache
    signature covers light movement/removal, which also keeps the shadow
    update in the same frame as the light change instead of leaving a stale
    bright spot behind.

    Dynamic (non-static) lights are merged on every call, and each call
    returns fresh lists so callers cannot mutate cached state.
    """

    def __init__(
        self,
        default_tile_size: int = 32,
        max_lights: Optional[int] = DEFAULT_MAX_LIGHTS,
    ) -> None:
        self.default_tile_size = int(default_tile_size)
        self.max_lights = max_lights
        self._cameras: Dict[Hashable, _CameraCache] = {}

    def clusters(
        self,
        camera: Hashable,
        lights: Iterable[Any],
        screen: Screen,
        *,
        tile_size: Optional[float] = None,
        max_lights: Optional[int] = _MISSING,  # type: ignore[assignment]
        layer: Optional[Hashable] = None,
        force_rebuild: bool = False,
    ) -> Dict[Tile, List[Any]]:
        """Return this camera's clusters for the supplied lights."""
        size = self.default_tile_size if tile_size is None else float(tile_size)
        if size <= 0:
            raise ValueError("tile_size must be positive")
        limit = self.max_lights if max_lights is _MISSING else max_lights
        if limit is not None:
            limit = int(limit)

        lights = list(lights)
        static_lights = [
            light
            for light in lights
            if _is_static(light) and _light_matches_layer(light, layer)
        ]
        dynamic_lights = [
            light
            for light in lights
            if not _is_static(light) and _light_matches_layer(light, layer)
        ]

        static_signature = tuple(_light_signature(light) for light in static_lights)
        entry = self._cameras.get(camera)
        cache_valid = (
            not force_rebuild
            and entry is not None
            and entry.screen == (float(screen[0]), float(screen[1]))
            and entry.tile_size == size
            and entry.layer == layer
            and entry.static_signature == static_signature
        )

        if not cache_valid:
            static_visible = cull_offscreen(static_lights, screen)
            entry = _CameraCache(
                screen=(float(screen[0]), float(screen[1])),
                tile_size=size,
                layer=layer,
                static_signature=static_signature,
                static_clusters=_clusters_from(
                    sort_lights(static_visible), size, screen, None
                ),
            )
            self._cameras[camera] = entry

        # Copy the cached static buckets before merging this frame's dynamics.
        merged: Dict[Tile, List[Any]] = {
            tile: list(bucket) for tile, bucket in entry.static_clusters.items()
        }
        dynamic_visible = cull_offscreen(dynamic_lights, screen)
        for light in sort_lights(dynamic_visible):
            x0, y0, x1, y1 = tile_range(light, size, screen)
            for tile_y in range(y0, y1):
                for tile_x in range(x0, x1):
                    merged.setdefault((tile_x, tile_y), []).append(light)

        if dynamic_visible:
            for tile, bucket in merged.items():
                # Static buckets are already sorted; re-sorting is only needed
                # in tiles that received dynamic lights.
                merged[tile] = sort_lights(bucket)
        if limit is not None:
            for bucket in merged.values():
                del bucket[limit:]
        return merged

    def invalidate(self, camera: Optional[Hashable] = None) -> None:
        """Drop cached clusters for one camera, or all cameras."""
        if camera is None:
            self._cameras.clear()
        else:
            self._cameras.pop(camera, None)
