import pytest

from render.light_cluster import assign_lights, cull_offscreen, sort_lights, tile_range


SCREEN = (64, 64)


def light(name, x, y, w, h, intensity=1.0, distance=1.0):
    return {
        "name": name,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "intensity": intensity,
        "distance": distance,
    }


def test_tile_range_full_tile():
    assert tile_range({"x": 0, "y": 0, "w": 64, "h": 64}, 32, SCREEN) == (0, 0, 2, 2)


def test_tile_range_offset_tile():
    assert tile_range({"x": 32, "y": 32, "w": 32, "h": 32}, 32, SCREEN) == (1, 1, 2, 2)


def test_tile_range_clamps_negative_origin():
    assert tile_range({"x": -32, "y": 0, "w": 64, "h": 32}, 32, SCREEN) == (0, 0, 1, 1)


def test_tile_range_uses_sixteen_pixel_tiles():
    assert tile_range({"x": 0, "y": 0, "w": 40, "h": 40}, 16, SCREEN) == (0, 0, 3, 3)


def test_small_light_lands_in_one_tile():
    assert list(assign_lights([light("lamp", 0, 0, 32, 32)], 32, SCREEN)) == [(0, 0)]


def test_large_light_lands_in_four_tiles():
    result = assign_lights([light("sun", 0, 0, 64, 64)], 32, SCREEN)
    assert set(result) == {(0, 0), (1, 0), (0, 1), (1, 1)}


def test_max_lights_keeps_brightest():
    lights = [
        light("dim", 0, 0, 32, 32, intensity=1.0),
        light("bright", 0, 0, 32, 32, intensity=5.0),
    ]
    result = assign_lights(lights, 32, SCREEN, max_lights=1)
    assert [item["name"] for item in result[(0, 0)]] == ["bright"]


def test_sort_lights_by_intensity_desc():
    lights = [light("a", 0, 0, 1, 1, intensity=1.0), light("b", 0, 0, 1, 1, intensity=3.0), light("c", 0, 0, 1, 1, intensity=2.0)]
    assert [item["name"] for item in sort_lights(lights)] == ["b", "c", "a"]


def test_sort_lights_by_distance_asc_when_intensity_ties():
    lights = [light("far", 0, 0, 1, 1, distance=3.0), light("close", 0, 0, 1, 1, distance=1.0), light("mid", 0, 0, 1, 1, distance=2.0)]
    assert [item["name"] for item in sort_lights(lights)] == ["close", "mid", "far"]


def test_cull_removes_offscreen_light():
    assert cull_offscreen([light("off", 100, 100, 10, 10)], SCREEN) == []


def test_cull_keeps_only_visible_light():
    lights = [light("off", 100, 100, 10, 10), light("on", 10, 10, 10, 10)]
    result = cull_offscreen(lights, SCREEN)
    assert [item["name"] for item in result] == ["on"]


def test_assign_respects_max_lights_per_tile():
    lights = [
        light("a", 0, 0, 32, 32, intensity=3.0),
        light("b", 0, 0, 32, 32, intensity=2.0),
        light("c", 0, 0, 32, 32, intensity=1.0),
    ]
    result = assign_lights(lights, 32, SCREEN, max_lights=2)
    assert len(result[(0, 0)]) == 2


def test_assign_light_on_tile_boundary():
    result = assign_lights([light("edge", 32, 0, 1, 1)], 32, SCREEN)
    assert set(result) == {(1, 0)}


def test_sort_is_stable_for_equal_scores():
    lights = [light("first", 0, 0, 1, 1, intensity=1.0, distance=1.0), light("second", 0, 0, 1, 1, intensity=1.0, distance=1.0)]
    assert [item["name"] for item in sort_lights(lights)] == ["first", "second"]


def test_cull_keeps_edge_touching_light():
    lights = [light("edge", 63, 63, 2, 2), light("off", 100, 100, 2, 2)]
    result = cull_offscreen(lights, SCREEN)
    assert [item["name"] for item in result] == ["edge"]
