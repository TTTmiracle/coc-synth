"""Assembling the ground-truth label for one rendered base.

Instances the renderer dropped (fully or almost entirely hidden behind something
nearer) are deliberately left out. Asking a model to detect a building that is not
visible in the image teaches it to hallucinate; `min_visibility` on the renderer is
the knob that decides where that line sits.
"""
from __future__ import annotations

import json
from pathlib import Path

from .catalog import Catalog
from .placement import PlacementResult
from .project import Projection
from .render import RenderResult
from .schema import BaseLabel, BuildingLabel, GridMeta, heading_of, rotation_deg_of


def build_label(
    image_name: str,
    seed: int,
    placement: PlacementResult,
    render: RenderResult,
    catalog: Catalog,
) -> BaseLabel:
    """Combine placement facts and measured pixel geometry into one label."""
    proj: Projection = render.projection
    by_id = {p.instance_id: p for p in placement.placements}

    buildings: list[BuildingLabel] = []
    for iid, inst in sorted(render.instances.items()):
        p = by_id[iid]
        buildings.append(
            BuildingLabel(
                id=iid,
                type=p.type_id,
                level=p.level,
                tile=p.tile,
                footprint=p.footprint,
                direction=p.direction,
                heading=heading_of(p.direction, p.directions),
                rotation_deg=rotation_deg_of(p.direction, p.directions),
                bbox_px=inst.bbox_px,
                footprint_polygon_px=inst.polygon_px,
                visibility=inst.visibility,
                sprite=inst.sprite_name,
            )
        )

    return BaseLabel(
        image=image_name,
        image_size=render.image.size,
        town_hall_level=placement.town_hall_level,
        seed=seed,
        grid=GridMeta(
            tiles=catalog.grid_tiles,
            tile_w=proj.tile_w,
            tile_h=proj.tile_h,
            origin_px=proj.origin,
        ),
        buildings=buildings,
    )


def write_label(label: BaseLabel, path: Path) -> None:
    """Write the label JSON next to its image."""
    path.write_text(
        json.dumps(label.model_dump(mode="json"), indent=1, ensure_ascii=False),
        encoding="utf-8",
    )


def read_label(path: Path) -> BaseLabel:
    """Read and validate a label file."""
    return BaseLabel.model_validate_json(Path(path).read_text(encoding="utf-8"))
