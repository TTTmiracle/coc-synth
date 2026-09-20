import logging
from pathlib import Path
from random import Random

import pytest

from cocsynth.assets import SpriteLibrary
from cocsynth.catalog import Catalog
from cocsynth.placement import Placer
from cocsynth.render import Renderer

# Placeholder warnings are expected throughout the suite.
logging.getLogger("cocsynth.assets").setLevel(logging.ERROR)

TILE_W = 32


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    return Catalog.load()


@pytest.fixture(scope="session")
def library() -> SpriteLibrary:
    # Deliberately empty: the suite must pass on placeholders alone, so it keeps
    # working before any real art is added.
    return SpriteLibrary(Path("/nonexistent-sprites"), tile_w=TILE_W)


@pytest.fixture(scope="session")
def placer(catalog) -> Placer:
    return Placer(catalog)


@pytest.fixture(scope="session")
def renderer(catalog, library) -> Renderer:
    return Renderer(catalog, library, tile_w=TILE_W)


@pytest.fixture(scope="session")
def sample_label(catalog, placer, renderer):
    """One fully rendered and labelled TH9 base, shared across tests."""
    from cocsynth.labels import build_label

    seed = 20260920
    placement = placer.generate(9, Random(seed))
    rendered = renderer.render(placement.placements, Random(seed))
    return build_label("th9_sample.png", seed, placement, rendered, catalog), placement, rendered
