from collections.abc import Sequence

from torchgeo.datamodules import Sentinel2SouthAmericaSoybeanDataModule


class Sentinel2SASExplicit(
    Sentinel2SouthAmericaSoybeanDataModule
):

    def __init__(
        self,
        batch_size: int = 2,
        patch_size: int | tuple[int, int] = 256,
        num_workers: int = 0,
        south_america_soybean_paths: str = "",
        south_america_soybean_years: Sequence[int] | None = None,
        sentinel2_paths: str = "",
        sentinel2_cache: bool = False,
        sentinel2_bands: Sequence[str] | None = None,
    ) -> None:

        if south_america_soybean_years is None:
            south_america_soybean_years = [2021]

        if sentinel2_bands is None:
            sentinel2_bands = [
                "B01", "B02", "B03", "B04",
                "B05", "B06", "B07", "B08",
                "B8A", "B09", "B10", "B11", "B12",
            ]

        super().__init__(
            batch_size=batch_size,
            patch_size=patch_size,
            num_workers=num_workers,
            south_america_soybean_paths=south_america_soybean_paths,
            south_america_soybean_years=list(
                south_america_soybean_years
            ),
            sentinel2_paths=sentinel2_paths,
            sentinel2_cache=sentinel2_cache,
            sentinel2_bands=list(sentinel2_bands),
        )
