from collections.abc import Sequence

from torchgeo.datamodules import Sentinel2NCCMDataModule


class Sentinel2NCCMExplicit(Sentinel2NCCMDataModule):

    def __init__(
        self,
        batch_size: int = 2,
        patch_size: int | tuple[int, int] = 256,
        num_workers: int = 0,
        nccm_paths: str = "",
        nccm_years: Sequence[int] | None = None,
        sentinel2_paths: str = "",
        sentinel2_cache: bool = False,
        sentinel2_bands: Sequence[str] | None = None,
    ) -> None:

        if nccm_years is None:
            nccm_years = [2019]

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
            nccm_paths=nccm_paths,
            nccm_years=list(nccm_years),
            sentinel2_paths=sentinel2_paths,
            sentinel2_cache=sentinel2_cache,
            sentinel2_bands=list(sentinel2_bands),
        )
