"""
Sentinel2CDLDataModule 的显式参数包装类。

原数据模块通过 **kwargs 接收 Sentinel-2 和 CDL 数据参数。
当前 LightningCLI/jsonargparse 无法正确解析 YAML 中的 dict_kwargs，
因此将需要的参数显式声明。
"""

from collections.abc import Sequence

from torchgeo.datamodules import Sentinel2CDLDataModule


class Sentinel2CDLExplicit(Sentinel2CDLDataModule):
    """可由 LightningCLI 正确解析的 CDL 域内数据模块。"""

    def __init__(
        self,
        batch_size: int = 2,
        patch_size: int | tuple[int, int] = 256,
        num_workers: int = 0,
        cdl_paths: str = "",
        cdl_years: Sequence[int] | None = None,
        sentinel2_paths: str = "",
        sentinel2_cache: bool = False,
        sentinel2_bands: Sequence[str] | None = None,
    ) -> None:

        if cdl_years is None:
            cdl_years = [2023]

        if sentinel2_bands is None:
            sentinel2_bands = [
                "B01",
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B09",
                "B10",
                "B11",
                "B12",
            ]

        super().__init__(
            batch_size=batch_size,
            patch_size=patch_size,
            num_workers=num_workers,
            cdl_paths=cdl_paths,
            cdl_years=list(cdl_years),
            sentinel2_paths=sentinel2_paths,
            sentinel2_cache=sentinel2_cache,
            sentinel2_bands=list(sentinel2_bands),
        )