"""
Sentinel2RasterizedEuroCropsDataModule 的显式参数包装类。

原数据模块通过 **kwargs 接收 Sentinel-2 影像路径和
已经栅格化的 EuroCrops 标签路径。

当前 LightningCLI/jsonargparse 无法正确解析 YAML 中的 dict_kwargs，
因此将训练所需参数显式声明，再传递给原始数据模块。
"""

from collections.abc import Sequence

from torchgeo.datamodules import Sentinel2RasterizedEuroCropsDataModule


class Sentinel2EuroCropsExplicit(
    Sentinel2RasterizedEuroCropsDataModule
):
    """可由 LightningCLI 正确解析的 EuroCrops 域内数据模块。"""

    def __init__(
        self,
        batch_size: int = 2,
        patch_size: int | tuple[int, int] = 256,
        num_workers: int = 0,
        eurocrops_paths: str = "",
        sentinel2_paths: str = "",
        sentinel2_cache: bool = False,
        sentinel2_bands: Sequence[str] | None = None,
    ) -> None:

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
            eurocrops_paths=eurocrops_paths,
            sentinel2_paths=sentinel2_paths,
            sentinel2_cache=sentinel2_cache,
            sentinel2_bands=list(sentinel2_bands),
        )
