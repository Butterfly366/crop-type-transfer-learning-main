"""
为 crop-type-transfer-learning 的 Sentinel2Global 提供显式参数接口。

原始 Sentinel2Global 使用 **kwargs 接收数据路径和年份参数。
较新的 jsonargparse 无法稳定解析 YAML 中的 dict_kwargs，
因此这里将论文配置中需要的参数全部显式写在 __init__ 中，
再传给原始 Sentinel2Global。

数据读取、增强、训练/验证划分和采样逻辑仍由原始类完成。
"""

from collections.abc import Sequence

from torchgeo.datamodules import Sentinel2Global


class Sentinel2GlobalExplicit(Sentinel2Global):
    """可被当前 LightningCLI 正确解析的 Sentinel2Global 包装类。"""

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: int | tuple[int, int] = 256,
        length: int | None = None,
        num_workers: int = 0,
        sentinel2_paths: str = "",
        sentinel2_cache: bool = False,
        sentinel2_bands: Sequence[str] | None = None,
        cdl_paths: str = "",
        cdl_years: Sequence[int] | None = None,
        eurocrops_paths: str = "",
        nccm_paths: str = "",
        nccm_years: Sequence[int] | None = None,
        sact_paths: str = "",
        sas_paths: str = "",
        sas_years: Sequence[int] | None = None,
    ) -> None:
        """
        初始化数据模块。

        所有带前缀的参数最终仍传给原始 Sentinel2Global：
            sentinel2_* → Sentinel2 数据集
            cdl_*       → CDL 数据集
            eurocrops_* → EuroCrops 数据集
            nccm_*      → NCCM 数据集
            sact_*      → South Africa Crop Type 数据集
            sas_*       → South America Soybean 数据集
        """

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

        if cdl_years is None:
            cdl_years = [2023]

        if nccm_years is None:
            nccm_years = [2019]

        if sas_years is None:
            sas_years = [2021]

        super().__init__(
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            num_workers=num_workers,
            sentinel2_paths=sentinel2_paths,
            sentinel2_cache=sentinel2_cache,
            sentinel2_bands=list(sentinel2_bands),
            cdl_paths=cdl_paths,
            cdl_years=list(cdl_years),
            eurocrops_paths=eurocrops_paths,
            nccm_paths=nccm_paths,
            nccm_years=list(nccm_years),
            sact_paths=sact_paths,
            sas_paths=sas_paths,
            sas_years=list(sas_years),
        )
