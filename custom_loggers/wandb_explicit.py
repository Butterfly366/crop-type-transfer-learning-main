"""
为旧版 LightningCLI/jsonargparse 提供显式参数的 WandbLogger。

原始 WandbLogger 的复杂类型注解可能导致 jsonargparse
在解析 class_path 时出现 "__args__" 错误。
"""

from lightning.pytorch.loggers import WandbLogger


class WandbLoggerExplicit(WandbLogger):
    """显式参数版本的 WandbLogger。"""

    def __init__(
        self,
        project: str,
        name: str,
        save_dir: str = ".",
        offline: bool = False,
        log_model: bool = False,
    ) -> None:
        super().__init__(
            project=project,
            name=name,
            save_dir=save_dir,
            offline=offline,
            log_model=log_model,
        )
