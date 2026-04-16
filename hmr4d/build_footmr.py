from omegaconf import OmegaConf
from hmr4d import PROJ_ROOT
from hydra.utils import instantiate
from hmr4d.model.footmr.footmr_pl_demo import DemoPL


def build_footmr_demo():
    cfg = OmegaConf.load(PROJ_ROOT / "hmr4d/configs/demo_footmr_model/siga24_release.yaml")
    footmr_demo_pl: DemoPL = instantiate(cfg.model, _recursive_=False)
    footmr_demo_pl.load_pretrained_model(PROJ_ROOT / "inputs/checkpoints/footmr/footmr_siga24_release.ckpt")
    return footmr_demo_pl.eval()
