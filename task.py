import hydra
import numpy as np
from scipy.optimize import minimize, rosen

from utils import save_result


@hydra.main(config_path="config", config_name="task")
def main(config):
    if config.x0 is None:
        rng = np.random.default_rng()
        x0 = rng.uniform(-1, 1, size=2)
    else:
        x0 = np.array(config.x0)

    method = config.method
    result = minimize(
        rosen, x0, method=method.name, jac=config.get("jac"), options=method.options
    )

    save_result(config, result)
    return float(result.fun)


if __name__ == "__main__":
    main()
